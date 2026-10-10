"""Chapter 3 drop-in model; inherits the Chapter 1 forward/cache/checkpoint path."""

from collections import Counter
from dataclasses import dataclass, asdict
import importlib
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
reference = importlib.import_module("chapters.02_performance_model.code.model")


@dataclass(frozen=True)
class Optimizations:
    rmsnorm: bool = True
    swiglu: bool = True
    qk_rope: bool = True
    prefill_attention: bool = True
    decode_attention: bool = True
    key_tile: int = 32
    attention_warps: int = 4
    activation_block: int = 256

    def __post_init__(self):
        if (
            self.key_tile <= 0
            or self.attention_warps not in (1, 2, 4, 8)
            or self.activation_block not in (128, 256, 512)
        ):
            raise ValueError("Invalid tile/warp/block configuration")


PRESETS = {
    "optimized": Optimizations(),
    "fusion_only": Optimizations(prefill_attention=False, decode_attention=False),
    "attention_only": Optimizations(rmsnorm=False, swiglu=False, qk_rope=False),
}


def kernels():
    return importlib.import_module("chapters.03_kernels.code.cute_kernels")


class Dispatch:
    def supported(self, name, *xs):
        reason = None
        if self.c.head_dim != 128:
            reason = "head_dim must be 128"
        elif any(
            x.device.type != "cuda" or x.dtype != torch.bfloat16 or x.stride(-1) != 1
            for x in xs
        ):
            reason = "CUDA BF16 with unit last stride required"
        if (
            reason is None
            and name == "swiglu"
            and any(not x.is_contiguous() for x in xs)
        ):
            reason = "SwiGLU requires contiguous projection outputs"
        if reason is None and name == "rmsnorm":
            for x in xs:
                leading = [i for i, size in enumerate(x.shape[:-1]) if size > 1]
                if any(
                    x.stride(a) != x.stride(b) * x.shape[b]
                    for a, b in zip(leading, leading[1:])
                ):
                    reason = "RMSNorm leading dimensions must flatten into rows"
        if reason is not None:
            if self.strict:
                raise RuntimeError(f"Unexpected {name} fallback: {reason}")
            self.fallbacks[f"{name}: {reason}"] += 1
            return False
        self.dispatches[name] += 1
        return True


class OptimizedBlock(Dispatch, reference.Block):
    def normalize(self, norm, x):
        if self.optimizations.rmsnorm and self.supported("rmsnorm", x):
            return kernels().rmsnorm(x, norm.weight, norm.eps)
        return super().normalize(norm, x)

    def activation(self, gate, up):
        if self.optimizations.swiglu and self.supported("swiglu", gate, up):
            return kernels().swiglu(gate, up, self.optimizations.activation_block)
        return super().activation(gate, up)

    def normalize_rotary(self, q, k, positions):
        if self.optimizations.qk_rope and self.supported("qk_rope", q, k):
            return kernels().qk_norm_rope(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                self.c.eps,
                self.c.theta,
            )
        return super().normalize_rotary(q, k, positions)

    def attention(self, q, k, v, positions):
        name = "decode_attention" if q.shape[2] == 1 else "prefill_attention"
        if getattr(self.optimizations, name) and self.supported(name, q, k, v):
            return kernels().attention(
                q, k, v, self.optimizations.key_tile, self.optimizations.attention_warps
            )
        return super().attention(q, k, v, positions)


class OptimizedQwen3(Dispatch, reference.TinyQwen3):
    """forward(token_ids, kv_cache=None, decode=False) -> (logits, new cache).

    Parameters and caller-owned [B,Hkv,P,128] cache match Chapter 1 exactly.
    decode selects last logits; attention dispatch depends on input length.
    No paging, in-place cache mutation, hidden persistent state, or GEMM changes.
    strict=True rejects unsupported optimized dispatch (compilation errors always
    propagate). strict=False records explicit reference fallbacks. Disabled flags
    intentionally select reference operations and are not unexpected fallbacks.
    """

    block_class = OptimizedBlock

    def __init__(self, config, optimizations=None, strict=True):
        super().__init__(config)
        self.optimizations = (
            optimizations
            if isinstance(optimizations, Optimizations)
            else Optimizations(**(optimizations or {}))
        )
        self.strict = strict
        self.fallbacks = Counter()
        self.dispatches = Counter()
        for layer in self.layers:
            layer.optimizations = self.optimizations
            layer.strict = strict
            layer.fallbacks = self.fallbacks
            layer.dispatches = self.dispatches

    def normalize_final(self, x):
        if self.optimizations.rmsnorm and self.supported("rmsnorm", x):
            return kernels().rmsnorm(x, self.norm.weight, self.norm.eps)
        return super().normalize_final(x)

    def execution_metadata(self):
        return dict(
            optimizations=asdict(self.optimizations),
            strict=self.strict,
            fallbacks=dict(self.fallbacks),
            dispatches=dict(self.dispatches),
        )


def load_checkpoint(
    snapshot,
    device="cuda:0",
    dtype=torch.bfloat16,
    *,
    optimizations=None,
    strict=True,
    audit_only=False,
):
    model, inventory = reference.load_custom(
        snapshot,
        device,
        dtype,
        audit_only,
        model_factory=lambda c: OptimizedQwen3(c, optimizations, strict),
        name_map_factory=reference.model_weight_name_mapping,
    )
    if model is not None:
        _, metadata = reference.configuration(snapshot)
        model.max_position_embeddings = metadata.get("max_position_embeddings", 32768)
    return model, inventory
