"""Shared Chapter 2/3 analytical models, CUDA measurements, and plots.

Extracted from Chapter 2: one source of truth, usable without notebook state.
Compulsory bytes are a model, never hardware-counter traffic.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATHS = {}


@dataclass(frozen=True)
class Hardware:
    name: str
    bandwidth_gbps: float
    dense_peak_tflops: dict
    compute_basis: str  # 'assumed' or 'documented'
    compute_source: str
    bandwidth_source: str

    def peak(self, precision):
        value = self.dense_peak_tflops[precision]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "Provide a finite, positive dense peak for this precision."
            )
        if not math.isfinite(self.bandwidth_gbps) or self.bandwidth_gbps <= 0:
            raise ValueError("Bandwidth must be finite and positive.")
        if self.compute_basis not in {"assumed", "documented"}:
            raise ValueError("Compute basis must be assumed or documented.")
        return value

    def ceiling(self, intensity, precision):
        return np.minimum(
            self.peak(precision), np.asarray(intensity) * self.bandwidth_gbps / 1000
        )


def roofline_ms(flops, byte_count, tflops, gbps, launch_us=0):
    if not all(math.isfinite(x) for x in (flops, byte_count, tflops, gbps, launch_us)):
        raise ValueError("Finite inputs required.")
    if min(flops, byte_count, launch_us) < 0 or min(tflops, gbps) <= 0:
        raise ValueError("Nonnegative work and positive rates required.")
    return (
        1000 * max(flops / (tflops * 1e12), byte_count / (gbps * 1e9))
        + launch_us / 1000
    )


def gemm(m, k, n, element_bytes=2):
    if min(m, k, n, element_bytes) <= 0:
        raise ValueError("Positive GEMM dimensions and element size required.")
    return 2 * m * k * n, element_bytes * (m * k + k * n + m * n)


@dataclass(frozen=True)
class ModelDimensions:
    layers: int
    hidden: int
    intermediate: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocab: int

    @property
    def query_width(self):
        return self.query_heads * self.head_dim

    @property
    def kv_width(self):
        return self.kv_heads * self.head_dim

    @property
    def layer_matrix_parameters(self):
        d, q, k, i = self.hidden, self.query_width, self.kv_width, self.intermediate
        return 2 * d * q + 2 * d * k + 3 * d * i

    @property
    def parameters(self):
        # Two residual-stream RMSNorms plus shared per-head Q/K norm weights per layer.
        return (
            self.layers
            * (self.layer_matrix_parameters + 2 * self.hidden + 2 * self.head_dim)
            + 2 * self.vocab * self.hidden
            + self.hidden
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            *(
                getattr(config, field)
                for field in (
                    "num_hidden_layers",
                    "hidden_size",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "vocab_size",
                )
            )
        )


MODELS = {
    "8b": ModelDimensions(36, 4096, 12288, 32, 8, 128, 151936),
    "32b": ModelDimensions(64, 5120, 25600, 64, 8, 128, 151936),
}
REVISIONS = {
    "8b": "b968826d9c46dd6066d109eabc6255188de91218",
    "32b": "9216db5781bf21249d130ec9da846c4624c16137",
}
REPOS = {"8b": "Qwen/Qwen3-8B", "32b": "Qwen/Qwen3-32B"}


def decoder_layer_flops(
    c, batch, new_tokens, prefix_tokens=0, attention_count="causal"
):
    if min(batch, new_tokens) <= 0 or prefix_tokens < 0:
        raise ValueError("Positive batch/new_tokens and nonnegative prefix required.")
    m = batch * new_tokens
    if attention_count == "causal":
        pairs = batch * (
            prefix_tokens * new_tokens + new_tokens * (new_tokens + 1) // 2
        )
    elif attention_count == "dense":
        pairs = batch * new_tokens * (prefix_tokens + new_tokens)
    else:
        raise ValueError("attention_count must be causal or dense")
    d, q, k, i = c.hidden, c.query_width, c.kv_width, c.intermediate
    return {
        "q_proj": 2 * m * d * q,
        "k_proj": 2 * m * d * k,
        "v_proj": 2 * m * d * k,
        "qk": 2 * pairs * q,
        "av": 2 * pairs * q,
        "o_proj": 2 * m * q * d,
        "gate_proj": 2 * m * d * i,
        "up_proj": 2 * m * d * i,
        "down_proj": 2 * m * i * d,
    }


def model_work(
    c,
    batch,
    new_tokens,
    prefix_tokens=0,
    logits_tokens=1,
    element_bytes=2,
    attention_count="causal",
):
    if not 1 <= logits_tokens <= new_tokens or element_bytes <= 0:
        raise ValueError(
            "Logits positions must be in [1, new_tokens]; element size must be positive."
        )
    ops = decoder_layer_flops(c, batch, new_tokens, prefix_tokens, attention_count)
    flops = (
        c.layers * sum(ops.values()) + 2 * batch * logits_tokens * c.hidden * c.vocab
    )
    # Compulsory-data proxy, not a DRAM counter or a complete operator-traffic model.
    # Weights once per forward, selected embedding rows, old KV read and new KV written once.
    weight_bytes = element_bytes * (
        c.layers * (c.layer_matrix_parameters + 2 * c.hidden + 2 * c.head_dim)
        + c.hidden
        + c.hidden * c.vocab
    )
    embedding_bytes = element_bytes * batch * new_tokens * c.hidden
    kv_read_bytes = element_bytes * 2 * c.layers * batch * prefix_tokens * c.kv_width
    kv_write_bytes = element_bytes * 2 * c.layers * batch * new_tokens * c.kv_width
    io_bytes = element_bytes * (
        batch * new_tokens * c.hidden + batch * logits_tokens * c.vocab
    )
    byte_count = (
        weight_bytes + embedding_bytes + kv_read_bytes + kv_write_bytes + io_bytes
    )
    return dict(
        flops=flops,
        bytes_proxy=byte_count,
        intensity=flops / byte_count,
        weight_bytes=weight_bytes,
        kv_read_bytes=kv_read_bytes,
        kv_write_bytes=kv_write_bytes,
    )


def metrics(work, seconds):
    """Arithmetic throughput for the roofline diagnostic, independent of peak inputs."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Positive finite measured seconds required.")
    return dict(
        achieved_tflops=work["flops"] / seconds / 1e12,
        bandwidth_proxy_gbps=work["bytes_proxy"] / seconds / 1e9,
    )


def predict_tradeoffs(c, batches, context, hw, precision):
    """Ideal aggregate roofline predictions at one fixed context, not measured latency."""
    if precision not in {"bfloat16", "float16"}:
        raise ValueError("This prediction uses two-byte BF16/FP16 weights and KV.")
    if context <= 0 or not batches or any(b <= 0 for b in batches):
        raise ValueError(
            "Positive context and a nonempty list of positive batch sizes required."
        )
    rows = []
    for batch in batches:
        for phase, tokens, prefix in [("prefill", context, 0), ("decode", 1, context)]:
            work = model_work(c, batch, tokens, prefix)
            compute_ms = work["flops"] / (hw.peak(precision) * 1e9)
            memory_ms = work["bytes_proxy"] / (hw.bandwidth_gbps * 1e6)
            latency_ms = roofline_ms(
                work["flops"],
                work["bytes_proxy"],
                hw.peak(precision),
                hw.bandwidth_gbps,
            )
            rows.append(
                dict(
                    batch=batch,
                    context=context,
                    phase=phase,
                    prefix_tokens=prefix,
                    attended_positions=prefix + tokens,
                    flops=work["flops"],
                    bytes_proxy=work["bytes_proxy"],
                    compute_ms=compute_ms,
                    memory_ms=memory_ms,
                    limiting_term="compute" if compute_ms >= memory_ms else "memory",
                    latency_ms=latency_ms,
                    total_tokens_per_second=batch * tokens * 1000 / latency_ms,
                    token_kind="input" if phase == "prefill" else "output",
                    interactivity_tokens_per_second=1000 / latency_ms
                    if phase == "decode"
                    else None,
                )
            )
    return rows


def plot_tradeoffs(
    predictions, context, precision, compute_basis, observed=None, compact_labels=False
):
    """Use identical axes and analytical curves in Sections 2 and 5."""
    predicted = [r for r in predictions if r["context"] == context]
    if not predicted:
        raise ValueError(
            "No frozen analytical predictions at this context; rerun Section 2."
        )
    models = list(dict.fromkeys(r["model"] for r in predicted))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for model_index, key in enumerate(models):
        for ax, phase, x_field in zip(
            axes,
            ["prefill", "decode"],
            ["latency_ms", "interactivity_tokens_per_second"],
        ):
            points = sorted(
                (r for r in predicted if r["model"] == key and r["phase"] == phase),
                key=lambda r: r["batch"],
            )
            (line,) = ax.plot(
                [r[x_field] for r in points],
                [r["total_tokens_per_second"] for r in points],
                marker="o",
                markerfacecolor="none",
                linestyle="--",
                label=f"Qwen3-{key.upper()} analytical",
            )
            color = line.get_color()
            for i, r in enumerate(points):
                offset = (
                    (10 if (i + model_index) % 2 == 0 else -16)
                    if phase == "prefill"
                    else 0
                )
                ax.annotate(
                    f"B={r['batch']}",
                    (r[x_field], r["total_tokens_per_second"]),
                    xytext=(4, offset),
                    textcoords="offset points",
                    fontsize=8,
                    va="center",
                    color=color,
                )
            measured = [
                r
                for r in (observed or [])
                if r["model"] == key and r["context"] == context and r["phase"] == phase
            ]
            for i, r in enumerate(measured):
                x, y = r[x_field], r["total_tokens_per_second"]
                ax.errorbar(
                    x,
                    y,
                    xerr=[[x - r[x_field + "_min"]], [r[x_field + "_max"] - x]],
                    yerr=[
                        [y - r["total_tokens_per_second_min"]],
                        [r["total_tokens_per_second_max"] - y],
                    ],
                    fmt="s"
                    if r.get("implementation", "baseline") == "baseline"
                    else "^",
                    markersize=7,
                    capsize=3,
                    color=color,
                    label=f"Qwen3-{key.upper()} {r.get('implementation', 'baseline')} measured"
                    if not any(
                        p.get("implementation", "baseline")
                        == r.get("implementation", "baseline")
                        for p in measured[:i]
                    )
                    else None,
                )
                ax.annotate(
                    f"B={r['batch']}" if compact_labels else f"B={r['batch']} measured",
                    (x, y),
                    xytext=(6, -14)
                    if compact_labels
                    and r.get("implementation", "baseline") != "baseline"
                    else (6, 7),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
                match = next((p for p in points if p["batch"] == r["batch"]), None)
                if match is not None:
                    ax.plot(
                        [match[x_field], x],
                        [match["total_tokens_per_second"], y],
                        linestyle=":",
                        color=color,
                        alpha=0.4,
                    )
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.2)
            ax.margins(x=0.14, y=0.22)
            ax.legend(fontsize=8)
    axes[0].set(
        xscale="log",
        xlabel="Local TTFT (ms; analytical curve is a lower bound)",
        ylabel="Total prefill input throughput (tok/s)",
        title=f"Prefill · fixed prompt S={context}",
    )
    axes[1].set(
        xlabel="Interactivity (output tok/s per user)",
        ylabel="Total decode output throughput (tok/s)",
        title=f"Decode · fixed prefix S={context}",
    )
    scope = (
        "Analytical predictions only"
        if observed is None
        else "Analytical predictions + real benchmark points"
        if observed
        else "Analytical predictions · matching benchmarks unmeasured"
    )
    fig.suptitle(f"{scope} · {precision} · {compute_basis} compute ceiling")
    return fig, axes


def require_cuda(device, precision):
    import torch

    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("CUDA required; CPU timing fallback is disabled.")
    if precision not in {"bfloat16", "float16"}:
        raise ValueError("This benchmark supports bfloat16 and float16 only.")
    torch.cuda.set_device(device)
    # Chapter 1 attention matmuls use FP32 even with BF16/FP16 weights.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16.")
    return torch


def cuda_samples(fn, repeats=3, inner=10, warmup=5, device="cuda:0"):
    import torch

    if min(repeats, inner) <= 0 or warmup < 0:
        raise ValueError(
            "Positive repeat/inner counts and nonnegative warmup required."
        )
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    with torch.inference_mode(), torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start.record()
            for _ in range(inner):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / inner)
    return samples


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def calibrate(device, precision, repeats, out_dir):
    torch = require_cuda(device, precision)
    dtype = getattr(torch, precision)
    # 256 MiB per buffer; increase if necessary to exceed your GPU's cache.
    source = torch.ones(64 * 1024 * 1024, device=device, dtype=torch.float32)
    dest = torch.empty_like(source)
    copy_bytes = 2 * source.numel() * source.element_size()
    copies = cuda_samples(
        lambda dest=dest, source=source: dest.copy_(source), repeats, device=device
    )
    gbps = copy_bytes / (statistics.median(copies) * 1e6)
    del source, dest
    small = torch.ones(1, 1, device=device, dtype=dtype)
    small_out = torch.empty_like(small)
    floor_ms = min(
        cuda_samples(
            lambda: torch.mm(small, small, out=small_out),
            repeats,
            inner=50,
            device=device,
        )
    )
    calibration, rates = [], []
    for m in [1, 16, 512]:
        x = torch.randn(m, 4096, device=device, dtype=dtype)
        w = torch.randn(4096, 4096, device=device, dtype=dtype)
        y = torch.empty(m, 4096, device=device, dtype=dtype)
        samples = cuda_samples(lambda: torch.mm(x, w, out=y), repeats, device=device)
        rates.append(gemm(m, 4096, 4096)[0] / (statistics.median(samples) * 1e9))
        calibration.extend(
            dict(m=m, repeat=r, cuda_ms=s) for r, s in enumerate(samples)
        )
    effective_tflops = max(rates)
    predicted = {
        m: roofline_ms(*gemm(m, 4096, 4096), effective_tflops, gbps, floor_ms * 1000)
        for m in [2, 4, 8, 32, 64, 128]
    }
    result = dict(
        effective_tflops=effective_tflops,
        copy_gbps=gbps,
        small_call_floor_ms=floor_ms,
        predicted_gemm_ms=predicted,
        copy_buffer_bytes=copy_bytes // 2,
    )
    write_json(
        out_dir / "calibration_prediction.json", result
    )  # Before held-out timings.
    write_csv(out_dir / "calibration.csv", calibration)
    write_csv(
        out_dir / "bandwidth.csv",
        [dict(repeat=r, bytes=copy_bytes, cuda_ms=s) for r, s in enumerate(copies)],
    )
    measured = []
    for m in predicted:
        x = torch.randn(m, 4096, device=device, dtype=dtype)
        w = torch.randn(4096, 4096, device=device, dtype=dtype)
        y = torch.empty(m, 4096, device=device, dtype=dtype)
        samples = cuda_samples(lambda: torch.mm(x, w, out=y), repeats, device=device)
        measured.extend(
            dict(m=m, repeat=r, predicted_ms=predicted[m], cuda_ms=s)
            for r, s in enumerate(samples)
        )
    write_csv(out_dir / "gemm_results.csv", measured)
    return result, measured


def resolve_checkpoint(key):
    if key in LOCAL_MODEL_PATHS:
        path = Path(LOCAL_MODEL_PATHS[key]).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")
        return str(path), {}
    local = ROOT / "models" / f"qwen3-{key}" / REVISIONS[key]
    if (local / "config.json").is_file():
        return str(local), {}
    return REPOS[key], {"revision": REVISIONS[key]}


def reference_model_module():
    # Import by package path so this works from the course root or notebook folder.
    import importlib
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("chapters.02_performance_model.code.model")


def load_resident_model(snapshot, device, precision, backend):
    """Load the Chapter 1 implementation through its strict checkpoint audit."""
    import torch

    if backend != "eager":
        raise ValueError("The Chapter 1 reference supports only eager attention.")
    reference = reference_model_module()
    model, _ = reference.load_checkpoint(snapshot, device, getattr(torch, precision))
    return model


def check_capacity(c, device, cases, decode_steps):
    import torch

    free, total = torch.cuda.mem_get_info(device)
    # GB10 shares host/device RAM; MemAvailable includes reclaimable file cache.
    available = free
    if "GB10" in torch.cuda.get_device_name(device) and Path("/proc/meminfo").is_file():
        info = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        available = min(total, int(info["MemAvailable"].split()[0]) * 1024)
    weights = 2 * c.parameters
    max_kv = max(4 * c.layers * b * (s + decode_steps) * c.kv_width for b, s in cases)
    # Conservative working allowance, not an exact activation-capacity model.
    reserve = max(8 * 1024**3, weights * 0.15) + max_kv
    if weights + reserve > available:
        raise torch.cuda.OutOfMemoryError(
            f"Capacity preflight: need about {(weights + reserve) / 1e9:.1f} GB including reserve; "
            f"{available / 1e9:.1f} GB available. Reduce work or use a larger GPU."
        )
    return dict(available_bytes=available, weight_bytes=weights, reserve_bytes=reserve)


def load_qwen(key, device, precision, backend, cases, decode_steps):
    torch = require_cuda(device, precision)
    from transformers import AutoConfig
    from huggingface_hub import snapshot_download

    # Loading preflight uses the smallest KV case; larger cases can be marked unmeasured individually.
    smallest_case = min(cases, key=lambda case: case[0] * (case[1] + decode_steps))
    capacity = check_capacity(MODELS[key], device, [smallest_case], decode_steps)
    source, kwargs = resolve_checkpoint(key)
    config = AutoConfig.from_pretrained(source, **kwargs)
    if ModelDimensions.from_config(config) != MODELS[key] or config.tie_word_embeddings:
        raise ValueError(
            "Checkpoint dimensions/tied weights differ from the analytical model."
        )
    if getattr(config, "quantization_config", None) or getattr(
        config, "use_sliding_window", False
    ):
        raise ValueError("This lab expects dense, unquantized, full-attention Qwen3.")
    print(
        f"Loading {key} from {source}; CUDA free/total GB:",
        [round(x / 1e9, 2) for x in torch.cuda.mem_get_info(device)],
        flush=True,
    )
    if kwargs:
        source = snapshot_download(
            repo_id=source,
            revision=kwargs["revision"],
            allow_patterns=["*.json", "*.safetensors"],
        )
    model = load_resident_model(source, device, precision, backend)
    if any(
        p.device != torch.device(device) or p.dtype != getattr(torch, precision)
        for p in model.parameters()
    ):
        raise RuntimeError(
            "All parameters must reside on the target GPU in the selected precision."
        )
    if sum(p.numel() for p in model.parameters()) != MODELS[key].parameters:
        raise ValueError("Actual parameter count differs from the model ledger.")
    metadata = dict(
        model=key,
        source=source,
        capacity_preflight=capacity,
        requested_revision=kwargs.get("revision"),
        resolved_revision=getattr(config, "_commit_hash", None),
        config_sha256=hashlib.sha256(
            (Path(source) / "config.json").read_bytes()
        ).hexdigest(),
        implementation="Chapter 1 TinyQwen3",
        attention_backend="eager",
        source_sha256={
            **reference_model_module().source_hashes(),
            "shared/performance.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    return model, metadata


def check_cached_logits(model, device):
    import torch

    with torch.inference_mode():
        ids = torch.tensor([[11, 25, 76, 93]], device=device)
        oracle, _ = model(ids, decode=True)
        _, prefix = model(ids[:, :-1], decode=True)
        cached, _ = model(ids[:, -1:], prefix, decode=True)
        # Freeze this BF16 tolerance before accepting benchmark data; do not loosen on failure.
        torch.testing.assert_close(cached.float(), oracle.float(), atol=0.25, rtol=0.02)
        return float((cached.float() - oracle.float()).abs().max().item())


def benchmark_model(
    model,
    key,
    cases,
    repeats,
    warmup_runs,
    decode_steps,
    device,
    precision,
    implementation="baseline",
    forced_ids=None,
    compilation_counter=None,
):
    torch = require_cuda(device, precision)
    if min(repeats, decode_steps) <= 0 or warmup_runs < 1:
        raise ValueError(
            "Positive repeats/decode steps and at least one full warmup run required."
        )
    rows = []
    c = MODELS[key]
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for batch, prompt in cases:
        if (
            min(batch, prompt) <= 0
            or prompt + decode_steps > model.max_position_embeddings
        ):
            raise ValueError("Workload outside configured context range.")
        generator = torch.Generator(device=device).manual_seed(42)
        ids = torch.randint(
            100, 10000, (batch, prompt), device=device, generator=generator
        )
        with torch.inference_mode():
            for repeat in range(-warmup_runs, repeats):
                if repeat == 0 and compilation_counter is not None:
                    compiled_before = compilation_counter()
                cache = None
                next_ids = ids
                for step in range(-1, decode_steps):
                    phase = "prefill" if step == -1 else "decode"
                    prefix = 0 if step == -1 else prompt + step
                    expected_cache = prefix + next_ids.shape[1]
                    # Synchronized wall interval runs through greedy token availability.
                    # Events bracket forward only, not token selection or the final host wait.
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    start.record()
                    logits, updated_cache = model(next_ids, cache, decode=True)
                    end.record()
                    next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if forced_ids is not None:
                        next_ids = forced_ids[:, step + 1 : step + 2]
                    torch.cuda.synchronize(device)
                    wall_ms = (time.perf_counter() - begin) * 1000
                    cuda_ms = start.elapsed_time(end)
                    cache = updated_cache
                    if any(
                        k.shape[2] != expected_cache or v.shape[2] != expected_cache
                        for k, v in cache
                    ):
                        raise AssertionError(
                            "Cache length does not match the modeled prefix."
                        )
                    if repeat >= 0:
                        if (
                            compilation_counter is not None
                            and compilation_counter() != compiled_before
                        ):
                            raise RuntimeError(
                                "JIT compilation occurred during a measured repeat"
                            )
                        work = model_work(c, batch, prompt if step == -1 else 1, prefix)
                        rows.append(
                            dict(
                                model=key,
                                implementation=implementation,
                                batch=batch,
                                prompt=prompt,
                                phase=phase,
                                repeat=repeat,
                                step=max(step, 0),
                                prefix_tokens=prefix,
                                attended_positions=expected_cache,
                                cuda_ms=cuda_ms,
                                token_ready_ms=wall_ms,
                                throughput_tokens=batch * prompt
                                if phase == "prefill"
                                else batch,
                                throughput_token_kind="input"
                                if phase == "prefill"
                                else "output",
                                **work,
                                **metrics(work, cuda_ms / 1000),
                            )
                        )
                    del logits, updated_cache
                del cache, next_ids
    return rows


def summarize_model_rows(rows):
    summary = []
    groups = sorted(
        {
            (
                r.get("implementation", "baseline"),
                r["model"],
                r["batch"],
                r["prompt"],
                r["phase"],
            )
            for r in rows
        }
    )
    for implementation, key, batch, prompt, phase in groups:
        group = [
            r
            for r in rows
            if (
                r.get("implementation", "baseline"),
                r["model"],
                r["batch"],
                r["prompt"],
                r["phase"],
            )
            == (implementation, key, batch, prompt, phase)
        ]
        latency, throughput, forward, achieved = [], [], [], []
        for repeat in sorted({r["repeat"] for r in group}):
            run = [r for r in group if r["repeat"] == repeat]
            wall_ms = sum(r["token_ready_ms"] for r in run)
            cuda_ms = sum(r["cuda_ms"] for r in run)
            if any(
                not math.isfinite(r[field]) or r[field] <= 0
                for r in run
                for field in ("token_ready_ms", "cuda_ms")
            ):
                raise ValueError("Positive finite timing observations required.")
            latency.append(wall_ms / len(run))
            throughput.append(sum(r["throughput_tokens"] for r in run) * 1000 / wall_ms)
            forward.append(cuda_ms / len(run))
            achieved.append(sum(r["flops"] for r in run) / (cuda_ms * 1e9))
        summary.append(
            dict(
                model=key,
                implementation=implementation,
                batch=batch,
                prompt=prompt,
                phase=phase,
                latency_metric="local TTFT" if phase == "prefill" else "mean local TBT",
                throughput_token_kind="input" if phase == "prefill" else "output",
                repeats=len(latency),
                latency_ms=statistics.median(latency),
                latency_min_ms=min(latency),
                latency_max_ms=max(latency),
                tokens_per_second=statistics.median(throughput),
                tokens_per_second_min=min(throughput),
                tokens_per_second_max=max(throughput),
                prefix_tokens_min=min(r["prefix_tokens"] for r in group),
                prefix_tokens_max=max(r["prefix_tokens"] for r in group),
                forward_ms=statistics.median(forward),
                intensity=sum(r["flops"] for r in group)
                / sum(r["bytes_proxy"] for r in group),
                achieved_tflops=statistics.median(achieved),
                min_tflops=min(achieved),
                max_tflops=max(achieved),
            )
        )
    return summary


def summarize_tradeoff_observations(rows, context):
    """Match the fixed prompt/first decode prefix used by the analytical curves."""
    matched = [
        r
        for r in rows
        if r["prompt"] == context
        and (
            (r["phase"] == "prefill" and r["prefix_tokens"] == 0)
            or (r["phase"] == "decode" and r["prefix_tokens"] == context)
        )
    ]
    points = []
    for r in summarize_model_rows(matched):
        decode = r["phase"] == "decode"
        # Median(rate) is not necessarily 1/median(time) for an even repeat count.
        # Derive interactivity from the same per-repeat throughput aggregates.
        points.append(
            dict(
                model=r["model"],
                implementation=r["implementation"],
                batch=r["batch"],
                context=context,
                phase=r["phase"],
                repeats=r["repeats"],
                token_kind=r["throughput_token_kind"],
                latency_ms=r["latency_ms"],
                latency_ms_min=r["latency_min_ms"],
                latency_ms_max=r["latency_max_ms"],
                total_tokens_per_second=r["tokens_per_second"],
                total_tokens_per_second_min=r["tokens_per_second_min"],
                total_tokens_per_second_max=r["tokens_per_second_max"],
                interactivity_tokens_per_second=r["tokens_per_second"] / r["batch"]
                if decode
                else None,
                interactivity_tokens_per_second_min=r["tokens_per_second_min"]
                / r["batch"]
                if decode
                else None,
                interactivity_tokens_per_second_max=r["tokens_per_second_max"]
                / r["batch"]
                if decode
                else None,
            )
        )
    return points


def plot_latency_sweeps(summary, x_axis="prompt"):
    """Shared sweeps: Chapter 2 context axis, Chapter 3 batch axis."""
    if x_axis not in ("prompt", "batch"):
        raise ValueError("Choose prompt or batch axis")
    series = "batch" if x_axis == "prompt" else "prompt"
    figures = {}
    implementations = sorted({r.get("implementation", "baseline") for r in summary})
    for key in sorted({r["model"] for r in summary}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for col, phase in enumerate(("prefill", "decode")):
            for impl in implementations:
                for value in sorted({r[series] for r in summary if r["model"] == key}):
                    selected = sorted(
                        (
                            r
                            for r in summary
                            if r["model"] == key
                            and r["phase"] == phase
                            and r[series] == value
                            and r.get("implementation", "baseline") == impl
                        ),
                        key=lambda r: r[x_axis],
                    )
                    if not selected:
                        continue
                    label = ("B=" if series == "batch" else "S=") + str(value)
                    if len(implementations) > 1:
                        label = impl + " " + label
                    for row, field, lo, hi in [
                        (0, "latency_ms", "latency_min_ms", "latency_max_ms"),
                        (
                            1,
                            "tokens_per_second",
                            "tokens_per_second_min",
                            "tokens_per_second_max",
                        ),
                    ]:
                        y = np.array([r[field] for r in selected])
                        axes[row, col].errorbar(
                            [r[x_axis] for r in selected],
                            y,
                            yerr=[
                                y - np.array([r[lo] for r in selected]),
                                np.array([r[hi] for r in selected]) - y,
                            ],
                            marker="o",
                            capsize=3,
                            label=label,
                        )
            axes[0, col].set_title(
                "Prefill (local TTFT)"
                if phase == "prefill"
                else "Decode (mean local TBT)"
            )
            axes[0, col].set_ylabel("Latency (ms)")
            axes[1, col].set_ylabel(
                "Input throughput (tok/s)"
                if phase == "prefill"
                else "Output throughput (tok/s)"
            )
            for ax in axes[:, col]:
                ax.set_xlabel(
                    "Batch"
                    if x_axis == "batch"
                    else "Prompt tokens per sequence"
                    if phase == "prefill"
                    else "Initial decode context (tokens)"
                )
                if x_axis == "prompt":
                    ax.set_xscale("log", base=2)
                else:
                    ax.set_xticks(
                        sorted({r["batch"] for r in summary if r["model"] == key})
                    )
                ax.grid(True, alpha=0.2)
                ax.legend()
        fig.suptitle(
            f"Qwen3-{key.upper()} · batch/context sweep; median and observed range"
        )
        fig.tight_layout()
        figures[key] = fig
    return figures


def plot_roofline(summary, hardware, precision, calibration=None, compact=False):
    intensities = np.logspace(-1, 5, 400)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.loglog(
        intensities,
        hardware.ceiling(intensities, precision),
        color="black",
        label=f"Configured roofline ({hardware.compute_basis} compute peak)",
    )
    if calibration:
        ax.loglog(
            intensities,
            np.minimum(
                calibration["effective_tflops"],
                intensities * calibration["copy_gbps"] / 1000,
            ),
            "--",
            color="gray",
            label="Measured GEMM/copy calibration envelope",
        )
    multiple = len({r.get("implementation", "baseline") for r in summary}) > 1
    seen = set()
    groups = sorted({(r["model"], r["phase"]) for r in summary})
    colors = {group: f"C{i}" for i, group in enumerate(groups)}
    for r in summary:
        y = r["achieved_tflops"]
        label = f"{r['model']} {r['phase']} B={r['batch']} S={r['prompt']}"
        if multiple:
            label = r.get("implementation", "baseline") + " " + label
        style = {}
        if compact:
            implementation = r.get("implementation", "baseline")
            group = (r["model"], r["phase"], implementation)
            label = (
                f"{implementation} {r['model']} {r['phase']}"
                if group not in seen
                else None
            )
            seen.add(group)
            style = dict(
                color=colors[r["model"], r["phase"]],
                markersize=4 + r["batch"],
                markerfacecolor="none"
                if implementation == "baseline"
                else colors[r["model"], r["phase"]],
            )
        ax.errorbar(
            r["intensity"],
            y,
            yerr=[[y - r["min_tflops"]], [r["max_tflops"] - y]],
            fmt={
                "baseline": "o",
                "optimized": "^",
                "fusion_only": "s",
                "attention_only": "D",
            }.get(r.get("implementation", "baseline"), "x")
            if compact
            else "o"
            if r["phase"] == "prefill"
            else "s",
            capsize=3,
            label=label,
            **style,
        )
    ax.set(
        xlabel="Modeled arithmetic intensity: causal FLOPs / compulsory-byte proxy (FLOP/byte)",
        ylabel="Measured causal matmul throughput (TFLOP/s)",
        title=hardware.name
        + " · Qwen3 prefill and decode"
        + ("\nMarker size follows batch; all saved contexts shown" if compact else ""),
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    return fig, ax
