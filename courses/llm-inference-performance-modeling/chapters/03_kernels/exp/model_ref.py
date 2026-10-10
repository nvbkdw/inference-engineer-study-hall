from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

# Direct script execution adds exp/ to sys.path; course imports need its root.
ROOT = Path(__file__).resolve().parents[3]
CHAPTER1_CODE = ROOT / 'chapters' / '01_reconstruct_qwen3' / 'code'
# Chapter 1 utilities also import sibling modules such as checkpoint.
for path in (ROOT, CHAPTER1_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

lab1_utils = importlib.import_module("chapters.01_reconstruct_qwen3.code.notebook_utils")


@dataclass
class Config:
    layers: int = 2
    hidden: int = 4096
    intermediate: int = 12288
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128  # Use explicit head_dim; full 32B has rectangular Q/O projections.
    vocab: int = 151936
    eps: float = 1e-6
    theta: float = 1e6

    def parameters(self):
        d, h, g, r, i = self.hidden, self.q_heads, self.kv_heads, self.head_dim, self.intermediate
        return self.layers * (2*d*h*r + 2*d*g*r + 3*d*i + 2*d + 2*r) + 2*self.vocab*d + d

    def kv_bytes(self, tokens, element_bytes=2):
        return 2*self.layers*self.kv_heads*self.head_dim*element_bytes*tokens


"""Qwen3 decoder operators; verify their checkpoint logits in the cells below."""

class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        y = x.float()
        mean_square = y.square().mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(mean_square + self.eps)
        return self.weight * y.to(x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float):
    # x: [B, H, T, R]; positions: [B, T], using absolute cached positions.
    _, _, T, R = x.shape
    assert R % 2 == 0
    # Match the oracle's FP32 constants: reciprocal of a positive power on CPU,
    # then transfer to the input device. GPU pow (including a negative exponent)
    # rounds differently; those small phase errors accumulate across layers.
    exponents = torch.arange(0, R, 2, dtype=torch.float32, device='cpu') / R
    freqs = (1.0 / (theta ** exponents)).to(x.device)
    angles = positions.float().unsqueeze(-1) * freqs  # [B, T, R/2]
    angles = torch.cat([angles, angles], dim=-1).unsqueeze(1)  # [B, 1, T, R]
    a, b = x.chunk(2, dim=-1)
    rotate_half = torch.cat([-b, a], dim=-1)
    return x * angles.cos().to(x.dtype) + rotate_half * angles.sin().to(x.dtype)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.input_norm = RMSNorm(c.hidden, eps=c.eps)
        self.post_norm = RMSNorm(c.hidden, eps=c.eps)
        self.q_norm = RMSNorm(c.head_dim, eps=c.eps)
        self.k_norm = RMSNorm(c.head_dim, eps=c.eps)
        self.k = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.q = nn.Linear(c.hidden, c.q_heads * c.head_dim, bias=False)
        self.o = nn.Linear(c.q_heads * c.head_dim, c.hidden, bias=False)
        self.gate = nn.Linear(c.hidden, c.intermediate, bias=False)
        self.up = nn.Linear(c.hidden, c.intermediate, bias=False)
        self.down = nn.Linear(c.intermediate, c.hidden, bias=False)

    def normalize(self, norm, x):
        return norm(x)

    def normalize_rotary(self, q, k, positions):
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        return rope(q, positions, self.c.theta), rope(k, positions, self.c.theta)

    def activation(self, gate, up):
        return F.silu(gate) * up

    def attention(self, q, k, v, positions):
        assert self.c.q_heads % self.c.kv_heads == 0
        groups = self.c.q_heads // self.c.kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = (q.float() @ k.float().transpose(-2, -1)) / self.c.head_dim**0.5
        # Query at prefix+t can attend to keys 0..prefix+t, including itself.
        allowed = torch.arange(k.shape[2], device=q.device)[None, None, :] <= positions[:, :, None]
        scores = scores.masked_fill(~allowed[:, None, :, :], float('-inf'))
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
        out = (probabilities @ v.float()).to(q.dtype)
        return out

    def forward(self, x, kv_cache=None):
        # x: [B, T, D]; each cache tensor: [B, Hkv, prefix, R].
        B, T, _ = x.shape
        z = self.normalize(self.input_norm, x)  # Keep x for the first residual addition.
        q = self.q(z).view(B, T, self.c.q_heads, self.c.head_dim)
        k = self.k(z).view(B, T, self.c.kv_heads, self.c.head_dim)
        v = self.v(z).view(B, T, self.c.kv_heads, self.c.head_dim)
        v = v.transpose(1, 2)

        prefix = kv_cache[0].shape[2] if kv_cache is not None else 0
        positions = torch.arange(prefix, prefix + T, device=x.device).unsqueeze(0).expand(B, T)
        q, k = self.normalize_rotary(q, k, positions)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)  # Store Hkv heads, before expansion for the dense oracle.

        out = self.attention(q, k, v, positions)
        out = out.transpose(1, 2).reshape(B, T, self.c.q_heads * self.c.head_dim)
        x = x + self.o(out)

        z = self.normalize(self.post_norm, x)  # Keep the updated x for the second residual addition.
        x = x + self.down(self.activation(self.gate(z), self.up(z)))
        return x, new_cache

class Qwen3(nn.Module):
    block_class = Block

    def normalize_final(self, x):
        return self.norm(x)

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.hidden)
        self.layers = nn.ModuleList([self.block_class(c) for _ in range(c.layers)])
        self.norm = RMSNorm(c.hidden, eps=c.eps)
        self.head = nn.Linear(c.hidden, c.vocab, bias=False)

    def forward(self, x, kv_cache=None, decode=False):
        x, new = self.emb(x), []
        for i, block in enumerate(self.layers):
            x, cache = block(x, kv_cache[i] if kv_cache is not None else None)
            new.append(cache)
        if decode:
            x = x[:, -1:, :]
        x = self.normalize_final(x)
        o = self.head(x)
        return o, new

def model_weight_name_mapping(config):
    """Map published Qwen3 weight names to this notebook's custom module names."""
    names = {
        'model.embed_tokens.weight': 'emb.weight',
        'model.norm.weight': 'norm.weight',
        'lm_head.weight': 'head.weight',
    }
    block_weights = {
        'input_layernorm.weight': 'input_norm.weight',
        'post_attention_layernorm.weight': 'post_norm.weight',
        'self_attn.q_norm.weight': 'q_norm.weight',
        'self_attn.k_norm.weight': 'k_norm.weight',
        'self_attn.q_proj.weight': 'q.weight',
        'self_attn.k_proj.weight': 'k.weight',
        'self_attn.v_proj.weight': 'v.weight',
        'self_attn.o_proj.weight': 'o.weight',
        'mlp.gate_proj.weight': 'gate.weight',
        'mlp.up_proj.weight': 'up.weight',
        'mlp.down_proj.weight': 'down.weight',
    }
    for layer in range(config.layers):
        for checkpoint_suffix, custom_suffix in block_weights.items():
            names[f'model.layers.{layer}.{checkpoint_suffix}'] = (
                f'layers.{layer}.{custom_suffix}'
            )
    return names

weight_name_mapping = model_weight_name_mapping(Config())

if __name__ == '__main__':
    from make_fixture import SOURCE_REVISION

    # Verify the full 8B checkpoint at the course's pinned source revision.
    verification = lab1_utils.VerificationRun(
        Qwen3,
        name_map_factory=model_weight_name_mapping,
        model_case='8b',
        revision=SOURCE_REVISION,
        snapshot=ROOT / 'models' / 'qwen3-8b' / SOURCE_REVISION,
        tokens_path=ROOT / 'results' / f'qwen3-8b-{SOURCE_REVISION}-tokens.json',
        notebook_path=Path(__file__),
        device='cuda:0',
        dtype='fp32',
    )
    verification.prepare_checkpoint(download=False)
    verification.audit()
    verification.generate_logits('transformers')
    verification.generate_logits('custom')
    comparison = verification.compare(rtol=1e-4, atol=1e-5)
    print(json.dumps(comparison, indent=2))
