"""Tiny Qwen3-style FP32 reference. No pretrained weights or fast kernels."""
import argparse
from dataclasses import dataclass
import json
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    layers: int = 2
    hidden: int = 48
    intermediate: int = 96
    q_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 8  # Deliberately hidden != q_heads * head_dim.
    vocab: int = 101
    eps: float = 1e-6
    theta: float = 1e6

    def parameters(self):
        d, h, g, r, i = self.hidden, self.q_heads, self.kv_heads, self.head_dim, self.intermediate
        return self.layers * (2*d*h*r + 2*d*g*r + 3*d*i + 2*d + 2*r) + 2*self.vocab*d + d

    def kv_bytes(self, tokens, element_bytes=2):
        return 2*self.layers*self.kv_heads*self.head_dim*element_bytes*tokens


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rope(x, positions, theta):
    """x: [B,H,T,R]; split-half rotary convention, explicit absolute positions."""
    r = x.shape[-1]
    assert r % 2 == 0
    freq = theta ** (-torch.arange(0, r, 2, device=x.device).float() / r)
    angles = positions.float()[:, None] * freq[None, :]
    angles = torch.cat((angles, angles), -1)[None, None]
    a, b = x.chunk(2, -1)
    rotated = torch.cat((-b, a), -1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.input_norm = RMSNorm(c.hidden, c.eps)
        self.post_norm = RMSNorm(c.hidden, c.eps)
        self.q_norm = RMSNorm(c.head_dim, c.eps)
        self.k_norm = RMSNorm(c.head_dim, c.eps)
        for name, inp, out in [
            ('q', c.hidden, c.q_heads*c.head_dim),
            ('k', c.hidden, c.kv_heads*c.head_dim),
            ('v', c.hidden, c.kv_heads*c.head_dim),
            ('o', c.q_heads*c.head_dim, c.hidden),
            ('gate', c.hidden, c.intermediate),
            ('up', c.hidden, c.intermediate),
            ('down', c.intermediate, c.hidden),
        ]:
            setattr(self, name, nn.Linear(inp, out, bias=False))

    def forward(self, x, cache=None):
        c = self.c
        b, t, _ = x.shape
        prefix = 0 if cache is None else cache[0].shape[2]
        positions = torch.arange(prefix, prefix+t, device=x.device)
        z = self.input_norm(x)
        q = self.q_norm(self.q(z).view(b, t, c.q_heads, c.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(z).view(b, t, c.kv_heads, c.head_dim)).transpose(1, 2)
        v = self.v(z).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, positions, c.theta), rope(k, positions, c.theta)
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        new_cache = (k, v)
        assert c.q_heads % c.kv_heads == 0
        k = k.repeat_interleave(c.q_heads // c.kv_heads, dim=1)
        v = v.repeat_interleave(c.q_heads // c.kv_heads, dim=1)
        scores = (q.float() @ k.float().transpose(-1, -2)) / c.head_dim**0.5
        allowed = torch.arange(k.shape[2], device=x.device)[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None, None], -float('inf'))
        out = (scores.softmax(-1) @ v.float()).to(x.dtype)
        x = x + self.o(out.transpose(1, 2).reshape(b, t, -1))
        z = self.post_norm(x)
        x = x + self.down(F.silu(self.gate(z)) * self.up(z))
        return x, new_cache


class TinyQwen(nn.Module):
    def __init__(self, c=Config()):
        super().__init__()
        self.embed = nn.Embedding(c.vocab, c.hidden)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = RMSNorm(c.hidden, c.eps)
        self.head = nn.Linear(c.hidden, c.vocab, bias=False)

    def forward(self, ids, caches=None, last_only=False):
        x, new = self.embed(ids), []
        for j, layer in enumerate(self.layers):
            x, cache = layer(x, None if caches is None else caches[j])
            new.append(cache)
        if last_only:
            x = x[:, -1:]
        return self.head(self.norm(x)), new


@torch.inference_mode()
def check():
    torch.manual_seed(42)
    c = Config()
    model = TinyQwen(c).eval()
    assert sum(p.numel() for p in model.parameters()) == c.parameters()
    ids = torch.randint(0, c.vocab, (2, 11))
    full, _ = model(ids)
    for chunks in ([1]*11, [3, 1, 7], [5, 6]):
        cache, pieces, start = None, [], 0
        for size in chunks:
            logits, cache = model(ids[:, start:start+size], cache)
            pieces.append(logits)
            start += size
        torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-4, atol=1e-5)
    print('PASS: parameter count, incremental decode, unequal chunk equivalence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Inventory a downloaded Qwen config; no weights loaded')
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            d = json.load(f)
        c = Config(d['num_hidden_layers'], d['hidden_size'], d['intermediate_size'],
                   d['num_attention_heads'], d['num_key_value_heads'], d['head_dim'], d['vocab_size'])
        print(json.dumps({'parameters': c.parameters(), 'bf16_weight_bytes': 2*c.parameters(),
                          'bf16_kv_bytes_per_token': c.kv_bytes(1)}, indent=2))
    else:
        check()
