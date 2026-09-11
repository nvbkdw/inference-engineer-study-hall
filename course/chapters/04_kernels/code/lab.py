"""Tiled FP32 attention oracle for decode and causal prefill; not a fused GPU kernel."""
import torch


def online_attention(q, k, v, offset=0, tile=32):
    """[B,Hq,T,R], [B,Hkv,S,R]. offset is the first absolute query position.

    GQA repeat here is an oracle convenience; a GPU kernel should reuse KV.
    The score temporary is [B,Hq,T,tile], not [B,Hq,T,S].
    """
    if tile <= 0 or offset < 0 or offset+q.shape[2] > k.shape[2]:
        raise ValueError('invalid tile or causal positions')
    if q.shape[1] % k.shape[1]:
        raise ValueError('query heads must divide into KV groups')
    k = k.repeat_interleave(q.shape[1]//k.shape[1], 1).float()
    v = v.repeat_interleave(q.shape[1]//v.shape[1], 1).float()
    q = q.float()
    m = torch.full((*q.shape[:-1], 1), -float('inf'), device=q.device)
    ell = torch.zeros_like(m)
    u = torch.zeros_like(q)
    positions = offset + torch.arange(q.shape[2], device=q.device)
    for begin in range(0, k.shape[2], tile):
        end = min(begin+tile, k.shape[2])
        score = q @ k[:, :, begin:end].transpose(-1, -2) / q.shape[-1]**0.5
        mask = torch.arange(begin, end, device=q.device)[None, :] <= positions[:, None]
        score = score.masked_fill(~mask[None, None], -float('inf'))
        new_m = torch.maximum(m, score.amax(-1, keepdim=True))
        # Every row has seen key 0 in the first tile, so new_m is finite.
        rescale = torch.exp(m-new_m)
        prob = torch.exp(score-new_m)
        u = rescale*u + prob @ v[:, :, begin:end]
        ell = rescale*ell + prob.sum(-1, keepdim=True)
        m = new_m
    return u/ell


def dense_attention(q, k, v, offset):
    k = k.repeat_interleave(q.shape[1]//k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1]//v.shape[1], 1)
    score = q @ k.transpose(-1, -2) / q.shape[-1]**0.5
    mask = torch.arange(k.shape[2], device=q.device)[None, :] <= offset + torch.arange(q.shape[2], device=q.device)[:, None]
    return score.masked_fill(~mask[None, None], -float('inf')).softmax(-1) @ v


def check():
    torch.manual_seed(7)
    for s in (1, 31, 32, 33, 127, 128, 129):
        k, v = torch.randn(2, 2, s, 8), torch.randn(2, 2, s, 8)
        for t, offset in ((1, s-1), (s, 0)):
            q = torch.randn(2, 8, t, 8)*10
            ref = dense_attention(q, k, v, offset)
            for tile in (1, 17, 32):
                torch.testing.assert_close(online_attention(q, k, v, offset, tile), ref, rtol=1e-4, atol=1e-5)
    print('PASS: online softmax, GQA, decode/prefill, extreme scores, partial tiles')


if __name__ == '__main__':
    check()
