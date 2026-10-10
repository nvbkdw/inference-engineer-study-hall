"""Symmetric groupwise INT4 numerics and byte packing; not an AWQ/packed GEMM."""
import torch


def quantize(w, group_size):
    if w.ndim != 2 or group_size <= 0 or w.shape[1] % group_size:
        raise ValueError('2D weights with input width divisible by group size required')
    groups = w.float().reshape(w.shape[0], -1, group_size)
    peak = groups.abs().amax(-1, keepdim=True)
    scale = torch.where(peak == 0, torch.ones_like(peak), peak/7)
    q = torch.round(groups/scale).clamp(-7,7).to(torch.int8)
    return q.reshape_as(w), scale.squeeze(-1)


def dequantize(q, scale, group_size):
    return (q.reshape(q.shape[0], -1, group_size).float()*scale[...,None]).reshape_as(q)


def pack(q):
    """Offset code = signed q + 8. Low nibble first; even number of elements."""
    if q.numel() % 2 or bool(((q < -8) | (q > 7)).any()):
        raise ValueError('even number of signed 4-bit values required')
    codes = (q.flatten().to(torch.int16)+8).to(torch.uint8)
    return codes[::2] | (codes[1::2] << 4)


def unpack(packed, shape):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten()
    return (codes.to(torch.int16)-8).to(torch.int8).reshape(shape)


def check():
    torch.manual_seed(42)
    w, x = torch.randn(32,256), torch.randn(16,256)
    w[0] = 0
    codes = torch.arange(-8,8, dtype=torch.int8)
    assert torch.equal(unpack(pack(codes), codes.shape), codes)
    for g in (32,128):
        q, scale = quantize(w,g)
        packed = pack(q)
        assert torch.equal(unpack(packed,q.shape),q)
        reconstructed = dequantize(q,scale,g)
        group_error = (w-reconstructed).abs().reshape(32,-1,g)
        assert bool((group_error <= scale[...,None]/2+1e-6).all())
        assert torch.equal(reconstructed[0],w[0])
        out, ref = x @ reconstructed.T, x @ w.T
        print({'kind':'random-weight numerical exercise', 'group':g,
               'packed_plus_fp32_scales_bytes':packed.numel()+4*scale.numel(),
               'relative_output_l2':float((out-ref).norm()/ref.norm())})
    print('PASS: nibble roundtrip, zero group, rounding error bound')


if __name__ == '__main__':
    check()
