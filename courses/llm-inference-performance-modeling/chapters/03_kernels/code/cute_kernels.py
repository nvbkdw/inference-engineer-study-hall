"""Forward-only teaching kernels: BF16 storage, FP32 reductions, SIMT on SM121.

No tensor-core architecture-specific instructions. Thread/value ownership is
explicit so tile/work partition experiments need no external attention engine.
Compilation and dispatch metadata are exposed for strict benchmark auditing.
"""

import time
import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

COMPILED = {}
COMPILATIONS = []


@dsl_user_op
def scaled_score(value, *, loc=None, ip=None):
    """Round the reference's FP32 score scaling before softmax subtraction."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, 0f3DB504F3;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def round_bf16(value, *, loc=None, ip=None):
    """Force the eager BF16 rounding boundary before a later addition.

    The 4.2.1 compiler can contract BF16 multiply + add into BF16 FMA even
    across truncation/extension casts. Explicit conversion assembly keeps
    both rotary products rounded, as in the separate PyTorch operators.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 rounded; cvt.rn.bf16.f32 rounded, $1; cvt.f32.bf16 $0, rounded; }",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def warp_sum(x):
    for delta in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, 1 << delta)
    return x


@cute.jit
def warp_sum_descending(x):
    for i in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, 16 >> i)
    return x


@cute.kernel
def rms_kernel(
    X: cute.Tensor,
    W: cute.Tensor,
    Y: cute.Tensor,
    width: cutlass.Constexpr,
    eps: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % 32
    row = row * 4 + tid // 32
    if row < X.shape[0]:
        squares = cutlass.Float32(0)
        for i in cutlass.range_constexpr(cute.ceil_div(width, 32)):
            col = lane + 32 * i
            if col < width:
                val = X[row, col].to(cutlass.Float32)
                squares += val * val
        scale = cute.rsqrt(warp_sum(squares) / width + eps)
        for i in cutlass.range_constexpr(cute.ceil_div(width, 32)):
            col = lane + 32 * i
            if col < width:
                # Match Chapter 1: round normalized value, then scale and round.
                val = (
                    (X[row, col].to(cutlass.Float32) * scale)
                    .to(cutlass.BFloat16)
                    .to(cutlass.Float32)
                )
                Y[row, col] = (val * W[col].to(cutlass.Float32)).to(cutlass.BFloat16)


@cute.jit
def rms_launch(
    X, W, Y, width: cutlass.Constexpr, eps: cutlass.Constexpr, stream: cuda.CUstream
):
    rms_kernel(X, W, Y, width, eps).launch(
        grid=(cute.ceil_div(X.shape[0], 4), 1, 1), block=(128, 1, 1), stream=stream
    )


@cute.kernel
def swiglu_kernel(
    G: cute.Tensor, U: cute.Tensor, Y: cute.Tensor, block: cutlass.Constexpr
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * block + tid
    if i < G.shape[0]:
        g = G[i].to(cutlass.Float32)
        silu = (g / (1 + cute.exp(-g))).to(cutlass.BFloat16).to(cutlass.Float32)
        Y[i] = (silu * U[i].to(cutlass.Float32)).to(cutlass.BFloat16)


@cute.jit
def swiglu_launch(G, U, Y, block: cutlass.Constexpr, stream: cuda.CUstream):
    swiglu_kernel(G, U, Y, block).launch(
        grid=(cute.ceil_div(G.shape[0], block), 1, 1),
        block=(block, 1, 1),
        stream=stream,
    )


def _tensor(x):
    return from_dlpack(x.detach()).mark_layout_dynamic(leading_dim=x.ndim - 1)


def _launch(name, fn, tensors, constants):
    args = tuple(_tensor(t) for t in tensors)
    # mark_layout_dynamic preserves ZERO strides as compile-time constants.
    # Broadcast positions [B,T] therefore have a different launch ABI from
    # an unbroadcast B=1 tensor. Never share those compiled argument packs.
    key = (
        name,
        tuple(
            (
                t.ndim,
                str(t.dtype),
                t.device.index,
                tuple(i for i, stride in enumerate(t.stride()) if stride == 0),
                tuple(i for i, stride in enumerate(t.stride()) if stride == 1),
            )
            for t in tensors
        ),
        constants,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(tensors[0].device).cuda_stream)
    if key not in COMPILED:
        begin = time.perf_counter()
        COMPILED[key] = cute.compile(fn, *args, *constants, stream)
        COMPILATIONS.append(
            dict(kernel=name, signature=str(key), seconds=time.perf_counter() - begin)
        )
    COMPILED[key](*args, stream)
    return COMPILED[key], args, stream


def _check_bf16(*tensors):
    if not tensors or any(
        t.device.type != "cuda" or t.dtype != torch.bfloat16 or t.stride(-1) != 1
        for t in tensors
    ):
        raise ValueError("CuTe kernels require CUDA BF16 tensors with unit last stride")
    if any(t.device != tensors[0].device for t in tensors):
        raise ValueError("All tensors must share a CUDA device")
    if any(any(s <= 0 for s in t.shape) for t in tensors):
        raise ValueError("Empty tensors are unsupported")


def rmsnorm(x, weight, eps=1e-6):
    """Leading dimensions must flatten to rows; final-token row strides are allowed."""
    _check_bf16(x, weight)
    if weight.shape != (x.shape[-1],) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Weight width and positive finite epsilon required")
    y = torch.empty_like(x)
    _launch(
        "rmsnorm",
        rms_launch,
        (x.view(-1, x.shape[-1]), weight, y.view(-1, x.shape[-1])),
        (x.shape[-1], eps),
    )
    return y


def swiglu(gate, up, block=256):
    """Contiguous projection outputs; no hidden layout copies in this adapter."""
    _check_bf16(gate, up)
    if gate.shape != up.shape or block not in (128, 256, 512):
        raise ValueError("Matching shapes and block 128/256/512 required")
    y = torch.empty_like(gate)
    _launch("swiglu", swiglu_launch, (gate.view(-1), up.view(-1), y.view(-1)), (block,))
    return y


@cute.jit
def norm_rope_row(
    X, W, Y, P, b, t, h, lane, eps: cutlass.Constexpr, theta: cutlass.Constexpr
):
    values = cute.make_fragment((4,), cutlass.Float32)
    squares = cutlass.Float32(0)
    for i in cutlass.range_constexpr(4):
        z = X[b, t, h, lane + 32 * i].to(cutlass.Float32)
        values[i] = z
        # Match the reference's four-consecutive-value FP32 mean reduction.
        adjacent = X[b, t, h, 4 * lane + i].to(cutlass.Float32)
        squares += adjacent * adjacent
    for i in cutlass.range_constexpr(5):
        squares += cute.arch.shuffle_sync_down(squares, 16 >> i)
    squares = cute.arch.shuffle_sync(squares, 0)
    scale = cute.rsqrt(squares / 128 + eps)
    for i in cutlass.range_constexpr(4):
        rounded = (values[i] * scale).to(cutlass.BFloat16).to(cutlass.Float32)
        values[i] = (
            (rounded * W[lane + 32 * i].to(cutlass.Float32))
            .to(cutlass.BFloat16)
            .to(cutlass.Float32)
        )
    for i in cutlass.range_constexpr(4):
        # Qwen rotate_half: pair the first/second 64, never adjacent elements.
        d = lane + 32 * (i % 2)
        angle = P[b, t].to(cutlass.Float32) * cute.exp(-math.log(theta) * d / 64)
        co = cutlass.Float32(cute.cos(angle)).to(cutlass.BFloat16).to(cutlass.Float32)
        si = cutlass.Float32(cute.sin(angle)).to(cutlass.BFloat16).to(cutlass.Float32)
        rotated = values[(i + 2) % 4]
        if cutlass.const_expr(i < 2):
            rotated = -rotated
        a = round_bf16(values[i] * co)
        z = round_bf16(rotated * si)
        Y[b, t, h, lane + 32 * i] = (a + z).to(cutlass.BFloat16)


@cute.kernel
def qk_rope_kernel(
    Q: cute.Tensor,
    K: cute.Tensor,
    WQ: cute.Tensor,
    WK: cute.Tensor,
    P: cute.Tensor,
    OQ: cute.Tensor,
    OK: cute.Tensor,
    eps: cutlass.Constexpr,
    theta: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    row = bid * 4 + tid // 32
    heads = Q.shape[2] + K.shape[2]
    h = row % heads
    t = row // heads % Q.shape[1]
    b = row // heads // Q.shape[1]
    if b < Q.shape[0]:
        if h < Q.shape[2]:
            norm_rope_row(Q, WQ, OQ, P, b, t, h, tid % 32, eps, theta)
        else:
            norm_rope_row(K, WK, OK, P, b, t, h - Q.shape[2], tid % 32, eps, theta)


@cute.jit
def qk_rope_launch(
    Q,
    K,
    WQ,
    WK,
    P,
    OQ,
    OK,
    eps: cutlass.Constexpr,
    theta: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    rows = Q.shape[0] * Q.shape[1] * (Q.shape[2] + K.shape[2])
    qk_rope_kernel(Q, K, WQ, WK, P, OQ, OK, eps, theta).launch(
        grid=(cute.ceil_div(rows, 4), 1, 1), block=(128, 1, 1), stream=stream
    )


def qk_norm_rope(q, k, wq, wk, positions, eps=1e-6, theta=1e6):
    _check_bf16(q, k, wq, wk)
    if (
        q.ndim != 4
        or k.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or q.shape[-1] != 128
        or k.shape[-1] != 128
        or wq.shape != (128,)
        or wk.shape != (128,)
    ):
        raise ValueError("Q/K must be [B,T,H,128] with [128] norm weights")
    if (
        positions.shape != q.shape[:2]
        or positions.device != q.device
        or positions.dtype != torch.int64
    ):
        raise ValueError("Positions must be int64 [B,T] on the same device")
    if not all(math.isfinite(v) and v > 0 for v in (eps, theta)):
        raise ValueError("Positive finite epsilon and theta required")
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    _launch(
        "qk_norm_rope", qk_rope_launch, (q, k, wq, wk, positions, oq, ok), (eps, theta)
    )
    return oq.transpose(1, 2), ok.transpose(1, 2)


@cute.kernel
def attention_kernel(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    key_tile: cutlass.Constexpr,
    warps: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, h, b = cute.arch.block_idx()
    t = bid * warps + tid // 32
    lane = tid % 32
    if t < Q.shape[2]:
        kh = h // (Q.shape[1] // K.shape[1])
        prefix = K.shape[2] - Q.shape[2]
        q = cute.make_fragment((4,), cutlass.Float32)
        acc = cute.make_fragment((4,), cutlass.Float32)
        for d in cutlass.range_constexpr(4):
            q[d] = Q[b, h, t, lane + 32 * d].to(cutlass.Float32)
            acc[d] = cutlass.Float32(0)
        maximum = cutlass.Float32(float("-inf"))
        denominator = cutlass.Float32(0)
        # Tile bounds avoid future-only tiles and predicate the partial final tile.
        for base in range(0, prefix + t + 1, key_tile):
            tile_max = cutlass.Float32(float("-inf"))
            for j in range(key_tile):
                s = base + j
                if s <= prefix + t:
                    score = cutlass.Float32(0)
                    for d in cutlass.range_constexpr(4):
                        score += q[d] * K[b, kh, s, lane + 32 * d].to(cutlass.Float32)
                    score = scaled_score(warp_sum(score))
                    tile_max = cute.arch.fmax(tile_max, score)
            new_max = cute.arch.fmax(maximum, tile_max)
            alpha = cute.exp(maximum - new_max)
            previous_mass = denominator * alpha
            lane_mass = cutlass.Float32(0)
            # Recompute tile scores instead of storing them or spilling a score
            # fragment. One rescaling per tile limits cumulative FP32 rounding.
            for j in range(key_tile):
                s = base + j
                if s <= prefix + t:
                    score = cutlass.Float32(0)
                    for d in cutlass.range_constexpr(4):
                        score += q[d] * K[b, kh, s, lane + 32 * d].to(cutlass.Float32)
                    score = scaled_score(warp_sum(score))
                    p = cute.exp(score - new_max)
                    if j % 32 == lane:
                        lane_mass = lane_mass + p
            denominator = previous_mass + warp_sum_descending(lane_mass)
            for d in cutlass.range_constexpr(4):
                acc[d] = acc[d] * (previous_mass / denominator)
            for j in range(key_tile):
                s = base + j
                if s <= prefix + t:
                    score = cutlass.Float32(0)
                    for d in cutlass.range_constexpr(4):
                        score += q[d] * K[b, kh, s, lane + 32 * d].to(cutlass.Float32)
                    score = scaled_score(warp_sum(score))
                    p = cute.exp(score - new_max) / denominator
                    for d in cutlass.range_constexpr(4):
                        acc[d] = acc[d] + p * V[b, kh, s, lane + 32 * d].to(
                            cutlass.Float32
                        )
            maximum = new_max
        for d in cutlass.range_constexpr(4):
            O[b, h, t, lane + 32 * d] = acc[d].to(cutlass.BFloat16)


@cute.jit
def attention_launch(
    Q,
    K,
    V,
    O,
    key_tile: cutlass.Constexpr,
    warps: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    attention_kernel(Q, K, V, O, key_tile, warps).launch(
        grid=(cute.ceil_div(Q.shape[2], warps), Q.shape[1], Q.shape[0]),
        block=(32 * warps, 1, 1),
        stream=stream,
    )


def attention(q, k, v, key_tile=32, warps=4):
    """Dense equal-length causal GQA; [B,Hq,T,128], [B,Hkv,P+T,128].

    Strided transposes from projections and contiguous appended caches supported.
    Scores/KV expansion never written to global memory. One warp per query;
    tiles stream key rows using online softmax. This SIMT teaching implementation
    prioritizes explicit ownership over tensor-core throughput.
    """
    _check_bf16(q, k, v)
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Attention requires rank-four tensors")
    if key_tile <= 0 or warps not in (1, 2, 4, 8):
        raise ValueError("Positive key tile and 1/2/4/8 warps required")
    if (
        q.shape[-1] != 128
        or k.shape[-1] != 128
        or k.shape != v.shape
        or q.shape[0] != k.shape[0]
        or q.shape[1] % k.shape[1]
        or q.shape[2] > k.shape[2]
    ):
        raise ValueError("Unsupported attention dimensions")
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _launch("attention", attention_launch, (q, k, v, out), (key_tile, warps))
    return out
