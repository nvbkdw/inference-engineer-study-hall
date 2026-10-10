import time
import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

if __package__:
    from .utils import _launch, _check_bf16
else:
    # Direct execution places this file's directory on Python's import path.
    from utils import _launch, _check_bf16

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
def _rms_norm_kernel(mX, weight, mY, width: cutlass.Constexpr, eps: cutlass.Constexpr, tv_layout: cute.Layout):
    # kernel 
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    
    # get CTA local tile
    blk_coord = (None, bidx)
    gX = mX[blk_coord]
    gY = mY[blk_coord]

    # get thread local vals
    w_tv_layout = cute.make_ordered_layout((32, cute.ceil_div(width, 32)), order=(1, 0))
    tidfrgX = cute.composition(gX, tv_layout)
    tidfrgY = cute.composition(gY, tv_layout)
    tidfrgW = cute.composition(weight, w_tv_layout)
    thr_coord = (tidx,)
    thrX = tidfrgX[thr_coord]
    thrY = tidfrgY[thr_coord] 

    # aggreate local vals, square sum
    v = thrX.load()
    squared_sum = cutlass.Float32(0)
    for i in cutlass.range_constexpr(thrX.shape[0]): 
        val = v[i].to(cutlass.Float32)
        squared_sum += val * val 
    
    # warp level shuffle
    scale = cute.rsqrt(warp_sum(squared_sum) / width + eps)
    # TODO: load weight to shared mem first
    w = tidfrgW.load().to(cutlass.Float32)

    # store result
    cute.static_assert(cute.shape(v) == cute.shape(w))
    thrY[None] = (v*scale * w).to(cutlass.BFloat16)


@cute.jit
def _rms_norm_launcher(x, weight, y, width: cutlass.Constexpr, eps: cutlass.Constexpr, stream: cuda.CUstream):
    M = 4 # M rows per CTA
    # coalesced_bytes = 128
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((cute.ceil_div(width, 32),), order=(0,))
    tiler_m, tv_layout = cute.make_layout_tv(thr_layout, val_layout)


    mX = cute.zipped_divide(x, tiler_m)
    mY = cute.zipped_divide(y, tiler_m)
    _rms_norm_kernel(
        mX, weight, mY, width, eps, tv_layout
    ).launch(
        grid=((x.shape[0]+M-1)//M, 1, 1),
        block=(32*M, 1, 1), # one row per warp, M warps
        stream=stream
    )

def rms_norm(x, weight, eps):
    _check_bf16(x, weight)
    if weight.shape != (x.shape[-1],) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Weight width and positive finite epsilon required")

    y = torch.empty_like(x)
    width = x.shape[-1]
    _launch(
        'rms_norm',
        _rms_norm_launcher,
        (x.view(-1, width), weight, y.view(-1, width)),
        (width, eps)
    )

def rms_norm_ref(x, weight, eps):
    scale = torch.rsqrt(torch.sum(x*x, dim=-1, keepdim=True)/x.shape[-1] + eps)
    return x*scale*weight


# run correctness check
def correctness_check():
    # build torch tensor of shape (M, N, 128), then compare the result of torch reference rms norm and kernel
    # use bf16 input
    M, N, D = 1024, 16, 128
    
    x = torch.randn((M, N, D), dtype=torch.bfloat16, device='cuda:0')
    weight = torch.randn(D, dtype=torch.bfloat16, device='cuda:0')
    eps = 1e-6
    norm_ref = rms_norm_ref(x, weight, eps)
    norm = rms_norm(x, weight, eps)
    torch.testing.assert_close(norm, norm_ref, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    correctness_check()
