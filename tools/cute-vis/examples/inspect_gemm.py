"""One real 16×8×16 FP16 warp GEMM, with FP32 accumulation.

A is row-major (M,K); B is row-major (N,K), i.e. column-major (K,N).
C is row-major (M,N). This small kernel demonstrates probe placement.

    CUTE_DSL_ARCH=sm_80 python examples/inspect_gemm.py
"""

from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp

import cuteviz


@cute.kernel
def gemm_kernel(a: cute.Pointer, b: cute.Pointer, c: cute.Pointer):
    gA = cute.make_tensor(a, cute.make_layout((16, 16), stride=(16, 1)))
    gB = cute.make_tensor(b, cute.make_layout((8, 16), stride=(16, 1)))
    gC = cute.make_tensor(c, cute.make_layout((16, 8), stride=(8, 1)))
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)))
    thr = mma.get_slice(cute.arch.thread_idx()[0])
    pA, pB, pC = thr.partition_A(gA), thr.partition_B(gB), thr.partition_C(gC)
    rA, rB, acc = mma.make_fragment_A(pA), mma.make_fragment_B(pB), mma.make_fragment_C(pC)
    load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    cute.copy(load, pA, rA)
    cute.copy(load, pB, rB)
    acc.fill(0)
    cuteviz.inspect("gA", gA)
    cuteviz.inspect("gB", gB)
    cuteviz.inspect("gC", gC)
    cuteviz.inspect_mma("warp_gemm", mma, a=rA, b=rB, c=acc)
    cute.gemm(mma, acc, rA, rB, acc)
    cute.copy(store, acc, pC)


@cute.jit
def entry(a: cute.Pointer, b: cute.Pointer, c: cute.Pointer):
    gemm_kernel(a, b, c).launch(grid=(1, 1, 1), block=(32, 1, 1))


if __name__ == "__main__":
    half = cute.runtime.make_ptr(cutlass.Float16, 0, cute.AddressSpace.gmem, assumed_align=16)
    fp32 = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    destination = Path(__file__).parent / "captures" / "gemm.cuteviz.json"
    with cuteviz.capture(destination):
        cute.compile(entry, half, half, fp32)
    print(destination)
