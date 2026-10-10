"""Compile dense MMA fragments for inspection without launching a GPU kernel.

    CUTE_DSL_ARCH=sm_90a CUTE_DSL_NO_CACHE=1 python examples/inspect_mma.py
    CUTE_DSL_ARCH=sm_100a CUTE_DSL_NO_CACHE=1 python examples/inspect_mma.py

This demonstrates operand setup for GEMM. It does not allocate tensor memory or
execute an asynchronous MMA; use probes beside these objects in your own GEMM.
"""

import os
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warp, warpgroup

import cuteviz


@cute.kernel
def fragments():
    if cutlass.const_expr(os.environ["CUTE_DSL_ARCH"] == "sm_90a"):
        tiled = cute.make_tiled_mma(
            warpgroup.MmaF16BF16Op(
                cutlass.Float16,
                cutlass.Float32,
                (64, 32, 16),
                warpgroup.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW32, cutlass.Float16)
    elif cutlass.const_expr(os.environ["CUTE_DSL_ARCH"] == "sm_100a"):
        tiled = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                cutlass.Float16,
                cutlass.Float32,
                (128, 32, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW32, cutlass.Float16)
    else:
        tiled = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
        )
        atom = cute.make_layout((8, 16), stride=(16, 1))
    m, n, k = tiled.get_tile_size(0), tiled.get_tile_size(1), tiled.get_tile_size(2)
    # Placeholder SMEM addresses are never dereferenced; this entry is compiled only.
    ptr = cute.make_ptr(cutlass.Float16, 0, cute.AddressSpace.smem, assumed_align=128)
    if cutlass.const_expr(os.environ["CUTE_DSL_ARCH"] in ("sm_90a", "sm_100a")):
        ptr = cute.recast_ptr(ptr, atom.inner, cutlass.Float16)
        atom = atom.outer
    sA = cute.make_tensor(ptr, cute.tile_to_shape(atom, (m, k), (1, 0)))
    sB = cute.make_tensor(ptr, cute.tile_to_shape(atom, (n, k), (1, 0)))
    thr = tiled.get_slice(0)
    a = tiled.make_fragment_A(thr.partition_A(sA))
    b = tiled.make_fragment_B(thr.partition_B(sB))
    acc = tiled.make_fragment_C(tiled.partition_shape_C((m, n)))
    cuteviz.inspect("sA", sA)
    cuteviz.inspect("sB", sB)
    cuteviz.inspect_mma("mma", tiled, a=a, b=b, c=acc)


@cute.jit
def entry():
    fragments().launch(grid=(1, 1, 1), block=(128, 1, 1))


if __name__ == "__main__":
    target = os.environ.get("CUTE_DSL_ARCH", "sm_80")
    os.environ.setdefault("CUTE_DSL_ARCH", target)
    destination = Path(__file__).parent / "captures" / f"{target}.cuteviz.json"
    with cuteviz.capture(destination):
        cute.compile(entry)
    print(destination)
