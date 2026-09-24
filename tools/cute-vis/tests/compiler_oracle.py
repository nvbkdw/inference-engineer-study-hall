"""Independent CuTe partition oracle, executed in a subprocess per target.

Production serializes TV layout descriptors and evaluates them with tensor-layouts.
This oracle instead partitions identity coordinate tensors with CuTe itself.
"""

import json
import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warp, warpgroup
from cutlass.cutlass_dsl import dsl_user_op

import cuteviz

EXPECTED = []
TARGET = os.environ["CUTE_DSL_ARCH"]


@dsl_user_op
def sample_mma(label, tiled, *, loc=None):
    scratch = ir.Module.create()
    with ir.InsertionPoint(scratch.body):
        m, n, k = (int(tiled.get_tile_size(i)) for i in range(3))
        for role, shape in zip("ABC", [(m, k), (n, k), (m, n)]):
            coords = cute.make_identity_tensor(shape)
            threads = sorted({0, 1, 3, 4, 7, 8, 31, 32, 63, 64, 95, 96, int(tiled.size) - 1})
            for t in (t for t in threads if t < tiled.size):
                part = getattr(tiled.get_slice(t), f"partition_{role}")(coords)
                count = int(cute.size(part))
                for v in sorted({0, 1, 2, 3, 7, 8, count // 2, count - 1}):
                    if v < count:
                        crd = part[v]
                        assert all(type(x) is int for x in crd), crd
                        EXPECTED.append(
                            {
                                "label": label,
                                "role": role,
                                "thread": t,
                                "value": v,
                                "coord": list(crd),
                            }
                        )


@dsl_user_op
def all_cases(*, loc=None):
    # Building atoms here uses the real target's compiler context. No GPU is launched.
    if TARGET == "sm_80":
        shapes = [(16, 8, k) for k in (8, 16)]
        sources = [None]
    elif TARGET == "sm_90a":
        shapes = [(64, n, 16) for n in range(8, 257, 8)]
        sources = [warpgroup.OperandSource.SMEM, warpgroup.OperandSource.RMEM]
    else:
        shapes = [(m, n, 16) for m in (64, 128) for n in range(8, 257, 8)]
        sources = [tcgen05.OperandSource.SMEM, tcgen05.OperandSource.TMEM]
    for shape in shapes:
        for dtype, acc in [
            (cutlass.Float16, cutlass.Float32),
            (cutlass.BFloat16, cutlass.Float32),
            (cutlass.Float16, cutlass.Float16),
        ]:
            for source in sources:
                if TARGET == "sm_80":
                    op = warp.MmaF16BF16Op(dtype, acc, shape)
                elif TARGET == "sm_90a":
                    op = warpgroup.MmaF16BF16Op(
                        dtype, acc, shape, source, OperandMajorMode.K, OperandMajorMode.K
                    )
                else:
                    op = tcgen05.MmaF16BF16Op(
                        dtype,
                        acc,
                        shape,
                        tcgen05.CtaGroup.ONE,
                        source,
                        OperandMajorMode.K,
                        OperandMajorMode.K,
                    )
                tiled = cute.make_tiled_mma(op)
                label = f"{shape}/{dtype}/{acc}/{source}"
                before = len(ir.InsertionPoint.current.block.operations)
                cuteviz.inspect_mma(label, tiled)
                assert len(ir.InsertionPoint.current.block.operations) == before
                sample_mma(label, tiled)
    # Exercise tiled replication and layout permutations separately from atom shapes.
    if TARGET != "sm_100a":
        op = (
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
            if TARGET == "sm_80"
            else warpgroup.MmaF16BF16Op(
                cutlass.Float16,
                cutlass.Float32,
                (64, 32, 16),
                warpgroup.OperandSource.RMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        tiled = cute.make_tiled_mma(op, atom_layout_mnk=(2, 1, 1))
        cuteviz.inspect_mma("tiled", tiled)
        sample_mma("tiled", tiled)
    if TARGET != "sm_80":
        module = warpgroup if TARGET == "sm_90a" else tcgen05
        args = [cutlass.Float16, cutlass.Float32, (64, 32, 16)]
        if TARGET == "sm_100a":
            args.append(tcgen05.CtaGroup.ONE)
        tiled = cute.make_tiled_mma(
            module.MmaF16BF16Op(
                *args, module.OperandSource.SMEM, OperandMajorMode.MN, OperandMajorMode.MN
            )
        )
        cuteviz.inspect_mma("transposed", tiled)
        sample_mma("transposed", tiled)
    # Copy ownership: verify source and destination partitions from coordinate tensors.
    for value_shape in [(1, 1), (2, 2), (4, 1)]:
        cp = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32),
            cute.make_layout((4, 8), stride=(8, 1)),
            cute.make_layout(value_shape),
        )
        label = f"copy/{value_shape}"
        cuteviz.inspect_copy(label, cp)
        shape = tuple(int(cute.size(s)) for s in cp.tiler_mn)
        coords = cute.make_identity_tensor(shape)
        for t in range(32):
            for role in "SD":
                part = getattr(cp.get_slice(t), f"partition_{role}")(coords)
                for v in range(int(cute.size(part))):
                    EXPECTED.append(
                        {
                            "label": label,
                            "role": role,
                            "thread": t,
                            "value": v,
                            "coord": list(part[v]),
                        }
                    )
    # ldmatrix redistributes elements across threads: coordinate correspondence
    # cannot be implemented by pairing identical source/destination (thread,value).
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)))
    for transpose in (False, True):
        atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4), cutlass.Float16
        )
        cp = cute.make_tiled_copy_A(atom, mma)
        label = f"ldmatrix/{transpose}"
        cuteviz.inspect_copy(label, cp)
        shape = tuple(int(cute.size(s)) for s in cp.tiler_mn)
        coords = cute.make_identity_tensor(shape)
        for t in range(32):
            for role in "SD":
                part = getattr(cp.get_slice(t), f"partition_{role}")(coords)
                for v in range(int(cute.size(part))):
                    EXPECTED.append(
                        {
                            "label": label,
                            "role": role,
                            "thread": t,
                            "value": v,
                            "coord": list(part[v]),
                        }
                    )


@cute.kernel
def kernel():
    all_cases()


@cute.jit
def entry():
    kernel().launch(grid=(1, 1, 1), block=(256, 1, 1))


if __name__ == "__main__":
    with cuteviz.capture(sys.argv[1]):
        cute.compile(entry)
    with open(sys.argv[2], "w") as stream:
        json.dump(EXPECTED, stream)
