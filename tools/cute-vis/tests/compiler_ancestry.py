"""Real compiler cases for backing coordinates, with probe IR neutrality checks."""

import sys

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op

import cuteviz


@dsl_user_op
def inspect_neutral(label, tensor, *, loc=None, ip=None):
    block = ir.InsertionPoint.current.block
    before = str(block)
    cuteviz.inspect(label, tensor)
    assert str(block) == before, "Ancestry inspection modified the kernel IR"


@cute.kernel
def kernel(pointer: cute.Pointer):
    g = cute.make_tensor(pointer, cute.make_layout((32, 64), stride=(64, 1)))
    static = cute.local_tile(g, (8, 16), (1, 2))
    inspect_neutral("static", static)
    bx, by, _ = cute.arch.block_idx()
    dynamic = cute.local_tile(g, (8, 16), (bx, by))
    inspect_neutral("dynamic", dynamic)
    projected = cute.local_tile(g, (8, 4, 16), (bx, 0, by), proj=(1, None, 1))
    inspect_neutral("projected", projected)
    rest = cute.local_tile(g, (8, 16), (None, 1))
    inspect_neutral("rest", rest)
    nested = cute.local_tile(static, (2, 4), (1, 2))
    inspect_neutral("nested", nested)
    column = dynamic[None, 3]
    inspect_neutral("column", column)
    shifted = cute.local_tile(g, (4, 8), (bx * 2 + 1, by))
    inspect_neutral("shifted", shifted)
    edge_root = cute.make_tensor(pointer, cute.make_layout((10, 11), stride=(11, 1)))
    edge = cute.local_tile(edge_root, (4, 4), (2, 2))
    inspect_neutral("edge", edge)
    gather = cute.local_tile(g, (cute.make_layout(4, stride=2), 8), (0, 0))
    inspect_neutral("unsupported_gather", gather)


@cute.jit
def entry(pointer: cute.Pointer):
    kernel(pointer).launch(grid=(4, 4, 1), block=(32, 1, 1))


with cuteviz.capture(sys.argv[1]):
    pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    cute.compile(entry, pointer)
