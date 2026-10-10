"""Capture a CTA tile and recover its unprobed global backing tensor."""

import cutlass
import cutlass.cute as cute

import cuteviz


@cute.kernel
def tile_kernel(pointer: cute.Pointer):
    global_tensor = cute.make_tensor(pointer, cute.make_layout((32, 64), stride=(64, 1)))
    block_m, block_n, _ = cute.arch.block_idx()
    tile = cute.local_tile(global_tensor, (8, 16), (block_m, block_n))
    # The original tensor is recovered from this view's CuTe compilation IR.
    cuteviz.inspect("cta_tile", tile)
    column = tile[None, 3]
    cuteviz.inspect("tile_column", column)


@cute.jit
def entry(pointer: cute.Pointer):
    tile_kernel(pointer).launch(grid=(4, 4, 1), block=(32, 1, 1))


def make_args():
    return (cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16),)
