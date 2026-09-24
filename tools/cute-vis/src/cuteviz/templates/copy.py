import cutlass
import cutlass.cute as cute
import cutlass.memory

import cuteviz


@cute.kernel
def copy_kernel(src: cute.Pointer, dst: cute.Pointer, probes: cutlass.Constexpr):
    gA = cute.make_tensor(src, cute.make_layout((8, 8), stride=(8, 1)))
    gB = cute.make_tensor(dst, cute.make_layout((8, 8), stride=(8, 1)))
    sA = cutlass.memory.SmemAllocator().allocate_tensor(cutlass.Float32, gA.layout)
    tiled_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32),
        cute.make_layout((4, 8), stride=(8, 1)),
        cute.make_layout((2, 1)),
    )
    if cutlass.const_expr(probes):
        cuteviz.inspect("gA", gA)
        cuteviz.inspect("sA", sA)
        cuteviz.inspect_copy("load_A", tiled_copy, src=gA, dst=sA)
        cuteviz.inspect("nested", cute.make_layout(((2, 4), 8), stride=((32, 8), 1)))
        swizzled = cute.make_tensor(
            sA.iterator, cute.make_composed_layout(cute.make_swizzle(2, 0, 3), 0, sA.layout)
        )
        cuteviz.inspect("swizzled_sA", swizzled, parent="sA", transform="Swizzle<2,0,3>")
        tile = cute.local_tile(gA, (4, 4), (1, 0))
        cuteviz.inspect("bottom_left", tile, parent="gA", transform="local_tile((4,4), (1,0))")
    thread_copy = tiled_copy.get_slice(cute.arch.thread_idx()[0])
    cute.copy(tiled_copy, thread_copy.partition_S(gA), thread_copy.partition_D(sA))
    cute.arch.sync_threads()
    cute.copy(tiled_copy, thread_copy.partition_S(sA), thread_copy.partition_D(gB))


@cute.jit
def entry(src: cute.Pointer, dst: cute.Pointer, probes: cutlass.Constexpr = True):
    copy_kernel(src, dst, probes).launch(grid=(1, 1, 1), block=(32, 1, 1))


def make_args():
    # Compile with pointer placeholders; the app does not launch this kernel.
    pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return (pointer, pointer)
