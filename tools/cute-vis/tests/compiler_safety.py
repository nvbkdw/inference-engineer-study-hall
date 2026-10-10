"""Probe neutrality and diagnostics in real CuTeDSL compilation contexts."""

import os
import re
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import dsl_user_op

import cuteviz
from cuteviz.model import Capture


@dsl_user_op
def neutral(layout, tensor, cp, mma, dynamic, *, loc=None):
    block = ir.InsertionPoint.current.block
    before = str(block)
    assert cuteviz.inspect("layout", layout) is None
    assert cuteviz.inspect("tensor", tensor) is None
    assert cuteviz.inspect_copy("copy", cp, src=tensor, dst=tensor) is None
    assert cuteviz.inspect_mma("mma", mma) is None
    assert cuteviz.inspect("dynamic", dynamic) is None
    assert str(block) == before, "A probe mutated the kernel's IR"


@cute.kernel
def kernel(g: cute.Tensor, runtime_stride: cutlass.Int32, enabled: cutlass.Constexpr):
    layout = cute.make_layout((8, 8), stride=(8, 1))
    cp = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32),
        cute.make_layout((4, 8), stride=(8, 1)),
        cute.make_layout((2, 1)),
    )
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)))
    dynamic = cute.make_layout((8, 8), stride=(runtime_stride, 1))
    if cutlass.const_expr(enabled):
        neutral(layout, g, cp, mma, dynamic)
    tid = cute.arch.thread_idx()[0]
    g[tid, 0] = g[tid, 0] + 1.0


@cute.jit
def entry(g: cute.Tensor, runtime_stride: cutlass.Int32, enabled: cutlass.Constexpr):
    kernel(g, runtime_stride, enabled).launch(grid=(1, 1, 1), block=(8, 1, 1))


@cute.kernel
def unsupported_kernel():
    cuteviz.inspect("before", cute.make_layout((4, 4)))
    tiled = cute.make_tiled_mma(warp.MmaTF32Op((16, 8, 8)))
    cuteviz.inspect_mma("unsupported_tf32", tiled)
    cuteviz.inspect("after", cute.make_layout((8, 8)))


@cute.jit
def unsupported_entry():
    unsupported_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.jit
def cached_entry():
    cuteviz.inspect("cached", cute.make_layout((8, 8)))


if __name__ == "__main__":
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_80")
    path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "cache":
        with cuteviz.capture(path):
            cached = cute.compile(cached_entry)
        try:
            with cuteviz.capture(path):
                cached()
        except cuteviz.EmptyCaptureError:
            assert Capture.load(path).outcome == "empty"
        else:
            raise AssertionError("Expected compiled-callable reuse to bypass probes")
        print("Compiled-callable reuse diagnostic: passed")
        raise SystemExit(0)
    fake = cute.runtime.make_fake_tensor(cutlass.Float32, (8, 8), (8, 1))
    with cuteviz.capture(path):
        observed = cute.compile(entry, fake, 8, True)
    baseline = cute.compile(entry, fake, 8, False)

    # Compare all PTX instructions; only symbol names and metadata may differ.
    def instructions(compiled):
        ptx = compiled.__ptx__
        if isinstance(ptx, dict):
            ptx = "\n".join(ptx.values())
        if isinstance(ptx, str) and Path(ptx).is_file():
            ptx = Path(ptx).read_text()
        match = re.search(r"\.entry\s+(\w+)", ptx)
        assert match, repr(ptx)
        symbol = match.group(1)
        ptx = ptx.replace(symbol, "kernel")
        return [
            s.strip()
            for s in ptx.splitlines()
            if s.strip().endswith(";") and not s.strip().startswith((".", "//"))
        ]

    assert instructions(observed) == instructions(baseline)
    capture = Capture.load(path)
    assert capture.outcome == "complete"
    dynamic = next(o for o in capture.objects if o.label == "dynamic")
    assert dynamic.views[0].layout.stride[0]["status"] == "symbolic"
    with cuteviz.capture(path):
        cute.compile(unsupported_entry)
    capture = Capture.load(path)
    assert capture.outcome == "partial"
    assert [o.label for o in capture.objects] == ["before", "unsupported_tf32", "after"]
    assert capture.objects[1].fields["adapter"].status == "unsupported"
    print("IR neutrality, PTX neutrality, dynamic fields, partial unsupported capture: passed")
