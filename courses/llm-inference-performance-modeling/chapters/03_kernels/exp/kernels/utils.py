
import time
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as cute_testing
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

# CuteDSL kernel compilation cache
COMPILED = {}
COMPILATIONS = []


def _tensor(x):
    return from_dlpack(x.detach(), assumed_align=64).mark_layout_dynamic(leading_dim=x.ndim - 1)


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


def _benchmark_stream(device=None, stream=None):
    """Select a CUDA stream and order it after the caller's current stream."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kernel benchmarks")
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError("stream must be a torch.cuda.Stream")
    selected = torch.device(device if device is not None else (stream.device if stream is not None else "cuda"))
    if selected.type != "cuda":
        raise ValueError("Benchmark device must be CUDA")
    if selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    if stream is not None and stream.device != selected:
        raise ValueError("stream and device must refer to the same CUDA device")
    current = torch.cuda.current_stream(selected)
    stream = current if stream is None else stream
    if stream != current:
        stream.wait_stream(current)
    return stream


def benchmark_cuda(
    fn, *, warmup=10, repeats=20, inner=100, device=None, stream=None,
    workspace_generator=None, workspace_count=1,
):
    """Return repeated mean latencies in ms using CuTe's testing.benchmark.

    ``fn`` is a zero-argument callable unless ``workspace_generator`` returns
    cute_testing.JitArguments for it. Workspaces are created once, then reused
    across repeats. Each is executed once to compile/initialize it outside
    timing. CuTe runs ``warmup`` calls before each group of ``inner`` timed
    calls, records CUDA events on ``stream``, and synchronizes the end event.
    No CUDA graphs are used for timing; CuTe 4.2.1 uses a capture-only stream
    check on non-default streams, so the callable must be capture-safe there.

    All work must run on the selected PyTorch stream (or join it before fn
    returns). Inputs from the caller's current stream are ordered before it;
    callers must establish dependencies on any other producer streams.

    One workspace measures repeated-input performance. Rotate enough distinct
    workspaces to exceed L2 capacity for CuTe's cache-control approach. This
    does not explicitly empty L2. For per-call eviction, use benchmark_cuda_cold.
    Correctness checks and fixture creation belong outside the callable.

    Example: samples_ms = benchmark_cuda(lambda: kernel(x, weight), repeats=5)
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if any(type(n) is not int or n < 1 for n in (warmup, repeats, inner, workspace_count)):
        raise ValueError("warmup, repeats, inner, and workspace_count must be positive integers")
    if workspace_generator is None and workspace_count != 1:
        raise ValueError("Multiple workspaces require workspace_generator")
    selected_stream = _benchmark_stream(device, stream)
    with torch.inference_mode(), torch.cuda.device(selected_stream.device), torch.cuda.stream(selected_stream):
        workspaces = (
            [workspace_generator() for _ in range(workspace_count)]
            if workspace_generator is not None else [cute_testing.JitArguments()]
        )
        for workspace in workspaces:
            if type(workspace) is not cute_testing.JitArguments:
                raise TypeError("workspace_generator must return cute_testing.JitArguments")
            fn(*workspace.args, **workspace.kwargs)
        selected_stream.synchronize()
        # CuTe returns microseconds; course observations consistently use ms.
        return [
            cute_testing.benchmark(
                fn, warmup_iterations=warmup, iterations=inner,
                stream=cuda.CUstream(selected_stream.cuda_stream),
                workspace_generator=iter(workspaces).__next__,
                workspace_count=workspace_count, use_cuda_graphs=False,
            ) / 1000.0
            for _ in range(repeats)
        ]


def benchmark_cuda_cold(fn, *, warmup_ms=25, repeat_ms=100, device=None, stream=None):
    """Return raw per-call milliseconds using triton.testing.do_bench.

    Triton handles initialization, warmup, repeat-count estimation, CUDA-event
    timing on the current stream, and synchronization. Its NVIDIA backend
    writes a 256 MiB eviction buffer before each timed call, outside the event
    interval. This is a cache-eviction technique, not a hardware cache reset;
    torch.cuda.empty_cache() would only release unused allocator memory.

    warmup_ms/repeat_ms are requested time budgets, not iteration counts.
    Sample count is adaptive. Triton synchronizes the selected device, so run
    benchmarks without competing GPU workloads. Inputs and stream ownership
    follow benchmark_cuda's contract. This helper requires Triton; kernels
    themselves do not need to be implemented in Triton.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if any(not math.isfinite(n) or n <= 0 for n in (warmup_ms, repeat_ms)):
        raise ValueError("warmup_ms and repeat_ms must be positive finite durations")
    from triton.testing import do_bench

    selected_stream = _benchmark_stream(device, stream)
    with torch.inference_mode(), torch.cuda.device(selected_stream.device), torch.cuda.stream(selected_stream):
        return do_bench(fn, warmup=warmup_ms, rep=repeat_ms, return_mode="all")
