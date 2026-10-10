
from contextlib import contextmanager
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
    # args = tuple(_tensor(t) for t in tensors)
    # mark_layout_dynamic preserves ZERO strides as compile-time constants.
    # Broadcast positions [B,T] therefore have a different launch ABI from
    # an unbroadcast B=1 tensor. Never share those compiled argument packs.
    key = (
        name,
        tuple(
            (
                len(t.shape),
                # str(t.value_type),
                tuple(i for i, stride in enumerate(t.stride) if stride == 0),
                tuple(i for i, stride in enumerate(t.stride) if stride == 1),
            )
            for t in tensors
        ),
        constants,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in COMPILED:
        begin = time.perf_counter()
        COMPILED[key] = cute.compile(fn, *tensors, *constants, stream)
        COMPILATIONS.append(
            dict(kernel=name, signature=str(key), seconds=time.perf_counter() - begin)
        )
    COMPILED[key](*tensors, stream)
    return COMPILED[key], tensors, stream

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
    """Return ms per call using preallocated workspaces and CUDA-graph replay.

    Allocate input, output, and scratch buffers before calling this helper, or
    supply a zero-argument ``workspace_generator`` that creates them once and
    returns ``cute_testing.JitArguments`` for ``fn``. A factory can also prepare
    compiled kernels and arguments. It runs on the capture stream, outside
    capture/timing. Workspaces stay alive until all replays complete.

    ``fn`` must enqueue capture-safe work into those reusable buffers. It must
    not allocate output/scratch on each call, synchronize, read GPU scalars on
    the host, or rely on Python state changing during replay. Without a factory,
    ``fn`` takes no arguments and closes over its already allocated workspace.

    Each workspace gets ``warmup`` eager calls, including compilation. One graph
    captures ``inner`` calls rotating through the workspaces. One untimed replay
    initializes it; all ``repeats`` reuse that graph and its buffers. CUDA events
    surround each replay; the end event is synchronized. Divide by ``inner`` to
    report ms/call. Setup, allocation, capture, and per-call Python dispatch are
    excluded. GPU work and amortized graph-launch overhead are included.

    Work must use the current stream during preparation, warmup, and capture.
    An explicit non-default ``stream`` is used directly; otherwise an internal
    non-default stream is ordered after the selected stream. The caller's stream
    is restored. Establish dependencies on any other producer streams yourself.

    One workspace measures reused-buffer performance. Rotating enough distinct
    workspaces to exceed L2 capacity can reduce cache reuse; it is not a cache
    flush. ``inner`` must cover every workspace. Use benchmark_cuda_cold for
    eviction before each single-call graph replay. Check correctness separately.

    Example: benchmark_cuda(lambda: torch.square(x, out=out), repeats=5)
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if any(type(n) is not int or n < 1 for n in (warmup, repeats, inner, workspace_count)):
        raise ValueError("warmup, repeats, inner, and workspace_count must be positive integers")
    with _capture_benchmark(
        fn, warmup=warmup, inner=inner, device=device, stream=stream,
        workspace_generator=workspace_generator, workspace_count=workspace_count,
    ) as (graph, selected_stream):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Materialize CUDA event handles before collecting the first sample.
        start.record(selected_stream)
        end.record(selected_stream)
        end.synchronize()
        samples = []
        for _ in range(repeats):
            start.record(selected_stream)
            graph.replay()
            end.record(selected_stream)
            end.synchronize()
            samples.append(start.elapsed_time(end) / inner)
        return samples


@contextmanager
def _capture_benchmark(
    fn, *, warmup, inner, device, stream, workspace_generator, workspace_count,
):
    """Own prepared workspaces and one graph through synchronized measurement."""
    if workspace_generator is not None and not callable(workspace_generator):
        raise TypeError("workspace_generator must be callable")
    if workspace_generator is None and workspace_count != 1:
        raise ValueError("Multiple workspaces require workspace_generator")
    if inner < workspace_count:
        raise ValueError("inner must cover every workspace")
    selected = _benchmark_stream(device, stream)
    capture_stream = selected
    if selected == torch.cuda.default_stream(selected.device):
        capture_stream = torch.cuda.Stream(device=selected.device)
        capture_stream.wait_stream(selected)
    with torch.inference_mode(), torch.cuda.device(selected.device), torch.cuda.stream(capture_stream):
        workspaces = (
            [workspace_generator() for _ in range(workspace_count)]
            if workspace_generator is not None else [cute_testing.JitArguments()]
        )
        for workspace in workspaces:
            if type(workspace) is not cute_testing.JitArguments:
                raise TypeError("workspace_generator must return cute_testing.JitArguments")
        for workspace in workspaces:
            for _ in range(warmup):
                fn(*workspace.args, **workspace.kwargs)
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=capture_stream):
            for index in range(inner):
                workspace = workspaces[index % workspace_count]
                fn(*workspace.args, **workspace.kwargs)
        status, _, nodes = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()))
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Unable to inspect captured benchmark graph: {status}")
        if nodes < inner:
            raise ValueError(
                "Benchmark must capture CUDA work for every call; enqueue work "
                "on the current capture stream (prepare arguments in workspace_generator)"
            )
        # First launch/upload is outside timing. Keep all workspace owners live.
        graph.replay()
        capture_stream.synchronize()
        try:
            yield graph, capture_stream
        finally:
            capture_stream.synchronize()


def benchmark_cuda_cold(
    fn, *, warmup_ms=25, repeat_ms=100, device=None, stream=None,
    workspace_generator=None,
):
    """Time a prepared single-call CUDA graph with cache eviction between calls.

    The preallocation, factory, stream, and capture contracts match
    ``benchmark_cuda``. One workspace and one graph are prepared once. Triton
    times ``graph.replay`` with CUDA events on the capture stream. It handles
    warmup, repeat-count estimation, and synchronization. Its NVIDIA backend
    writes a 256 MiB eviction buffer before each timed call, outside the event
    interval. This is a cache-eviction technique, not a hardware cache reset;
    torch.cuda.empty_cache() would only release unused allocator memory.

    warmup_ms/repeat_ms are requested time budgets, not iteration counts.
    Sample count is adaptive. Triton synchronizes the selected device, so run
    benchmarks without competing GPU workloads. Each sample includes one graph
    submission and GPU execution, but excludes workspace allocation, compilation,
    capture, and eviction. There is no per-call Python kernel/argument setup.
    This helper requires Triton; kernels need not be implemented in Triton.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if any(not math.isfinite(n) or n <= 0 for n in (warmup_ms, repeat_ms)):
        raise ValueError("warmup_ms and repeat_ms must be positive finite durations")
    from triton.testing import do_bench

    with _capture_benchmark(
        fn, warmup=10, inner=1, device=device, stream=stream,
        workspace_generator=workspace_generator, workspace_count=1,
    ) as (graph, _):
        return do_bench(graph.replay, warmup=warmup_ms, rep=repeat_ms, return_mode="all")
