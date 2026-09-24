import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import platform
import statistics
import sys
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


# Reuse course artifact helpers, including when run from this folder.
COURSE_ROOT = Path(__file__).resolve().parents[4]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
from shared import experiment

# Direct execution places this file's directory on Python's import path.
from utils import _launch, _check_bf16, benchmark_cuda

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
def _rms_norm_kernel(mX, weight, mY, rows: cutlass.Int32, width: cutlass.Constexpr, eps: cutlass.Constexpr, tv_layout: cute.Layout):
    # kernel 
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    
    # get CTA local tile
    blk_coord = (None, bidx)
    gX = mX[blk_coord]
    gY = mY[blk_coord]

    # get thread local vals
    w_tv_layout = cute.make_ordered_layout((32, cute.ceil_div(width, 32)), order=(1, 0))
    tidfrgX = cute.composition(gX, tv_layout)
    tidfrgY = cute.composition(gY, tv_layout)
    tidfrgW = cute.composition(weight, w_tv_layout)
    thr_coord = (tidx, None)
    thrX = tidfrgX[thr_coord]
    thrY = tidfrgY[thr_coord]
    # Each warp reuses the same weight vector; its thread coordinate is a lane.
    thrW = tidfrgW[(lane, None)]

    print('gX', gX)
    print('gY', gY)
    print('thrX', thrX)
    print('thrY', thrY)
    print('thrW', thrW)

    # A block owns four rows. Skip missing rows in the final block; the
    # predicate is identical for all lanes participating in a warp shuffle.
    if bidx * 4 + tidx // 32 < rows:
        # BF16 storage, FP32 arithmetic, and one rounding step at the store.
        v = thrX.load().to(cutlass.Float32)
        squared_sum = cutlass.Float32(0)
        for i in cutlass.range_constexpr(thrX.shape[0]):
            squared_sum += v[i] * v[i]

        scale = cute.rsqrt(warp_sum(squared_sum) / width + eps)
        # TODO: load weight to shared mem first
        w = thrW.load().to(cutlass.Float32)

        assert v.shape == w.shape
        thrY[None] = (v * scale * w).to(cutlass.BFloat16)


@cute.jit
def _rms_norm_launcher(x, weight, y, width: cutlass.Constexpr, eps: cutlass.Constexpr, stream: cuda.CUstream):
    M = 4 # M rows per CTA
    # coalesced_bytes = 128
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, cute.ceil_div(width, 32),), order=(1, 0))
    tiler_m, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print('tiler_m', tiler_m)
    print('tv_layout', tv_layout)

    mX = cute.zipped_divide(x, tiler_m)
    mY = cute.zipped_divide(y, tiler_m)
    _rms_norm_kernel(
        mX, weight, mY, x.shape[0], width, eps, tv_layout
    ).launch(
        grid=((x.shape[0]+M-1)//M, 1, 1),
        block=(32*M, 1, 1), # one row per warp, M warps
        stream=stream
    )

def rms_norm(x, weight, eps):
    """Return a new BF16 tensor normalized with FP32 arithmetic on CUDA.

    Width must be divisible by 32, and leading dimensions must be collapsible
    with view(). Each lane loads width / 32 consecutive columns of one row.
    """
    _check_bf16(x, weight)
    if weight.shape != (x.shape[-1],) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Weight width and positive finite epsilon required")

    width = x.shape[-1]
    if width % 32:
        raise ValueError("This vectorized RMSNorm kernel requires width divisible by 32")
    y = torch.empty_like(x)
    _launch(
        'rms_norm',
        _rms_norm_launcher,
        (x.view(-1, width), weight, y.view(-1, width)),
        (width, eps)
    )
    return y

def rms_norm_ref(x, weight, eps):
    """FP32 numerical oracle before the kernel's final BF16 storage rounding."""
    x_fp32 = x.float()
    scale = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return x_fp32 * scale * weight.float()


# run correctness check
def correctness_check():
    # build torch tensor of shape (M, N, 128), then compare the result of torch reference rms norm and kernel
    # use bf16 input
    M, N, D = 1024, 16, 128
    torch.manual_seed(2026)
    
    x = torch.randn((M, N, D), dtype=torch.bfloat16, device='cuda:0')
    weight = torch.randn(D, dtype=torch.bfloat16, device='cuda:0')
    eps = 1e-6
    norm_ref = rms_norm_ref(x, weight, eps)
    norm = rms_norm(x, weight, eps)
    # BF16 round-to-nearest contributes up to about 1/256 relative error.
    torch.testing.assert_close(norm.float(), norm_ref, rtol=4e-3, atol=1e-5)
    print("RMSNorm correctness passed against the FP32 oracle.")


def benchmark(
    shape=(1024, 16, 128), *, eps=1e-6, device="cuda:0", warmup=10,
    repeats=20, inner=100, seed=2026, out=None,
):
    """Compare eager Torch and CuTe function calls with CUDA events.

    Both paths use BF16 input/output and FP32 arithmetic. The Torch baseline
    includes rms_norm_ref(...).to(torch.bfloat16). Timing includes allocations
    and possible GPU idle gaps from Python dispatch, but excludes compilation,
    correctness checks, fixture creation, and warmup. No CUDA graphs are used.

    CuTe's testing utility performs ``warmup`` calls before each repeat.
    Each raw sample is the average of ``inner`` calls. Save metadata, a
    prediction, raw samples, and a summary in a new/empty ``out`` directory;
    the default is a timestamped directory under the course's results/ folder.
    Return the summary dictionary. See shared/PROTOCOL.md for timing scope.
    """
    shape = tuple(shape)
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("shape must contain positive integer dimensions")
    if shape[-1] % 32 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Width divisible by 32 and positive finite epsilon required")
    if any(type(n) is not int for n in (warmup, repeats, inner)):
        raise ValueError("warmup, repeats, and inner must be integers")
    if warmup < 10 or repeats < 3 or inner < 1:
        raise ValueError("Use warmup >= 10, repeats >= 3, and inner >= 1")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(out) if out is not None else COURSE_ROOT / "results" / f"rms_norm-{timestamp}"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output directory must be new or empty; preserve previous results")

    with torch.inference_mode(), torch.cuda.device(device):
        hardware = experiment.cuda_environment(device)
        driver_status, driver_version = cuda.cuDriverGetVersion()
        hardware["cuda_driver_version"] = (
            driver_version if driver_status == cuda.CUresult.CUDA_SUCCESS else None
        )
        out.mkdir(parents=True, exist_ok=True)
        manifest = dict(
            result_kind="measured synthetic CUDA RMSNorm workload; no model checkpoint",
            model_id=None, model_revision=None, timestamp_utc=timestamp,
            shape=list(shape), eps=eps, seed=seed, hardware=hardware,
            python=platform.python_version(), platform=platform.platform(),
            dependencies={name: version(name) for name in ("nvidia-cutlass-dsl", "cuda-python")},
            input_dtype="bfloat16", output_dtype="bfloat16", arithmetic_dtype="float32",
            backends=["torch eager + BF16 output cast", "CuTe DSL public rms_norm function"],
            warmup=warmup, repeats=repeats, calls_per_repeat=inner, cuda_graphs=False,
            timing_backend="cutlass.cute.testing.benchmark via utils.benchmark_cuda",
            warmup_policy="one initialization call, then warmup calls before each repeat",
            timing="CUDA events on current stream; end event synchronized; ms per call",
            includes="public function calls, allocation effects, and possible dispatch gaps",
            excludes="fixture creation, correctness checks, JIT compilation, and warmup",
            cache_policy="reuse the same input and weight tensors; no cache flush",
            correctness=dict(oracle="FP32 rms_norm_ref", rtol=4e-3, atol=1e-5),
            source_sha256={
                str(path.relative_to(COURSE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__).resolve(), Path(__file__).with_name("utils.py").resolve(),
                             COURSE_ROOT / "shared/experiment.py")
            },
        )
        experiment.write_json(out / "manifest.json", manifest)
        experiment.write_json(out / "prediction.json", dict(
            hypothesis="CuTe should benefit from fusing the reduction and scaling into one kernel, avoiding Torch's global FP32 intermediates; dispatch overhead can dominate small inputs.",
            fused_logical_min_bytes=4 * math.prod(shape) + 2 * shape[-1],
            byte_derivation="2 bytes per input + 2 per output + one 2-byte weight vector; minimum logical traffic, not measured DRAM traffic",
            expected_speedup=None,
            prediction_scope="qualitative; no calibrated bandwidth or launch-overhead model",
        ))

        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        weight = torch.randn(shape[-1], dtype=torch.bfloat16, device=device, generator=generator)
        functions = {
            "torch_reference_bf16": lambda: rms_norm_ref(x, weight, eps).to(x.dtype),
            "cute_kernel_bf16": lambda: rms_norm(x, weight, eps),
        }
        # This first CuTe call compiles the kernel before any timing starts.
        expected = rms_norm_ref(x, weight, eps)
        for fn in functions.values():
            torch.testing.assert_close(fn().float(), expected, rtol=4e-3, atol=1e-5)
        del expected

        samples = {
            name: benchmark_cuda(fn, repeats=repeats, inner=inner, warmup=warmup, device=device)
            for name, fn in functions.items()
        }

    rows = [
        dict(backend=name, repeat=i, calls=inner, cuda_ms=ms)
        for name, values in samples.items() for i, ms in enumerate(values)
    ]
    timings = {
        name: dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values))
        for name, values in samples.items()
    }
    speedup = timings["torch_reference_bf16"]["median_ms"] / timings["cute_kernel_bf16"]["median_ms"]
    summary = dict(shape=list(shape), timings=timings, speedup=speedup, output_dir=str(out.resolve()))
    experiment.write_csv(out / "results.csv", rows)
    experiment.write_json(out / "summary.json", summary)
    print(f"\nRMSNorm {shape} on {hardware['gpu_name']}; BF16 input/output, FP32 arithmetic")
    print(f"CUDA-event function latency: {warmup} warmups/repeat, {repeats} repeats, {inner} calls/repeat")
    print(f"{'Backend':<25} {'Median (us)':>12} {'Min (us)':>12} {'Max (us)':>12}")
    for name, timing in timings.items():
        print(f"{name:<25} {timing['median_ms'] * 1000:12.3f} {timing['min_ms'] * 1000:12.3f} {timing['max_ms'] * 1000:12.3f}")
    print(f"CuTe speedup (Torch / CuTe): {speedup:.2f}x")
    print("Torch includes the BF16 output cast; both timings may include Python dispatch gaps.")
    print(f"Saved raw measurements and metadata to {out.resolve()}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check or benchmark the experimental RMSNorm kernel")
    parser.add_argument("--benchmark", action="store_true", help="Run correctness gates, then compare Torch and CuTe latency")
    parser.add_argument("--shape", type=int, nargs="+", default=(1024, 16, 128))
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inner", type=int, default=100, help="Calls per timed repeat")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, help="New or empty result directory")
    args = parser.parse_args()
    if args.benchmark:
        options = vars(args).copy()
        options.pop("benchmark")
        benchmark(**options)
    else:
        correctness_check()
