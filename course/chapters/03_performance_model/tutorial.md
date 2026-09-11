# Tutorial: predict unseen inference workloads

## 1. Build an operation ledger (4 hours)

Trace one P1/P2 forward at B=1 and B=4. List each sequential operator, logical
shape, dtype, actual backend, fusion boundary, and phase. For Qwen3-32B, verify
Q `[M,5120] @ [5120,8192]` and O `[M,8192] @ [8192,5120]` independently.
Include final vocabulary projection, normalization, RoPE, activations, gathers,
sampling, and CPU scheduling even when the first model omits them.

Calculate FLOPs and logical bytes on paper for one Q projection, one SwiGLU, and
one attention call. Use `p*t+t*(t+1)/2` for chunked prefill. Mark whether S includes
the currently processed decode token; use that convention everywhere.

## 2. Run an explicit assumption model (2 hours)

```bash
python chapters/03_performance_model/code/lab.py --model 8b --phase decode --batch 1 --context 2048 --tflops 50 --gbps 200 --launch-us 5
python chapters/03_performance_model/code/lab.py --model 32b --phase prefill --batch 1 --context 256 --tflops 50 --gbps 200 --launch-us 5
```

**50 TFLOP/s, 200 GB/s, and 5 us are invented inputs**, provided to learn the
calculator. Outputs are modeled CSV rows, never measured benchmark results. The
sample sums separate projections and idealized fused attention; it omits norms,
RoPE, activations, scheduling, KV writes, and backend-specific rereads. It applies
the prefill vocabulary head only to the last position per request.

Double BW, then C, keeping everything else fixed. Which rows change? Explain
why a global doubling of peak compute need not halve decode time.

## 3. Calibrate hardware and shapes (6 hours)

Begin the measurement workflow on DGX Spark using
[the standalone CUDA experiment](standalone.md). Its `code/experiment.py`
uses CUDA events and records the actual GPU identity. CPU `lab.py` checks
are optional correctness preparation; they supply no timing baseline.
The remaining steps below extend the reference workload to the full project.

On the course GPU environment, run:

```bash
python chapters/03_performance_model/code/benchmark_gemm.py --m 4 --k 4096 --n 4096
```

Repeat for M=1,4,16,128,512 and selected Q/O/MLP shapes. The sample preallocates
the output, warms up, and reports each CUDA-event repeat. Allocate enough work
per timed region to amortize event overhead; record inner count and variability.
Repeated weights may be cache-resident: compare rotating buffers or the actual
model trace before treating a small GEMM's rate as a DRAM bandwidth estimate.

Add a streaming copy experiment using buffers larger than the GPU's last-level
cache. For `destination.copy_(source)`, logical traffic is one read plus one write;
compute `2*numel*element_size / seconds`. Avoid host transfers in that benchmark.
Measure a small attention shape set and a dispatch-heavy operation to estimate
launch/CPU effects. Capture actual GPU clock/thermal state and memory conditions.

**Checkpoint:** a calibration table contains shape-specific rates, counts,
uncertainty, and hardware identity. Do not use advertised sparse/low-precision
throughput for a dense BF16 workload.

## 4. Freeze held-out predictions (4 hours)

Choose six feasible points before measuring their full-model latency, for example:
8B decode `(B,S)=(2,512),(2,4096),(8,2048)`; 8B prefill `(1,1024)`;
32B decode `(1,2048)`; 32B prefill `(1,512)`. If capacity prevents a point, replace
it now and record the reason. These are inference conditions withheld from
full-model calibration; using a new component microbenchmark is extra calibration.

Replace illustrative rates with your measurements and add the missing component
terms supported by the operation ledger. Save predicted CSV and the current Git
revision before running those points. A simple lookup/interpolation model is fine;
state how it handles unseen GEMM shapes.

## 5. Measure and investigate residuals (6 hours)

Run three independent uninstrumented repeats per held-out point. Separate prefill
from steady decode and report the exact number/context of decode steps. Save all
rows, then plot prediction versus observation with an identity line. Report median
relative error, per-point error, and uncertainty.

Use a separate Nsight Systems run on the largest residual. Trace model phases
with NVTX ranges in your engine. Select only a few dominant kernels for Nsight
Compute; replay/profiling overhead invalidates those timings as ordinary latency.
Distinguish idle GPU time, gathers, redundant KV reads, and inefficient GEMMs.

## 6. Optimize one diagnosed term (5 hours)

Select an intervention with an explicit cause: reducing gathers, batching small
GEMMs, fusing an elementwise path, or removing an unnecessary synchronization.
Predict its effect by changing only the corresponding model term. Recheck
correctness and run the matched workload. Preserve the original held-out score;
evaluate the revised model on additional conditions to avoid presenting a fitted
result as an unseen prediction.

## 7. Submit (5 hours)

Include the operation ledger, calibration data, frozen predictions, six-point
comparison, one annotated trace, and an explained intervention. The syllabus's
25% median relative-error target is diagnostic. Missing it with a clear model
failure and a discriminating next experiment can satisfy the reasoning objective.

**Defense:** How can achieved FLOP/s and achieved bandwidth both be low? Why is
the sum of kernel rooflines different from a roofline for the summed work?
