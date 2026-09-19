# Chapter 3 — Profile, optimize, and measure Qwen3

Read the [background theory and references](background.md), then run these
notebooks in order, each from a fresh kernel:

1. [Lab 1 — Diagnose the performance gap](code/lab.ipynb): frozen predictions,
   ordinary timings, PyTorch Profiler, Nsight Systems, and Nsight Compute.
2. [Lab 2 — Implement and integrate optimizations](code/lab2.ipynb): CuTe
   normalization/activation/rotary fusions and causal GQA attention, correctness,
   integration, ablations, and repeated profiling.
3. [Lab 3 — Measure the optimized model](code/lab3.ipynb): matched 8B/32B sweeps,
   rooflines, latency/throughput overlays, speedups and Amdahl explanations.

Exercises and completion criteria live in the notebooks. Reusable
[optimized model](code/model_opt.py) and [CuTe kernels](code/cute_kernels.py)
extend Chapter 1; [shared estimators and timing](../../shared/performance.py)
are also used by Chapter 2. Follow the [setup](../../shared/SETUP.md) and
[experiment protocol](../../shared/PROTOCOL.md). Nsight Compute counter permission
failures must remain explicit; they do not replace other required measurements.

[Previous](../02_performance_model/background.md) · [Course](../../README.md) ·
[Next](../04_runtime_and_kv/README.md)
