# Validation of the DGX Spark measurement path

Validated on 2026-09-11. The course now defaults to DGX Spark GPU measurements.
CPU fixtures remain optional correctness/modeling aids; historical CPU timings
are not the baseline for the revised labs.

## Hardware and environment used

- NVIDIA GB10, compute capability 12.1, one visible GPU.
- Driver 580.126.09; CUDA runtime 13.0.
- Python 3.12.3; PyTorch 2.13.0+cu130.
- Matplotlib 3.11.1 and NumPy 2.5.3 in an isolated course-local dependency directory.
- The existing working CUDA PyTorch environment was used without changing its
  installed packages, drivers, or GPU clocks.

GPU identity, runtime, reported memory, source hashes, warmup, seed, and measurement
boundary are saved in each run's manifest. The validation runs are smoke/correctness
and artifact checks, not a controlled published GB10 performance characterization.

## Executed on Spark

The P1, P3, P4, and P6 `code/experiment.py` commands ran successfully on the GB10
with three repeats, using CUDA events and CUDA tensors:

| Chapter | Checked workload and evidence |
|---|---|
| P1 | Scaled two-layer FP32 CUDA model; cached/recomputed next logits agree; five cache-byte checks at S=128–2048; 30 timing rows |
| P3 | BF16 K=N=4096 GEMMs; 128 MiB copy calibration; predictions saved before six held-out shapes; 24 prediction/measurement rows |
| P4 | FP32 CUDA attention with R=128; dense/tiled outputs agree for 12 cases; 36 timing rows and score-storage accounting |
| P6 | FP32 CUDA 1024x4096 random layers; nibble roundtrip, output errors, storage, and reconstruction/GEMM timing; 30 error rows |

P2 and P5 model exercises also ran. They retain invented service/cycle costs and
perform **no hardware timing**. Their output is explicitly simulation/numerical
modeling, not GPU or CPU performance measurement.

Raw migration-run artifacts are under `results/spark-migration/` in the authoring
workspace. They are ignored experiment outputs; regenerate them using the chapter
commands when copying the course elsewhere.

## Test results

In the Spark CUDA environment, the suite discovered **16 tests: 14 passed and
2 were explicitly skipped**. The skips were the optional Transformers checkpoint
workflow (those optional packages were not in the GPU environment) and the
two-physical-GPU NCCL measurement (this machine has one GB10).

The CUDA test executed all four measured sweeps and checked manifests, row counts,
numerical gates, predictions, CSV, and SVG/PNG artifacts. Additional tests verify
that `--device cpu` is rejected for measurement and that a single Spark cannot
supply two local GPU ranks.

In the existing CPU correctness environment, **13 passed and 3 were skipped**:
single-GPU measurement, the one-Spark GPU guard exercise, and two-GPU NCCL.
The optional offline checkpoint workflow passed there using Python 3.13.9,
PyTorch 2.14.0+cpu, Transformers 5.17.0, safetensors 0.8.0, and Accelerate 1.15.0.
It remains a numerical loader check, not a CPU performance baseline.

## Not established by these checks

No real 8B/32B weights, model-quality evaluation, full serving benchmark, custom
CuTe kernel, or packed INT4 GPU kernel was executed in this migration. The scaled
references identify their mathematical scope and precision. CUDA event intervals
for Python multi-kernel references can include host-dispatch gaps.

P7/P8 now require NCCL on two physical GPUs, with commands for two connected Sparks
or a two-GPU rental. That path was not measured on the available single-GPU host.
Old two-process Gloo results do not validate the revised NCCL measurement path.
Keep this milestone unmeasured until the second GPU session.

Run the suite in the selected environment with:
```bash
python -m unittest discover -s tests -v
```

An `OK` result with skips only validates the executed scopes. Use the
[setup guide](SETUP.md) for the Spark environment and two-host launches.
