# Validation of the DGX Spark measurement path

Historical project IDs below are preserved with their original validation records: P2 runtime is now Chapter 4, P3 performance modeling is now Chapter 2, and P4 kernels is now Chapter 3. Current commands and navigation use the new chapter numbers.

The course defaults to DGX Spark GPU measurements. The newest integrated validation
is dated 2026-09-18; earlier checks below are historical.
CPU fixtures remain optional correctness/modeling aids; historical CPU timings
are not the baseline for the revised labs.

## September 18: Chapter 3 real-model kernels, profiles and matched sweeps

Chapter 3 now has three runnable notebooks and importable CuTe kernels, an
optimized subclass of Chapter 1's model, and profiling/measurement drivers.
Chapter 2 and Chapter 3 share the same estimators, timing, aggregation and plots.
The tested environment was GB10 SM121, driver 580.126.09, CUDA toolkit 13.0.2,
Python 3.13.9, PyTorch 2.14.0+cu130, CUTLASS DSL 4.2.1, and cuda-python 13.4.1.
The matching official elementwise example passed; its source, output and version
record are in `results/p03-toolchain`.

The final regression suite ran **38 tests: 37 passed and one two-GPU test skipped**.
This includes ten optimized-model/kernel tests covering both model widths,
partial attention tiles, nonzero prefixes, GQA, extreme scores, supported strides,
current-stream execution, fallback rules, cache ownership, and BF16 rounding.
Chapter 1's affected notebook passed from a fresh kernel. Chapter 2's analytical
and GPU calibration paths passed from fresh kernels; its full real-model sweep
was exercised through the shared driver by Chapter 3 Lab 3. Chapter 3 Lab 1
executed from a fresh kernel replaying its completed live baseline artifacts;
Labs 2 and 3 executed live from fresh kernels. Lab 3’s final reporting additions
were also checked by replaying the saved observations from a fresh kernel; its
live execution is preserved as `lab3.live.executed.ipynb`. Executed notebooks and
test logs are retained under `results/p03-validation`. A trace-attribution test
covers custom driver launches without PyTorch operator correlation IDs.

Both pinned real checkpoints passed full-vocabulary baseline/optimized and
cached/full-prefix checks on the documented seven-token fixture, with batches
1 and 4. The acceptance thresholds stayed `atol=0.25, rtol=0.02` throughout;
operator thresholds stayed `atol=0.02, rtol=0.02`. These are combined elementwise
bounds, so a maximum absolute error above 0.25 can pass for a sufficiently large
reference logit. This establishes the stated fixtures and operator cases, not
language quality or a general numerical error bound for every prompt.

Integration checks found and fixed a broadcast-position JIT argument-layout bug,
compiler contraction across BF16 rotary product rounding, Q/K mean reduction
order, and contraction across FP32 attention score scaling. Failed candidates
remain marked invalid and are excluded from the final performance comparison.
The final attention implementation uses normalized online output and three key
tile passes; its extra QK work is explicitly excluded from the plots' *useful*
FLOP ledger and discussed in the background.

Final evidence directories:

- Baseline profiles: `results/p03-baseline-initial` — four timed cases and eight
  PyTorch traces/operator tables, plus two Nsight Systems reports and summaries.
- Optimized/ablation profiles: `results/p03-optimized-20260918-002326-7d9af3` — eight timed cases,
  sixteen PyTorch traces with the expected custom kernels, and six Nsight Systems
  reports and summaries.
- Matched sweep: `results/p03-sweep-20260918-010508-b32074` — all 36 cases completed, with 972 raw
  observations, frozen predictions, matched token IDs, source/revision hashes,
  flags, compilation records, plots, 36 matched speedups and 44 Amdahl comparisons
including the profiled ablations. No unexpected fallback or compilation during
  measured repeats was accepted. All workloads completed without memory failures.

Reporting refinements were checked against the saved observations. Original
sources matching the run manifests are retained in `source-at-start`; final
analysis/rendering hashes are in `profile_comparison_scope.json` and
`plot_sources.json`. The sweep saves its complete main-process compilation
inventory. The profiling study's supplementary main-kernel inventory export
failed; its aggregate inventory explicitly identifies the six capture processes
that supplied the complete records. Per-case main-process records remain in
`status.json`. No compilation times were invented for the missing export.

The sweep uses two complete warmups, three repeats and eight decode calls.
At prompt length 2048, the following synchronized wall-time medians include greedy
selection; decode is the mean call latency across the full eight-call window.
Full repeat ranges are retained in `speedups.csv`. Tradeoff overlays separately
use only the first decode call, matching the frozen prefix.

| Model | Batch | Prefill baseline → optimized (ms) | Speedup | Decode baseline → optimized (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| 8B | 1 | 1778.6 → 1904.2 | 0.934× | 128.6 → 149.7 | 0.859× |
| 8B | 4 | 6949.5 → 7378.9 | 0.942× | 230.6 → 181.7 | 1.269× |
| 32B | 1 | 6649.4 → 7133.6 | 0.932× | 417.4 → 395.2 | 1.056× |
| 32B | 4 | 25527.3 → 27751.6 | 0.920× | 795.9 → 481.0 | 1.655× |

At batch 1 / context 2048, prefill kernel counts fell from 3,430 to 579 for 8B
and from 6,090 to 1,027 for 32B, while integrated prefill became slower.
Instrumented attention time rose from 981 to 1,508 ms and from 3,794 to 5,278 ms,
respectively. Fusion-only prefill achieved 1.124× / 1.086× speedups; attention-only
achieved 0.846× / 0.868×. This supports the explanation that this SIMT attention
kernel's repeated score work outweighs its storage savings at this context.
These profiler durations remain separate from the uninstrumented measurements.
The final comparison includes trace idle intervals and 210 selected operator
records with shapes and inclusive allocation bytes; allocation totals are not
DRAM traffic and cannot be summed across nested operators. Chapter 2's eleven
regression tests also passed after the Chapter 3 plot-label refinements.

These are local measurements, not a controlled published GB10 characterization.
The roofline uses the same compulsory-byte proxy for both implementations and
labels 125 dense BF16 TFLOP/s as an assumption. Modeled bytes are not counter traffic.
**Nsight Compute remains incomplete:** the actual attempt and permission preflight
both report `ERR_NVGPUCTRPERM`. Failure logs and per-target unmeasured statuses are
retained; administrator-enabled counters are still required. PyTorch/Nsight
Systems capture and ordinary timing completed independently of this limitation.

## September 12: consolidated notebooks and full-model custom execution

Chapter 1's custom operators and mapping now have a single source in `lab.ipynb`;
the duplicate Python model file was removed. Cache checks and timing receive the
notebook's class and mapping explicitly. The CLI and full-model notebook import
only tagged definition cells, skipping lab actions. All 12 Lab 1 code cells passed,
including the original FP32 logit gate and 30-row timing sweep in
`results/p01-notebook-4584385b`. The same notebook definitions passed header audits
for tiny (25 tensors), 8B (399), and 32B (707), without loading the full model weights
again. The regression suite ran 18 tests: 17 passed and the two-GPU NCCL test was
skipped. This includes checking that definition import does not execute lab actions.

The subsequent utility consolidation moved saved-logit acceptance and the CUDA
cache sweep into `notebook_utils.py`. Lab 1 reran successfully using direct
`compare_checkpoint_logits()` and `run_cache_experiment()` calls; its accepted
report is now `comparison.json`. The new timing artifacts are in
`results/p01-notebook-116d8d29`, with 30 rows and all five cache checks passing.
The regression comparison also rejects a changed non-top logit even when argmax
is unchanged. Notebook schemas, source compilation, and local links passed.
The full regression suite ran 17 tests: 16 passed and the two-physical-GPU NCCL
test was skipped on this single-GPU machine.

Chapter 1 now has one theory/reading page, `background.md`, and two practical
notebooks. All 11 code cells in `lab.ipynb` and all five code cells in `lab2.ipynb`
executed successfully on GB10. Both notebook schemas and local documentation links
were checked. JupyterLab and ipykernel are declared course dependencies.

Lab 1 retained its FP32 reference/custom acceptance result (maximum absolute logit
error `7.152557373046875e-06`) and produced 30 timing rows, five exact cache-byte
checks, and both embedded figures. Timing artifacts are in
`results/p01-notebook-15e6c7ca`. These are execution/artifact checks; background
checkpoint downloads make this run unsuitable as a controlled timing baseline.
The checkpoint loader/comparison regression test passed, including identity and
shape rejection. The token utility's reuse and changed-identity rejection were
also checked.

Lab 2 downloaded the complete snapshots at these pinned revisions and ran the
shared custom decoder in BF16, releasing the model/cache between sizes:

| Model | Revision | Audited tensors | Parameters | Saved logits |
|---|---|---:|---:|---|
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 399 | 8,190,735,360 | `[5, 151936]`, all finite |
| Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` | 707 | 32,762,123,264 | `[5, 151936]`, all finite |

Snapshots are under `models/qwen3-8b/<revision>` and `models/qwen3-32b/<revision>`.
The run directories are `results/p1-full-8b/20260912T222537Z-12da81e1` and
`results/p1-full-32b/20260912T222537Z-be3fb651`. Each preserves model metadata,
tensor inventory, token provenance, memory predictions, CPU logits, notebook source,
and implementation hashes. Saved notebook-source hashes were verified against the
manifests. Both models processed the prompt and four forced continuation tokens.

This establishes full-checkpoint loading and custom forward execution on the
short fixture. The full-model Transformers comparisons were **not executed**;
BF16 oracle equivalence, mixed-length batching, long-context behavior, model quality,
and serving performance remain separate validation tasks in the notebooks.

## September 12: real weights in the two-layer practice model

Chapter 1 now uses unchanged tensors from `Qwen/Qwen3-8B` at commit
`b968826d9c46dd6066d109eabc6255188de91218`: layers 0–1, embeddings, final norm,
and vocabulary head. Source shards are in `models/qwen3-8b-source`; the extracted
checkpoint and tokenizer are in `models/qwen3-tiny`. All 25 selected tensors
were compared exactly with their source tensors, and output-shard SHA-256 hashes
were verified. The model has 1,630,556,672 parameters; the selected source shards
occupy about 7.9 GiB and the extracted BF16 checkpoint about 3.1 GiB.

The notebook ran on GB10 using Python 3.13.9, PyTorch 2.14.0+cu130, and Transformers
5.17.0. Its FP32 reference/custom comparison passed across five positions and
151,936 vocabulary entries, with maximum absolute error `7.152557373046875e-06`
and relative L2 error `3.321475787743111e-07`. The elementwise acceptance thresholds
remain `rtol=1e-4, atol=1e-5`. Both paths execute the same two-layer truncation;
these results do not establish full 8B/32B behavior or output quality.

The nine mathematical tests and the CUDA checkpoint loader/comparison test passed.
The checkpoint test also checks identity rejection and a deliberately incorrect
weight shape. P1 GPU tests now require prepared real weights and explicitly skip
when those resources are absent.

### Why cache equivalence uses a separate numerical criterion

The old random-fixture elementwise threshold flagged a few near-zero logits when
real weights were evaluated using full-sequence versus single-token GEMMs. An
independent precision check promoted the mathematical reference's operations to
FP64 while keeping exactly the same stored BF16 weights. For held calibration
inputs of 11 and 129 tokens, FP64 full/cached outputs agreed within `2.94e-14`.
FP32 relative L2 error against that FP64 result ranged from `3.65e-7` to `9.56e-7`;
maximum error divided by reference maximum magnitude stayed below `1.25e-6`.
The calibration is saved as `results/qwen3-tiny-fp64-calibration.json`.

Before the subsequent timing sweep, the separate FP32 cache gate was set to
per-row relative L2 ≤ 2e-6 (allowing two roughly 1e-6 reference errors) and maximum
error / reference maximum magnitude ≤ 5e-6. Unlike an absolute floor near zero,
these criteria test each entire vocabulary vector and its worst component at
its measured scale. The checkpoint comparison's elementwise gate was unchanged.
An optional CPU checkpoint comparison exceeded that elementwise floor in seven
near-zero entries; its tolerance has not been established as a CPU acceptance
baseline. The required real-weight acceptance baseline is Spark CUDA.

The subsequent S=128–2048 CUDA sweep passed all five cache-byte checks and both
normwise gates, producing 30 timing rows in `results/p01-real-two-layer-validated`.
Maximum relative L2 was `1.844e-6`, maximum scaled error `1.851e-6`, and maximum
absolute error `3.100e-5`. Logical FP32 KV grows by 16,384 bytes/token. This run
checks correctness and artifact generation; concurrent validation activity means
its timings are not a controlled hardware characterization. No general error
bound or validity outside these workloads follows from this calibration.

## September 11 migration: hardware and environment used

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
| P1 (original migration run) | Earlier larger random fixture, now retired; cached/recomputed next logits agree; five cache-byte checks at S=128–2048; 30 timing rows |
| P3 | BF16 K=N=4096 GEMMs; 128 MiB copy calibration; predictions saved before six held-out shapes; 24 prediction/measurement rows |
| P4 | FP32 CUDA attention with R=128; dense/tiled outputs agree for 12 cases; 36 timing rows and score-storage accounting |
| P6 | FP32 CUDA 1024x4096 random layers; nibble roundtrip, output errors, storage, and reconstruction/GEMM timing; 30 error rows |

P2 and P5 model exercises also ran. They retain invented service/cycle costs and
perform **no hardware timing**. Their output is explicitly simulation/numerical
modeling, not GPU or CPU performance measurement.

Raw migration-run artifacts are under `results/spark-migration/` in the authoring
workspace. They are ignored experiment outputs; regenerate them using the chapter
commands when copying the course elsewhere.

### Historical random tiny configuration (retired September 12)

After removing the larger timing fixture, P1 was rerun on the same GB10 using
the single `Config()` shared by mathematical checks, checkpoint generation,
and timing: `(L,D,I,Hq,Hkv,R,V)=(2,48,96,8,2,8,101)` in FP32.
Artifacts are in `results/p01-unified-tiny/`: 30 timing rows, five exact cache-byte
matches, a 256-byte/token cache slope, and maximum absolute cached/recomputed
logit error of `2.384185791015625e-07`. The existing nine mathematical tests and
the offline sharded-checkpoint comparison test also passed in the CPU correctness
environment. The earlier larger-fixture results remain historical evidence;
neither random-fixture run describes the current real-weight Chapter 1 model.

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
