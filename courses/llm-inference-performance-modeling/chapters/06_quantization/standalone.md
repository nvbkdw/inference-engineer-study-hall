# Standalone lab: Packing, activation error, and reconstruction overhead

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

Read scale/group definitions and `delta_Y=X E.T` in [background.md](background.md). You can run this layer experiment without a model or previous chapters.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

Compute bytes for a 1024x4096 weight matrix: INT4 codes plus one FP32 scale per row/group. Predict how changing group size affects metadata and local dynamic range. Explain why emphasizing an activation channel changes output error without changing quantized weights.

The optional `lab.py` command is an untimed CPU correctness oracle. All times from `experiment.py` use CUDA events on the GPU; no CPU fallback is provided.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check correctness, then measure on Spark

```bash
OMP_NUM_THREADS=1 python chapters/06_quantization/code/lab.py
python chapters/06_quantization/code/experiment.py --out results/p06-first --repeats 3
```

`experiment.py` uses a random weight matrix with a large first-column outlier and two activation conditions. It checks pack/unpack, reconstructs FP32 weights, and times ordinary FP32 CUDA GEMM with and without reconstruction. No packed INT4 GEMM executes.

The CUDA helper warms the workload ten times and synchronizes its end event. Python reference paths can include host-dispatch gaps, so these are workload intervals, not isolated fused-kernel times. The manifest records the actual GPU and CUDA build.

Use a different `--out` directory for every rerun. Nonempty destinations are
rejected so earlier predictions and observations remain reviewable.

## 3. Inspect concrete outputs

At `--repeats 3`, expect **30 rows** in `results.csv`, plus `manifest.json`,
`prediction.json`, and `summary.json`. Chapter-specific artifacts are `timing.csv`, `error.svg`.
Every SVG figure also has a PNG copy. Points are medians; whiskers show observed
minimum and maximum, not a confidence interval.

Three random-weight fixtures times two activation conditions times five groups produce 30 error/storage rows. Every packing check passes. Relative output errors and timing depend on fixtures/hardware; there is no model-quality pass based on these random layers.

## 4. Explain and perturb

For group size 32, compute packed-plus-scale bytes and compare with the row. Compare error across activation conditions for the same fixture/group, then compare median reconstruction-plus-GEMM time to baseline using timing.csv. Do not call the reconstructed path W4A16 throughput.

Change only the activation multiplier from 20 to 1 while preserving seeded weights. Predict which error differences disappear. Add a clipping experiment and explain why the simple half-scale rounding bound no longer applies to clipped outliers.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

For G=32, codes use 2,097,152 bytes and FP32 scales use 524,288 bytes, for 2,621,440 total before other metadata. The original FP32 matrix uses 16,777,216 bytes. Lower representation size does not imply that reconstructing it before each GEMM is faster.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

Complete [tutorial.md](tutorial.md) for disjoint real calibration/evaluation, actual supported packed execution, quality intervals, both model sizes, and quantization/speculation interactions.

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. If CUDA preflight fails, check that this Python environment has a Spark-compatible CUDA PyTorch build. The experiment deliberately does not fall back to CPU timing.

