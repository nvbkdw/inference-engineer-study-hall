# Standalone lab: Freeze predictions before measuring

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

Read FLOPs, bytes, and sequential rooflines in [background.md](background.md). The standalone experiment uses CUDA BF16 GEMMs and does not require P1/P2 engine code.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

Derive BF16 GEMM FLOPs `2*M*4096*4096` and ideal bytes `2*(M*4096+4096*4096+M*4096)`. Predict which cost changes fastest with M. The program calibrates only M=1,16,512 before freezing six unseen M values.

The optional `lab.py` command is an untimed analytical calculator with invented hardware inputs. All times from `experiment.py` use CUDA events on the GPU; no CPU fallback is provided.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check correctness, then measure on Spark

```bash
python chapters/03_performance_model/code/lab.py --model 8b --phase decode --tflops 50 --gbps 200
python chapters/03_performance_model/code/experiment.py --out results/p03-first --repeats 3
```

The first command is an illustrative Qwen calculator: 50 TFLOP/s and 200 GB/s are invented. The second command performs the actual Spark CUDA calibration/measurement described here.

Inspect calibration of the small-call overhead, copy rate, and effective compute rate. `prediction.json` is written before timing M=2,4,8,32,64,128. The copy buffer is 128 MiB; the resulting rate is specific to this local experiment, not guaranteed Spark LPDDR bandwidth.

The CUDA helper warms the workload ten times and synchronizes its end event. Python reference paths can include host-dispatch gaps, so these are workload intervals, not isolated fused-kernel times. The manifest records the actual GPU and CUDA build.

Use a different `--out` directory for every rerun. Nonempty destinations are
rejected so earlier predictions and observations remain reviewable.

## 3. Inspect concrete outputs

At `--repeats 3`, expect **24 rows** in `results.csv`, plus `manifest.json`,
`prediction.json`, and `summary.json`. Chapter-specific artifacts are `calibration.csv`, `prediction.svg`.
Every SVG figure also has a PNG copy. Points are medians; whiskers show observed
minimum and maximum, not a confidence interval.

There are six held-out conditions, one prediction plus three timing rows each. `summary.json` contains six relative errors and their median. The 25% target is diagnostic; exceeding it does not imply that the code is incorrect.

## 4. Explain and perturb

Recompute `abs(predicted-median(measured))/median(measured)` for two points and the median over all six. Identify whether residuals grow with M or are largest for small M. Keep the original predictions; do not fit them retrospectively and label the result held out.

Add M=12 and 48 as two new held-out conditions after diagnosing the first result. Change one justified modeling assumption, freeze new predictions, and evaluate only on the new conditions. Explain the calibration cost and avoid claiming universal hardware rates.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

A single effective compute rate can miss shape-dependent GEMM behavior. The small-call floor and dispatch gaps matter at small M, and local caches can make the copy rate a poor proxy for weight traffic. A good diagnostic identifies which measurement distinguishes these explanations.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

Use the same chronology with model-derived operators and the full six-point 8B/32B held-out matrix in [tutorial.md](tutorial.md). The GPU benchmark sample supplies CUDA-event timing; model latency includes more than GEMM.

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. If CUDA preflight fails, check that this Python environment has a Spark-compatible CUDA PyTorch build. The experiment deliberately does not fall back to CPU timing.

