# Standalone lab: Exact sampling and draft break-even

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

This script is an **untimed model exercise**, not CPU performance measurement.
Run the [Spark GEMM/calibration experiment](../03_performance_model/standalone.md)
to establish a measured GPU starting point. Use the full tutorial for actual
Spark serving/speculation timing; keep invented service costs labeled.

Read conditional probabilities, rejection residuals, and pending-token indexing in [background.md](background.md). No checkpoint, tokenizer, or prior engine is required for the finite-vocabulary exercise.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

For the exact draft, predict k+1 useful tokens per cycle with no stopping boundary. For uniform and misaligned drafts, predict which larger k values could lose under the stated invented cost model. These are sampling/cost predictions, not Qwen forecasts.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check the mechanism and run

```bash
OMP_NUM_THREADS=1 python chapters/05_speculative_decoding/code/lab.py
python chapters/05_speculative_decoding/code/experiment.py --out results/p05-first --repeats 3
```

`target` changes with token history. Each `cycle` evaluates probabilities at the accepted prefix. Inspect how first-token counts test the target distribution and how accepted-prefix histograms determine empirical useful tokens.

Use a different `--out` directory for every rerun. Nonempty destinations are
rejected so earlier predictions and observations remain reviewable.

## 3. Inspect concrete outputs

At `--repeats 3`, expect **36 rows** in `results.csv`, plus `manifest.json`,
`prediction.json`, and `summary.json`. Chapter-specific artifacts are `accepted_prefix.csv`, `break_even.svg`.
Every SVG figure also has a PNG copy. Points are medians; whiskers show observed
minimum and maximum, not a confidence interval.

Three draft conditions times four k values times three repeats produce 36 rows. Every condition runs 2000 cycles. First-token deviations must stay below the declared family-wise Hoeffding bound; `lab.py` separately checks exact probability mass and stopping boundaries.

## 4. Explain and perturb

Reconstruct one empirical mean emitted count from its histogram as `sum((accepted+1)*count)/2000`. Insert it into the saved cost equation and reproduce one modeled speedup. Identify a condition with speedup below one and explain the rejected draft work.

Double the draft cost coefficient while keeping probability functions and sampled outcomes fixed. Replot from the stored data without rerunning sampling. Then add a history-dependent target fixture of your own and repeat the distribution check.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

Without EOS/length truncation, each cycle emits accepted-prefix length plus one correction/bonus token. Exact-draft output is k+1; increasing k still has a finite cost. Distribution equality is different from identical random-seed output strings.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

Use [tutorial.md](tutorial.md) for tokenizer compatibility, real target block verification, forced cache rollback, and real prose/code/extraction acceptance on 8B→32B.

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. If CUDA preflight fails, check that this Python environment has a Spark-compatible CUDA PyTorch build. The experiment deliberately does not fall back to CPU timing.

