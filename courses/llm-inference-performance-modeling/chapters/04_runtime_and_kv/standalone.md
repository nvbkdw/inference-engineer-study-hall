# Standalone lab: Scheduling under a bounded page budget

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

This script is an **untimed model exercise**, not CPU performance measurement.
Run the [Spark GEMM/calibration experiment](../02_performance_model/code/lab.ipynb)
to establish a measured GPU starting point. Use the full tutorial for actual
Spark serving/speculation timing; keep invented service costs labeled.

Read the request state diagram and logical/physical page distinction in [background.md](background.md). The supplied allocator and simulator run without a model or previous engine.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

Predict how chunks 16,64,256 change decode blocking, iteration overhead, peak populated pages, and goodput. Use the explicit invented service equation and frozen 20 ms TTFT/5 ms maximum-gap SLOs; do not interpret them as real hardware targets.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check the mechanism and run

```bash
OMP_NUM_THREADS=1 python chapters/04_runtime_and_kv/code/lab.py
python chapters/04_runtime_and_kv/code/experiment.py --out results/p04-first --repeats 3
```

Trace `simulate`: admission reserves worst-case future page capacity to avoid deadlock, while `Pool.append` populates only scheduled tokens. `select_work` prioritizes one token per decoding request. Every policy replays the same request arrivals within a repeat.

Use a different `--out` directory for every rerun. Nonempty destinations are
rejected so earlier predictions and observations remain reviewable.

## 3. Inspect concrete outputs

At `--repeats 3`, expect **9 rows** in `results.csv`, plus `manifest.json`,
`prediction.json`, and `summary.json`. Chapter-specific artifacts are `traces.json`, `requests.csv`, `goodput.svg`.
Every SVG figure also has a PNG copy. Points are medians; whiskers show observed
minimum and maximum, not a confidence interval.

Nine policy/trace rows must each complete 90 requests. The simulator asserts all page ownership is released. `summary.json` reports nine trace replays with three repeats. No field is a GPU measurement; the CSV time/rate columns are prefixed `simulated_`.

## 4. Explain and perturb

For each repeat compare chunk policies pairwise on the identical trace. Count requests satisfying both SLOs using `requests.csv` and divide by the whole observation duration in seconds. Reproduce one goodput row. Explain why peak populated pages differs from conservative reserved future capacity.

Change only the quadratic prefill-service coefficient from .00003 to 0. Save the changed source and use a new output directory. Predict whether the smaller-chunk advantage survives. Next reduce pool pages while keeping each individual request feasible; check bounded waiting and complete release.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

A simulated speedup depends on the service equation. Removing its superlinear prefill term can alter the preferred chunk size. Goodput excludes completed requests that violate either SLO, while the denominator includes the trace drain.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

Replace the invented service model with measured prefill/decode curves, then implement the full P4 scheduler, cancellation, exact-prefix reuse, and efficient paged adapter in [tutorial.md](tutorial.md).

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. If CUDA preflight fails, check that this Python environment has a Spark-compatible CUDA PyTorch build. The experiment deliberately does not fall back to CPU timing.

