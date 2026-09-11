# Standalone lab: Attention storage versus execution cost

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

Read online-softmax m/ell/u updates in [background.md](background.md). This experiment needs basic PyTorch; CuTe/GPU prerequisites apply to the later full tutorial.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

For B=1,Hq=8,FP32 scores, derive dense temporary bytes `32*S*S` and tiled bytes `32*S*min(S,tile)`. Predict why a loop over tiles may reduce that temporary and still increase CUDA workload latency.

The optional `lab.py` command is an untimed CPU correctness oracle. All times from `experiment.py` use CUDA events on the GPU; no CPU fallback is provided.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check correctness, then measure on Spark

```bash
OMP_NUM_THREADS=1 python chapters/04_kernels/code/lab.py
python chapters/04_kernels/code/experiment.py --out results/p04-first --repeats 3
```

Compare `dense_attention` and `online_attention`. Both use the same q/k/v fixture and mask. The tiled implementation repeats KV heads and makes many PyTorch calls; it is a numerical reference, not a fused kernel.

The CUDA helper warms the workload ten times and synchronizes its end event. Python reference paths can include host-dispatch gaps, so these are workload intervals, not isolated fused-kernel times. The manifest records the actual GPU and CUDA build.

Use a different `--out` directory for every rerun. Nonempty destinations are
rejected so earlier predictions and observations remain reviewable.

## 3. Inspect concrete outputs

At `--repeats 3`, expect **36 rows** in `results.csv`, plus `manifest.json`,
`prediction.json`, and `summary.json`. Chapter-specific artifacts are `score_storage.csv`, `latency.svg`, `score_storage.svg`.
Every SVG figure also has a PNG copy. Points are medians; whiskers show observed
minimum and maximum, not a confidence interval.

At R=128 and S=127,128,129,512, twelve method/length combinations pass `rtol=1e-4,atol=1e-5`, each with three timing rows. `score_storage.csv` reports only the modeled score temporary. Other tensors, GQA repetition, peak allocator memory, and GPU traffic are excluded.

## 4. Explain and perturb

At S=512, calculate the dense-to-tile32 temporary ratio and the measured CUDA workload latency ratio. Explain why they need not agree. Read `lab.py`'s extreme-score and partial-tile checks before attempting the split-context extension.

Add tile size 64 and S=257. Predict its temporary storage and ensure the last partial tile is masked. Implement the two-group m/ell/u merge and compare it against unsplit attention; do not average normalized outputs.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

At S=512, tile32 reduces the modeled score temporary by 16x. This says nothing by itself about end-to-end speed; extra dispatch/reductions and other allocations can dominate.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

Follow [tutorial.md](tutorial.md) to implement a real CuTe kernel, validate BF16/ragged cases, study one GEMM change, and measure adapter-inclusive and model-level performance.

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. If CUDA preflight fails, check that this Python environment has a Spark-compatible CUDA PyTorch build. The experiment deliberately does not fall back to CPU timing.

