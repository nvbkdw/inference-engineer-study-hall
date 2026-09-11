# Standalone lab: Cache correctness and memory slope

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

Read the block computation and offset causal mask in [background.md](background.md). You need tensor shapes, matrix multiplication, and next-token logits; no previous project code is required.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

For the Qwen3 tiny FP32 CUDA cache, derive `2*L*Hkv*R*4*S`. Predict the byte slope and explain why a cached one-token call can avoid most prefix computation without making its attention cost constant.

The optional `lab.py` command is an untimed CPU correctness oracle. All times from `experiment.py` use CUDA events on the GPU; no CPU fallback is provided.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check correctness, then measure on Spark

```bash
OMP_NUM_THREADS=1 python chapters/01_reconstruct_qwen3/code/lab.py
python chapters/01_reconstruct_qwen3/code/experiment.py --out results/p01-first --repeats 3
```

Inspect `Config.kv_bytes`, `TinyQwen.forward`, and the `cached`/`full` functions in `code/experiment.py`. Both correctness and timing use the same default `Config()`: L=2,D=48,I=96,Hq=8,Hkv=2,R=8,V=101. Timing runs in FP32 on CUDA; the offline checkpoint generator also reads this configuration. Each method produces the same next-position logits; both project only the last position to vocabulary. Cache preparation and token creation are outside timing; cache concatenation remains inside the cached forward.

The CUDA helper warms the workload ten times and synchronizes its end event. Python reference paths can include host-dispatch gaps, so these are workload intervals, not isolated fused-kernel times. The manifest records the actual GPU and CUDA build; `prediction.json` records the model dimensions. At this tiny scale, dispatch overhead can dominate, so explain the observed curve without assuming it predicts 8B/32B performance.

Use a different `--out` directory for every rerun. Nonempty destinations are
rejected so earlier predictions and observations remain reviewable.

## 3. Inspect concrete outputs

At `--repeats 3`, expect **30 rows** in `results.csv`, plus `manifest.json`,
`prediction.json`, and `summary.json`. Chapter-specific artifacts are `memory.csv`, `memory.svg`, `latency.svg`.
Every SVG figure also has a PNG copy. Points are medians; whiskers show observed
minimum and maximum, not a confidence interval.

`summary.json` must report five exact cache-byte matches at S=128,256,512,1024,2048 and `correctness: passed`. Absolute logit error is recorded; the actual assertion uses `rtol=1e-4, atol=1e-5`. Latency depends on your Spark GPU, so there is no fixed expected millisecond result.

## 4. Explain and perturb

Use `memory.csv` to compute `(bytes_at_2048 - bytes_at_128)/(2048-128)` and compare it with the saved prediction. From `results.csv`, compute the median recomputed/cached time ratio at every length. Explain the difference between equal logical work at the output boundary and different work performed internally.

Keep the model configuration fixed. Add S=768 to the context-length sweep and rerun to a new directory. Predict its cache bytes before running: `768*256 = 196608`. Then change only the chunk partition in `lab.py` and verify unchanged logits.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

The slope is 256 bytes/token: `2*2*2*8*4`. Doubling processed context length doubles this storage exactly. The attention cache does not store the query heads, and allocator reservation is not part of this logical count.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

Use the same predicted-versus-counted table for the pinned 8B/32B configs, then complete [tutorial.md](tutorial.md) for checkpoint mapping, real logits, mixed batches, and device-memory measurements.

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. If CUDA preflight fails, check that this Python environment has a Spark-compatible CUDA PyTorch build. The experiment deliberately does not fall back to CPU timing.

