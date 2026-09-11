# Standalone lab: Calibrating the cost of reductions

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

Read stored-weight sharding axes and the all-reduce model in [background.md](background.md). The measured sample uses two connected DGX Sparks, one GPU per host, or two physical GPUs on a rental host. No earlier engine implementation is required.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

Derive the two-rank ring starting model `2*alpha + bytes/BW`. Predict why small payloads need a latency term. The code fits a nonnegative affine model at 4 KiB,64 KiB,1 MiB, then freezes 16 KiB,256 KiB,4 MiB predictions.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check the mechanism and run

```bash
OMP_NUM_THREADS=1 python chapters/07_tensor_parallelism/code/lab.py
# The lab.py command above is an optional untimed correctness oracle.
```

`shared/communication.py` preallocates FP32 payloads and warms up three times. Each rank starts from rank+1; the sum must be 3. Timed wall duration includes the collective and its completion, while barriers, resets, correctness checks, and the max-rank timing reduction are outside timing.

For measurement, complete the two-Spark networking/NCCL setup in [SETUP.md](../../shared/SETUP.md).
Replace `<spark-0-ip>` with the reachable rendezvous address before execution.

Spark 0:
```bash
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=<spark-0-ip> --master_port=29500 chapters/07_tensor_parallelism/code/experiment.py --backend nccl --out results/p07-first --repeats 3
```

Spark 1:
```bash
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=<spark-0-ip> --master_port=29500 chapters/07_tensor_parallelism/code/experiment.py --backend nccl --out results/p07-first --repeats 3
```

On a host with two physical GPUs, replace the two-host launcher flags with
`--standalone --nproc_per_node=2`. This does not work as TP=2 hardware on a
single-GPU Spark. With only one Spark, retain the distributed result as unmeasured.

Use a different `--out` directory for every rerun. Nonempty destinations are
rejected so earlier predictions and observations remain reviewable.

## 3. Inspect concrete outputs

At `--repeats 3`, expect **12 rows** in `results.csv`, plus `manifest.json`,
`prediction.json`, and `summary.json`. Chapter-specific artifacts are `calibration.csv`, `transfer.svg`.
Every SVG figure also has a PNG copy. Points are medians; whiskers show observed
minimum and maximum, not a confidence interval.

Three held-out sizes yield one predicted and three measured rows each. Every payload must match. Both ranks run real NCCL collectives on distinct CUDA devices. The manifest records each GPU/host; this is communication measurement, not full Qwen TP.

## 4. Explain and perturb

Recompute the relative error at each withheld size and inspect the residual shape. Convert fitted slope from ms/byte into an effective payload GB/s only when slope>0. An NCCL all-reduce curve may not follow one transport algorithm across all sizes.

Add a payload near 10 KiB, the BF16 32B batch-1 residual size, as a new held-out condition. Predict the contribution of 128 such reductions per model step and label it a component projection that omits overlap and other work.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

The two-rank ring has two phases in the latency term and one payload/BW term. A measured large-transfer bandwidth alone cannot explain 128 small reductions. A zero fitted slope means the tiny calibration cannot resolve bandwidth, not infinite physical bandwidth.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

The optional untimed MLP `lab.py` proves the sharding identity. Continue with [tutorial.md](tutorial.md) for actual Qwen attention/cache sharding, NCCL topology, full-model latency, and equal-budget replicas.

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. For measured distributed labs, use two physical GPUs with NCCL as described in setup. A single Spark cannot provide two independent GPU ranks. CPU/Gloo is optional correctness-only work.

