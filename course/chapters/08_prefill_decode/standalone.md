# Standalone lab: The measured price of moving a payload

Allow 1–2 hours for reading, execution, and analysis. This is the entry experiment
for this chapter; the full two-week project is in [tutorial.md](tutorial.md).

## 1. Prepare and predict

Read the pending-token handoff diagram and serialization model in [background.md](background.md). The measured experiment needs two connected DGX Sparks or two physical rental GPUs and no previous engine. A single Spark can run only the optional untimed state oracle.

From the `course/` directory, complete the DGX Spark environment checks in
[setup](../../shared/SETUP.md), activate your Spark-compatible CUDA environment, and confirm that PyTorch and
Matplotlib import. Keep the whole course checkout so the shared helpers resolve.

Derive a byte/BW transfer term plus setup. Predict why timing through acknowledgment exceeds an ideal wire-time calculation. As in P7, fit three payload sizes and freeze three different sizes before measuring them.

Write your prediction in a short note before running. The script also saves its
assumptions and numerical predictions before the corresponding measurements.

## 2. Check the mechanism and run

```bash
OMP_NUM_THREADS=1 python chapters/08_prefill_decode/code/lab.py
# The lab.py command above is an optional untimed correctness oracle.
```

`shared/communication.py` times rank 0 from send through receipt of rank 1's completion acknowledgment. Rank 1 acknowledges only after the payload is ready. Identity/layout are pre-agreed and buffers preallocated in this transport exercise; allocation, packing, queueing, and model work are excluded.

For measurement, complete the two-Spark networking/NCCL setup in [SETUP.md](../../shared/SETUP.md).
Replace `<spark-0-ip>` with the reachable rendezvous address before execution.

Spark 0:
```bash
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=<spark-0-ip> --master_port=29500 chapters/08_prefill_decode/code/experiment.py --backend nccl --out results/p08-first --repeats 3
```

Spark 1:
```bash
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=<spark-0-ip> --master_port=29500 chapters/08_prefill_decode/code/experiment.py --backend nccl --out results/p08-first --repeats 3
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

The 3 withheld sizes produce 12 rows with three repeats. All transmitted elements equal 1, and the existing `lab.py` separately verifies independent destination storage and attention continuation. These two tests establish different parts of the mechanism.

## 4. Explain and perturb

Recompute prediction error and calculate observed payload GB/s from each median total handoff time. Explain why this is acknowledgment-inclusive throughput rather than a direct measurement of link capacity. Identify the largest omitted term when applying the curve to real paged KV.

Add a fixed receiver coordination delay immediately before acknowledgment and predict an intercept increase. Rerun to a new directory; treat the delay as injected synthetic work, not a property of the original transport. Next inspect how packed-page conversion would change the timed boundary.

Keep the original result and document changed code/input separately. The manifest
records source hashes, software, seed where relevant, and measurement scope.

## 5. Check your reasoning

An acknowledgment proves the destination has completed the agreed transfer boundary; source state can then be released. It does not automatically prove correct model identity, cancellation safety, or equivalent full-model continuation. Those require the full protocol's validation.

Submit the result directory and a 400–600 word entry-lab note containing your
original prediction, one calculation reproduced from CSV, a figure interpretation,
and one changed-condition result. This entry note is preparation for the full
project's 1,200–2,000 word research memo.

## 6. Continue to the full project

Use [tutorial.md](tutorial.md) to transfer actual Qwen state, handle cancellation/ownership, measure multi-GiB payloads, and compare two replicas against 1P/1D and feasible TP under equal resources.

Use [assessment.md](assessment.md) to distinguish completed entry goals from
remaining full-project goals. A passing reference exercise does not establish real-model
quality, GPU kernel behavior, or serving performance.

## Troubleshooting

If `torch` or `matplotlib` is missing, activate the environment used for setup and
install there. If an output directory exists, choose a new name rather than deleting
previous evidence. If an assertion fails, inspect the numerical/ownership condition
before timing again; never widen tolerance simply to pass. For measured distributed labs, use two physical GPUs with NCCL as described in setup. A single Spark cannot provide two independent GPU ranks. CPU/Gloo is optional correctness-only work.

