# Tutorial: transfer a request, then compare complete systems

## 1. Freeze the state contract (3 hours)

Use the topology characterized in P7. Start one prefill and one decode worker,
TP=1 each, with identical model revision, precision, and KV layout. Write the header
schema and pending-token convention before implementing the transport. Derive
logical payload and temporary duplication for W1/W2/W3 and both models.

Run the local state-copy oracle:

```bash
python chapters/08_prefill_decode/code/lab.py
```

The sample validates a tiny state, clones independent storage, and compares a
deterministic attention continuation. Its random K/V and generated Q are not Qwen
states or a full-model validation. Inspect the processed/history/pending invariant.

## 2. Exercise synchronous GPU transport (3 hours)

Follow [standalone.md](standalone.md) to launch one NCCL rank on each of two
connected Sparks, or use two physical rental GPUs. First substitute `code/lab.py`
for `code/experiment.py` to validate payload installation and acknowledgment;
then measure the payload sweep with `code/experiment.py`.

The fixture pre-agrees identity/layout/shape. Real requests need a versioned header
before allocation. Add a header mismatch case and reject it before installation.
Keep CPU/Gloo as an optional untimed protocol check. Two CPU ranks or two ranks
sharing the sole Spark GPU cannot establish this GPU transfer result.

## 3. Move actual model state (7 hours)

Connect P1/P4 prefill output to export/import functions. Export populated K/V,
absolute positions, history including the pending token, model/format identity,
and the declared sampler state. For paged storage, pack logical populated blocks
in a known order, allocate destination blocks, and rebuild its private page table.

Implement `header -> reserve -> ready -> payload -> install -> acknowledge`.
Keep source state alive until transfer completion and ownership acknowledgment.
If reservation fails, retain source ownership and apply bounded waiting/rejection.
Log request ID and epoch at each transition. Add cancellation during reservation,
during transfer, and immediately after installation; terminal transitions release
each allocation once. Make duplicate acknowledgments harmless.

Compare colocated and transferred greedy continuation using the same model path
at lengths 15/16/17,256,2048 and a feasible long context. Validate 8B broadly and
32B at selected points. Add EOS as the first token and output-budget exhaustion.
Stochastic validation must specify how sampler state is preserved or compare the
proper distributions; same seeds alone are not sufficient.

**Checkpoint:** complete Qwen continuation agrees, destination storage is independent,
and cancellation cannot expose stale state or leak source/destination blocks.

## 4. Calibrate transfer separately (5 hours)

Sweep actual KV payload sizes across W1/W2/W3. Measure pack, staging/copy, transport,
install/coordination, and total handoff, first in isolation, then under serving load.
Warm up transport setup where appropriate; separately report cold registration.
Record bytes transferred, effective payload GB/s, topology, and host staging.

Fit setup and bandwidth terms and compare predictions at a withheld prompt length.
Save at least three independent repeats. Avoid labeling the tool's small fixed
fixture timing as representative multi-GiB transfer bandwidth. Compute required
bytes/s at each offered arrival rate to expose a saturated transfer stage.

## 5. Compare the same two GPUs (6 hours)

Run two colocated TP=1 replicas, one P plus one D worker, and the best feasible
colocated TP arrangement from P7. Match models, precision, prompt traces, SLOs,
total GPUs, and useful output. Use W1–W4 plus the same fixed W1/W3 mixture used in
P4. Start at low load and increase identical offered rates for each arrangement.

Log queue times at both stages, stage utilization, transfer bytes, TTFT, first-to-
second-token gap, within-request p95 ITL, completion latency, failures, and goodput.
Include a short-output/long-prefill case and a long-output/short-prefill case to
expose stage imbalance. Plot goodput against offered load and handoff overhead
against payload. Explain which interference disappears and which delay replaces it.

If available, repeat a narrow cross-node transfer case with real hosts and the
P7 network calibration. Otherwise mark it unmeasured. The same-node project can
still support a scoped two-GPU design decision.

## 6. Study a production implementation (2 hours)

Trace the pinned Dynamo handoff design and map your state transitions onto it.
Optionally deploy one matching recipe only after your experiment is complete;
record actual transport/layout configuration. Deployment success is a code-path
comparison, not a substitute for the fixed-resource benchmark.

## 7. Defend the final design (6 hours)

Write a recommendation for each model under one declared workload and hardware
budget. Include selected precision and quality interval, batching/page policy,
whether speculation is enabled and k, TP versus replicas, and whether P/D is
justified. Cite your own measured memory, latency, goodput, and transfer evidence;
label projections. Name the workload change most likely to invalidate the design.

Submit the state schema, failure/cancellation checks, real continuation artifacts,
transfer model and held-out result, fixed-resource system comparison, and final
memo. Use a 30-minute defense: trace one request from arrival through release,
estimate its costs, explain one failed optimization, then redesign for a changed mix.

**Defense:** Which observation would convince you to turn P/D off even if TTFT
improved? When is the source allowed to reuse its KV pages?
