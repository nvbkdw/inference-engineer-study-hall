# Tutorial: one sharded model or two replicas

## 1. Derive the local layer (3 hours)

Partition a small `[N,K]` matrix by output and input axes on paper. Show where a
concatenation is needed and where a sum is needed. Run:

```bash
python chapters/07_tensor_parallelism/code/lab.py
```

The sample computes each SwiGLU shard sequentially on CPU and compares their sum
with an unsharded calculation. It generates full weights for an independent oracle;
a production TP loader must avoid keeping all full matrices on every GPU.

## 2. Run a real GPU collective (3 hours)

Use two connected Sparks with one rank per host, or two physical rental GPUs.
Follow the exact two-host launches in [standalone.md](standalone.md), first
substituting `code/lab.py` for `code/experiment.py` to check the MLP all-reduce.
Then run the measured payload sweep. A single Spark has one GPU and cannot
supply this measurement with two local processes.

Use NCCL and select each rank's device with LOCAL_RANK. Optional CPU/Gloo runs
of `lab.py --backend gloo` check protocol logic only; do not use their elapsed
time as a performance baseline. Add a deliberate sharding-axis error and verify
that the unsharded oracle catches it. Cleanly destroy the process group.

## 3. Shard Qwen layers and caches (8 hours)

Implement column/row linear wrappers in your engine. Slice checkpoint tensors on
the loading path, asserting shard shapes before copying. Shard Q/K/V by complete
head groups and O by its attention-input dimension. Shard gate/up over intermediate
features and down over that same input axis. Keep norms and vocabulary matrices
replicated initially and charge their full storage to each rank.

At TP=2, assign four KV heads per rank for both models. Validate one attention
layer from the real 8B checkpoint with GQA, then full 8B prefill/incremental decode against
TP=1. Repeat selected 32B inputs. Include chunked-prefill and ragged cache lengths.
Investigate reduction-order differences with documented numerical criteria.

**Checkpoint:** single-layer and full-model logits pass; each rank owns the intended
cache heads; no rank silently retains complete sharded weights.

## 4. Measure the communication path (5 hours)

Capture GPU SKU, memory, driver, interconnect topology, and `nvidia-smi topo -m`
where available. Build/run the pinned [NCCL tests](https://github.com/NVIDIA/nccl-tests)
according to their README. Sweep payloads around actual residual sizes from small
decode through prefill; include approximately 8 KiB, 10 KiB, and larger token batches.

Fit alpha and effective BW in the relevant ranges; preserve raw timings and note
algorithm changes. A bandwidth-only fit to multi-megabyte messages will usually
miss small reduction cost. Instrument the engine's collective count and payloads;
explain deviations from two reductions per layer.

Use P3 local compute/memory terms to predict TP=2 latency before measuring full
distributed workloads. Include reductions on the actual critical path and
replicated operations. Save those predictions.

## 5. Compare equal GPU allocations (6 hours)

Measure one TP=1 instance as the scaling reference, then one TP=2 instance and
two TP=1 replicas on the same pair of GPUs. Select a declared routing policy,
such as round-robin admission, and record queueing per replica. Use identical
precision, arrival traces, cache conditions, SLOs, and model revisions.

Start W1/W2/W3 at low load, then increase the same actual offered rates for each
configuration. Sweep per-instance batch limits only as a declared experiment.
Plot low-load latency, maximum goodput meeting SLOs, and per-rank memory. Restrict
replica comparisons to feasible single-GPU configurations; separately list TP-only
capacity points. Compare predicted and measured TP latency and scaling efficiency.

## 6. Transfer the explanation across nodes (3 hours)

If the syllabus's cross-node session is available, run one small decode and one
prefill TP=2 case on two real hosts. Configure rendezvous, rank placement, and NIC
selection from the pinned PyTorch/NCCL documentation. Repeat communication
calibration on that path. Record transport, NICs, host count, and payload sizes.

If it is unavailable, include an explicitly modeled network sensitivity analysis
and mark the real experiment unmeasured. Two local processes do not establish
multi-node performance. Complete the same-node requirements in either case.

## 7. Submit (4 hours)

Include shard diagrams, loader/state checks, collective curves, pre-measurement
predictions, matched TP/replica results, and a placement decision for each model.
Explain which workload change could reverse it. Successful `torchrun` on the tiny
MLP is a communication milestone, not complete Qwen TP validation.

**Defense:** Why can TP reduce memory use and still worsen batch-1 latency? Which
terms stay replicated, and which communication appears twice per decoder layer?
