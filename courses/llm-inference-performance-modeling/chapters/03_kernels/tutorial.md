# Tutorial: connect one custom kernel to model performance

## 1. Select a measured target (2 hours)

Use the prediction–measurement comparison and separate profile from Chapter 2
to choose a Qwen GEMM shape and an attention workload. Keep the theoretical bound,
calibrated estimate, and observed latency distinct. Explain which evidence links
the gap to data movement, tile utilization, parallelism, or launch overhead using
[the gap analysis](background.md#from-chapter-2s-model-to-an-optimization-hypothesis).
Record the baseline operation's fraction of total model time and predict an
Amdahl ceiling even for an infinitely fast replacement. Freeze a prediction for
one specific change, including adapter costs, before measuring it.

Reuse the model exported by [Chapter 2's model module](../02_performance_model/code/model.py)
and its benchmark fixtures. Keep reusable kernel and adapter code under this
chapter's `code/` directory so Chapter 4 can import it from a fresh process.
The workflow is prediction → implementation → correctness → measurement → explanation.

## 2. Bound the toolchain and GEMM change (7 hours)

Use the [official quick start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
to pin CuTe, CUDA, and the exact architecture. Run a tiny supported example before
porting your existing GEMM. Record the example path, commit, launch command, and
compiler target in the manifest. A Hopper TMA/warp-group example requires the
corresponding GPU; do not assume it runs on Spark.

Benchmark M=1,4,16,128,512 with model-derived K,N against the backend used in P2.
Use its [GEMM timing sample](../02_performance_model/code/lab.ipynb) as the
event-timing pattern. Choose one change—BM/BN/BK, pipeline stages, or layout—and
predict register/shared-memory demand and which shapes benefit. Compare numerical
results, latency, and a few counters before/after. Stop after one explained change.

## 3. Derive online attention on CPU (4 hours)

```bash
python chapters/03_kernels/code/lab.py
```

`online_attention` computes tiled decode and causal prefill without a full S-by-S
score tensor. It uses ordinary PyTorch loops and repeats KV heads for clarity; it
is a mathematical oracle, not a CuTe kernel or a speed claim. Run tile sizes 1,17,32
and inspect the partial-tile cases. Derive the m/ell/u merge on paper, then extend
the oracle to compute two disjoint key ranges and merge their triples.

**Checkpoint:** split and unsplit results match a dense FP32 reference, including
extreme logits. Deliberately average normalized partial outputs and construct a
case that shows why that implementation is wrong.

## 4. Implement the CuTe decode path (8 hours)

Create `code/gqa_decode.py` using the pinned official example's host/JIT
structure. The implementation is a required student deliverable, not included in
the PyTorch oracle. Implement in this order:

1. A host adapter validates dtype, R=128, head divisibility, strides, positive
   lengths, and output shape. Unsupported shapes dispatch to the reference backend.
2. Assign one program to `(request,query_head)` and derive its KV-head index.
3. Load Q once. For each key tile, load K/V using the declared contiguous cache layout.
4. Predicate loads for `key_index < length`; treat invalid scores as -infinity.
5. Reduce QK across head coordinates in FP32 and apply the `1/sqrt(128)` scale.
6. Update m, ell, and u using the background equations; normalize and store once.
7. Test a single full tile, then multiple tiles, then a partial tail and ragged batch.

Begin with simple supported loads/reductions before asynchronous staging. Explicitly
document which lanes hold the accumulator coordinates and where barriers are
required. If profiling shows too few active programs, add context splitting and a
second combine kernel; handle empty splits as neutral triples. Keep the unsplit
kernel for comparison.

## 5. Validate the real kernel (3 hours)

Test S=127,128,129,2048,8192 at B=1 and 4, with unequal lengths within B=4 and
both model head configurations. Compare BF16 input kernel output to an FP32 oracle
at declared tolerances. Include large finite logits and every masked load/store.
Run the installed toolchain's memory/race checker where supported. A PyTorch oracle
passing does not validate CuTe indexing, synchronization, or memory safety.

**Troubleshooting:** errors only at S=129 suggest tail handling; errors only in
GQA suggest head mapping; errors growing with split count suggest combine scaling.

## 6. Integrate and explain (5 hours)

Begin the measurement workflow on DGX Spark using
[the standalone CUDA experiment](standalone.md). Its `code/experiment.py`
uses CUDA events and records the actual GPU identity. CPU `lab.py` checks
are optional correctness preparation; they supply no timing baseline.
The remaining steps below extend the reference workload to the full project.

Extend the imported Chapter 1/2 model with an attention backend adapter; keep its
weights, projections, RoPE, normalization, and cache semantics shared. The caller
owns the cache. For one decode token, pass Q `[B,Hq,128]`, K/V
`[B,Hkv,Smax,128]`, and valid lengths **after** appending the token (`S = p+1`
for prior length `p`). Return `[B,Hq,128]` on the same device and in the input
dtype without mutating K/V. Use the declared BF16 kernel path and retain the
reference for prefill or unsupported shapes. The existing model handles equal
lengths; test ragged batches directly at the kernel boundary.

Time layout conversions and dispatch separately and include them in integrated
latency. Compare matched masks, precision, head grouping, graph settings, and
cache condition with the reference and explicitly labeled library baselines.
Warm up every compiled shape. Chapter 4 will import this adapter, add page
ownership, and account for any gather needed by the contiguous kernel.

Report kernel speedup, adapter-inclusive speedup, and full-model speedup on W1/W2
and one long-context case. Predict using baseline f and measured s, then explain
any discrepancy with a trace. Report logical minimum traffic separately from
counter-measured traffic and duplicate KV reads.

## 7. Submit (3 hours)

Provide one GEMM modification, the custom CuTe attention source, CPU/GPU oracle
comparisons, an annotated kernel design, and Amdahl versus observed integration
results. Competitive library performance is not required; measured correctness
and an explained performance gap are. If hardware is unavailable, label the CuTe
and integration milestones incomplete rather than reporting CPU checks as GPU work.

**Defense:** Which tensor is reused, where does it live, and how many times is it
read? When would splitting context make a decode kernel slower?
