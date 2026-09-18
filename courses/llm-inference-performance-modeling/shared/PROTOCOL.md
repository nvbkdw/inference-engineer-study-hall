# Experimental protocol

Every lab asks: **What do I predict? What did I measure? What change should remove
the bottleneck?** A negative result with a correct explanation can earn full credit.

## Fixed workloads

| ID | Prompt tokens | Output tokens | Primary question |
|---|---:|---:|---|
| W1 | 256 | 128 | Low-load overhead |
| W2 | 2048 | 256 | Main comparison |
| W3 | 8192 | 128 | Long prefill interference |
| W4 | 256 | 1024 | Sustained decode |
| W5 | 2048 shared + 256 unique | 256 | Prefix reuse |

Set maximum total context to at least prompt plus output, e.g. 8448 for W3.
Begin at batch 1; admit 4 and 16 only after memory accounting. Sweep 8B broadly;
repeat decisive 32B points. Fix token IDs, generation policy, EOS handling, and
prefix-cache condition. Cold and warm prefix results belong in separate rows.

## Three clocks

Kernel time uses CUDA events around warmed work on the relevant stream, ending
with an event synchronization. Model time includes the declared forward/generation
boundary. Serving time uses monotonic client timestamps including queueing.
Kernel timing does not measure network or admission delay.

For request arrival `a`, emitted token times `e[0:N]`, completion `c`:

```text
TTFT = e[0] - a
ITL[j] = e[j] - e[j-1], j >= 1
completion latency = c - a
TPOT = (e[-1] - e[0]) / (N - 1), N >= 2
```

For N=1, ITL/TPOT are undefined; qualify the request using TTFT and an explicitly
declared completion condition. Count speculative bursts at actual emission times.
Also report first-to-second-token gap and within-request p95 gap. Do not average
request percentiles and label the result a pooled percentile.

Freeze numeric SLOs after baseline, before tuning. Goodput = successful completed
requests that meet the chosen per-request SLOs / observation seconds. Include
failures and unfinished requests in the accounting, alongside output tokens/s.
For a finite arrival trace, use first arrival through last terminal event as the
observation interval and report the drain duration. A fixed-window experiment
instead must declare boundary/censoring handling; never silently drop stragglers.

Closed-loop concurrency measures saturation but can hide queue growth. Open-loop
arrivals are scheduled independently of completions. Freeze baseline capacity,
then generate three seeded traces at 25%, 50%, 75%, and 90% of that capacity; replay
the same actual arrivals across implementations. Start with about 200 requests
per trace. Label small-sample p99 estimates exploratory.

## Predict and measure

1. Copy [manifest.json](manifest.json) into `results/<experiment>/`. Replace nulls
   with actual metadata; null means unknown, not zero. Save a prose hypothesis.
2. Derive FLOPs, logical bytes, memory capacity, and the expected limiting term.
3. Run the chapter's correctness gates before timing. Record tolerances and oracle.
4. Warm up until compilation/graph capture finishes; exclude warmup from samples.
5. Collect at least three independent uninstrumented repeats. Save raw rows.
6. Profile a separate representative run. Explain gaps and critical-path costs.
7. Change one factor; repeat the matched comparison and compute uncertainty.
8. Publish both predictions and observations with residuals and a decision.

Use decimal GB/s and TFLOP/s, bytes for storage, and explicit ms/s column suffixes.
Fused multiply-add counts as two FLOPs. Logical minimum traffic is an analytical
quantity; hardware traffic requires counters and may include redundant reads.
Do not combine bandwidth calibrated on one GPU with latency measured on another.

For paired comparisons, resample whole requests/examples (or entire traces when
dependence matters), keeping baseline/intervention pairs together. Report a 95%
bootstrap interval, seed, and sample size. Distinguish variability between runs
from variability between requests. Reserve held-out conditions before calibration.

Suggested CSV schemas:

```text
components: run_id,phase,op,m,k,n,flops,logical_bytes,predicted_ms,measured_ms
requests: run_id,request_id,arrival_s,first_token_s,last_token_s,completion_s,output_tokens,status
tokens: run_id,request_id,token_index,emitted_s
memory: run_id,weights_bytes,live_kv_bytes,allocated_kv_bytes,peak_bytes,reserved_bytes
```

Serving definitions and profiling workflow companions:
[SGLang benchmarking](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)
and [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
