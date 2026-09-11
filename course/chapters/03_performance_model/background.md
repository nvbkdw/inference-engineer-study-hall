# Background: from tensor shapes to a latency model

Performance modeling begins with conservation: the device must perform some
arithmetic and move some data. Hardware throughput turns those quantities into
times. A model is useful when its assumptions are explicit and its residuals
suggest the next measurement.

## Compute and traffic

For `[M,K] @ [K,N]`, count `F = 2 M K N` FLOPs. If each input is read once and the
output written once, logical bytes are `Q = b(MK + KN + MN)`. Repeated cache-line
fetches, spills, transposes, and intermediate buffers can add physical traffic.
Arithmetic intensity `AI=F/Q` has units FLOP/byte. The ratio of effective compute
C to bandwidth BW, `C/BW`, is a crossover intensity. The
[roofline reading](https://jax-ml.github.io/scaling-book/roofline/) develops this framework.

For small M and large K,N, weight bytes dominate; with BF16, AI approaches M.
Growing the decode batch amortizes each weight read across more tokens. Eventually
compute, activations, KV, capacity, or latency limits change the behavior. Batch
size is a model input, not a universal tuning knob.

For sequential component j, begin with:

```text
t_j = max(F_j / C_eff,j, Q_j / BW_eff,j) + launch_j
t_step = sum(t_j) + CPU/synchronization costs on the critical path
```

Do not take one global maximum over attention and MLP: they run sequentially.
Within a fused kernel compute and transfers may overlap; across asynchronous
streams, count only dependencies on the critical path. A trace determines whether
CPU dispatch, GPU work, and copies overlap.

## Attention depends on pairs

For causal prefill of one sequence with S tokens, the number of allowed query-key
pairs is `S(S+1)/2`. Each pair uses 2R FLOPs for QK and 2R for AV. Across layers
and heads this gives `2 L Hq R S(S+1)`, excluding normalization. One decode query
attending S positions costs `4 L Hq R S` FLOPs. For ragged batches, sum over requests.
Padding can make executed arithmetic larger than this logical count.

GQA reduces unique KV bytes to `2 L Hkv R b sum(Si)`; it does not replace Hq with
Hkv in attention arithmetic. If a kernel rereads the same KV for multiple query
heads, the physical memory cost exceeds this unique-byte idealization.

For prefill with an existing prefix p and t new tokens, causal pairs are
`p*t + t*(t+1)/2`. This formula connects P2 chunk scheduling to service time.

## Which weights are actually touched?

The input embedding accesses selected rows. The vocabulary head often reads the
full output matrix. At generation-only prefill, an engine may apply the head only
to each request's last position. Teacher-forced scoring needs more positions.
Record the behavior instead of charging `V*D` to every prompt position by default.
The [inference chapter of the Scaling Book](https://jax-ml.github.io/scaling-book/inference/)
is the companion derivation; adapt its assumptions to the actual Qwen execution path.

## Calibration and falsification

Effective rates depend on shape, dtype, layout, and backend. A large GEMM's peak
TFLOP/s does not predict a matrix-vector operation. Repeated small buffers may sit
in cache and overestimate streaming bandwidth. Conversely, a roofline that omits
launch latency can underpredict a tiny operation even at perfect bandwidth.

Fit a small set of measured component rates, then freeze them. Reserve unseen
batch/context combinations and selected 32B cases. Relative error for one point
is `abs(predicted - measured)/measured`; plot signed residuals too. A systematic
error with context suggests KV/attention modeling; one with batch suggests shape
efficiency; an almost constant error suggests launches or CPU overhead. These are
hypotheses to test, not automatic diagnoses.

Keep service-time prediction separate from queueing. A low-load forward estimate
does not establish TTFT under overload. Feed measured service curves into a simple
replay only after the component model explains isolated execution.
