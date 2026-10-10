# Background: local work and communication

Tensor parallelism partitions a layer across devices. Replication places complete
independent models on devices. Both can increase serving capacity, but only TP can
reduce per-rank storage for a single model/request. Compare decisions at a fixed
hardware budget and matched workload.

## Derive the sharding axes

PyTorch stores a linear weight W as `[N,K]`, with `Y=X W.T`. Splitting W over
output features N gives independent local output slices: column parallelism in
the mathematical `[K,N]` layout. Splitting over input features K produces partial
outputs that must be summed: row parallelism in that layout. Always state whether
axis names refer to stored weights or mathematical multiplication.

For SwiGLU, shard gate/up `[I,D]` along I, producing local `[M,I/n]` activations.
Apply SiLU and multiplication locally. Shard down `[D,I]` along its input I;
each rank returns a partial `[M,D]`, followed by an all-reduce sum. The
[Megatron-LM construction](https://arxiv.org/abs/1909.08053) is the primary derivation.

For attention, shard Q/K/V over complete heads. Local query heads must align with
the corresponding KV-head groups. Compute attention locally, shard O over its
input `Hq*R`, and sum residual-width partial outputs. Replicate block norms,
embedding, and vocabulary head initially. The sample implements the SwiGLU
identity; full attention/cache/model distribution is the student extension.

## Memory is not divided uniformly

At TP=2, each rank owns half the attention and MLP matrices, four of eight KV heads,
and half the cache. Replicated vocabulary matrices/norms remain on every rank.
Per-rank storage is `sharded_bytes/2 + replicated_bytes + local_KV + workspace`.
Total fleet bytes can increase because of replication and collective buffers.
KV-head sharding cannot keep dividing unchanged beyond Hkv without another scheme.

## Collective costs

For n ranks, m-byte all-reduce payload, per-phase latency alpha, and effective
payload bandwidth BW, a ring starting model is:

```text
t_AR(m,n) ≈ 2(n-1) alpha + 2(n-1)/n * m/BW
```

NCCL can choose different algorithms; fit the actual latency curve in the relevant
size range. A decode residual at B=1 for BF16 32B is `5120*2=10240` bytes. The simple
construction performs two reductions per layer, or 128 per forward step for 32B
and 72 for 8B. Small-message latency can dominate even on a fast large-message link.

Build TP latency from local kernel times plus exposed communication. Do not assume
an asynchronous API implies overlap: dependent work may still wait for completion.
Use measured topology and the trace to identify the critical path.

## Compare useful systems

Scaling efficiency for matched single-request work is `t1/(n*tn)`. It is different
from throughput scaling and memory capacity. Two TP=1 replicas may deliver more
goodput than TP=2 when each model fits and requests are independent. TP can still
be necessary when one request/model cannot fit on a single GPU. Treat that as a
capacity benefit; an OOM replica is not a measured latency point.

Under low load, a request typically uses one replica; two GPUs do not halve its
latency. Under load, routing affects queue lengths and balance. Freeze routing
policy, total offered load, SLOs, and useful token accounting for the comparison.
