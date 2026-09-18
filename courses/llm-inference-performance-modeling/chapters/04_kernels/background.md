# Background: data reuse and stable tiled attention

GPU optimization makes data movement, parallel work, and resource use explicit.
A kernel can be mathematically correct but slow because it exposes too little
parallelism, rereads memory, spills registers, or waits at synchronization points.

## GEMM tiles

A tile computes `C[BM,BN]` while iterating over K in blocks BK. A stage loads
`BM*BK + BK*BN` input elements and performs `2*BM*BN*BK` FLOPs. Larger BM/BN
increases potential reuse, but accumulators consume registers and staged operands
consume shared memory. Increasing stage count can hide load latency while reducing
resident blocks. Occupancy is a resource constraint, not the objective itself.

For M=1, a tile designed for a large prefill can waste rows or expose too few work
groups. Use the actual Qwen shape and dtype before reasoning about tensor-core
utilization. Investigate one tile/layout/stage change, then stop and explain it.
The [CuTe programming model](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
is the implementation companion; instruction support depends on exact hardware.

## Online softmax from first principles

For scores x_j and vectors v_j, attention output is
`sum(exp(x_j)*v_j) / sum(exp(x_j))`. Subtracting a common maximum leaves the
ratio unchanged and prevents overflow. Store a triple for the processed keys:

```text
m = max_j x_j
ell = sum_j exp(x_j - m)
u = sum_j exp(x_j - m) v_j
output = u / ell
```

To merge two disjoint groups a,b, choose `m=max(m_a,m_b)`, then:

```text
ell = exp(m_a-m)*ell_a + exp(m_b-m)*ell_b
u   = exp(m_a-m)*u_a   + exp(m_b-m)*u_b
```

This supports processing keys in tiles without storing all scores, and combining
context splits without averaging already-normalized outputs. The two partial
groups generally have different probability masses, so their output means cannot
simply be averaged. This is the central numerical exercise behind
[FlashAttention](https://arxiv.org/abs/2205.14135).

Initialize m=-infinity, ell=0, u=0. An all-masked group is the neutral state;
avoid evaluating `exp(-inf - -inf)` when merging two empty groups. Decode lengths
are positive in the required contract. Prefill can contain future-only tiles;
skip those or apply the neutral-state rule. Mask ragged tails before maxima and sums.

## A bounded GQA decode kernel

Use Q `[B,Hq,128]`, KV `[B,Hkv,Smax,128]`, and a length per request. Map query
head h to KV head `h // (Hq/Hkv)`. Read only positions below the request's length.
Accumulate QK, ell, and u in FP32 with BF16 inputs/cache. Output one vector per
query head. No backward pass, dropout, RoPE, quantization, or page traversal in the
first custom kernel. Those functions live outside this kernel's contract.

One program per request/query head may underutilize a large GPU at B=1. Context
splitting creates more programs but adds partial-result writes and a combine
launch. Predict the extra traffic and parallelism before enabling it.
[FlashAttention-2](https://arxiv.org/abs/2307.08691) is the work-partitioning reading.

## Kernel speed versus model speed

If fraction f of baseline model time is replaced by a kernel s times faster,
the idealized integrated speedup is `1 / ((1-f)+f/s)`. For f=0.1,s=2, that is
about 1.053, assuming everything else is unchanged. Adapter copies, changed
layouts, graph behavior, and dispatch can invalidate that assumption. Measure
kernel-only, adapter-inclusive, and full-model paths separately.
