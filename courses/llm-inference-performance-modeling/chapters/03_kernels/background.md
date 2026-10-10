# Chapter 3 background: from performance gaps to kernel optimization

GPU optimization makes data movement, parallel work, and resource use explicit.
A kernel can be mathematically correct but slow because it exposes too little
parallelism, rereads memory, spills registers, or waits at synchronization points.

## From Chapter 2's model to an optimization hypothesis

[Chapter 2](../02_performance_model/background.md) derives arithmetic intensity
and roofline bounds, then compares predictions with measured Qwen3 latency.
For one operation, let `F` be its FLOPs, `Q` its modeled minimum bytes, `C` the
assumed compute ceiling in FLOP/s, and `beta` the bandwidth ceiling in byte/s.
Then `AI = F/Q` in FLOP/byte and the ideal latency bound in seconds is
`t_bound = max(F/C, Q/beta)`: the operation must both perform the arithmetic
and move the data, even if those costs overlap perfectly.

Keep three quantities separate: `t_bound`, a calibrated estimate `t_est` using
measured shape-specific rates and overheads, and observed latency `t_obs`.
The signed residual `t_obs - t_est` tests the calibrated model; the distance
`t_obs - t_bound` suggests potential headroom under the bound's assumptions.
Neither gap is automatically recoverable. Sequential kernels, launch latency,
cache copies, redundant reads, and small workloads can prevent attaining an
aggregate roofline. An observation below the supposed bound requires checking
counting assumptions, cache residency, ceilings, and timing boundaries first.

Profiler evidence connects the gap to a mechanism:

| Evidence at a fixed workload | Candidate mechanism | Kernel hypothesis |
|---|---|---|
| Traffic exceeds the compulsory-byte model | Materialized scores or repeated KV reads | Tiling, online softmax, or GQA reuse can reduce transfers |
| Small-M GEMMs fall well below large-M calibration | Wasted tile rows or too little parallel work | A different tile shape can improve useful work per launch |
| Long-context decode exposes too few active programs | Too few request/head work groups | Context splitting may help if its combine cost is small enough |
| Many short launches and intermediate tensors dominate | Dispatch and intermediate traffic | Fusion may help; reducing FLOPs alone may not |

Use a separate trace to distinguish these hypotheses, and keep ordinary timing
samples free of profiler overhead. These comparisons use the actual Qwen3-8B
and Qwen3-32B shapes from Chapter 2, including explicit query width `Hq*R` and
MLP intermediate width `I`; do not substitute hidden width `D` for query width.

As an **illustrative calculation, not a measurement**, suppose a model step
takes 10 ms, of which attention takes 3 ms. Its attention bound is 1 ms, but
a proposed tiled kernel is predicted to take 1.5 ms. Keeping other work fixed
predicts `10 - 3 + 1.5 = 8.5 ms`, or a speedup of `10/8.5 = 1.176`.
If the adapter adds 0.5 ms, the prediction becomes 9 ms, or `1.111` times
faster. Even eliminating attention entirely leaves 7 ms. This connects the
local gap to the model-level benefit before implementation.

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

## Three views of the same forward call

Start with [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html):
operator names, input shapes, allocations, and CPU/CUDA activity connect model
code to kernels. Self time excludes nested operators; inclusive parent times
cannot be added to child times. Shape and allocation collection introduce
instrumentation overhead. An allocation is not a measurement of DRAM traffic.
Direct CuTe driver launches may lack a PyTorch external correlation ID. Identify
the known custom kernels by their generated names and retain unknown kernels as
unattributed; missing operator correlation does not mean zero GPU work.

Then use [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)
to follow the CPU launch thread, CUDA stream, copies, synchronization, and idle
intervals. A short kernel separated by long launch gaps suggests a different
optimization from a long uninterrupted kernel. CUDA events around a forward
include device idle gaps between launches; they do not sum kernel busy time.
A CUDA profiler capture range excludes checkpoint loading and JIT warmup.

Finally use [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
on selected kernels. DRAM bytes, compute activity, occupancy and register use
can test a traffic/resource hypothesis. Counter collection may replay kernels
and materially perturb execution. `ERR_NVGPUCTRPERM` means counter evidence is
unavailable until an administrator enables access; preserve the diagnostic.
Neither profiler latency nor a modeled byte count substitutes for that evidence.
Run the three tools separately and measure ordinary latency without any of them.

## Fusion from tensor shapes and rounding boundaries

Let `N=B*T` be rows, `D` hidden width, `I` intermediate width, `R=128` head
coordinates, `Hq` query heads, `Hkv` KV heads, and `b=2` bytes per BF16 element.
Reduction arithmetic and attention accumulation use FP32 (four bytes).
The [pinned 8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json)
has `(D,I,Hq,Hkv)=(4096,12288,32,8)`; the
[pinned 32B config](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json)
has `(5120,25600,64,8)`. In 32B, `Hq*R=8192`, not `D`.

For RMSNorm, a row computes `a=sum(x_i*x_i)/D`, then
`y_i=w_i*x_i/sqrt(a+eps)`. The reduction needs each input once; the normalized
output needs the input again unless it stays in registers. Fusion eliminates
global FP32 square and normalized intermediates and separate reduction/scaling
launches. The teaching implementation reads X twice (reduction and output), W
once, and writes Y once: a logical `4*N*D*b` access count before caching. The
ideal retained-X implementation would use `3*N*D*b`; neither is measured traffic.
At `B=4,T=2048,D=5120`, one BF16 intermediate occupies 80 MiB and one FP32
intermediate 160 MiB. Avoiding a write followed by a read saves twice its size.
A second read of X may be cheaper than retaining 160 values per lane and spilling.

For SwiGLU, `z=SiLU(g)*u`, where `g,u` have shape `[N,I]` and
`SiLU(g)=g/(1+exp(-g))`. Separate activation and multiplication read/write
`g,a,a,u,z`, about `5*N*I*b` logical bytes. A fused kernel reads `g,u` and writes
`z`, about `3*N*I*b`, removing `2*N*I*b` and one launch. At `B=4,T=2048,I=25600`,
that removed BF16 intermediate write/read is 800 MiB per layer. This excludes
GEMMs, allocator effects and caches, so it is a hypothesis to test with profiles.

For Q/K head RMSNorm + RoPE, reduction is over R separately for each head.
Each rotary pair consists of coordinates `d` and `d+R/2`. Frequencies are
`theta^(-2*d/R)` for `0<=d<R/2`; the angle is absolute position times frequency.
For prefix P and T new tokens, positions are `P..P+T-1`; after append KV length
is `P+T`. Adjacent-pair rotation or restarting positions at zero is incorrect.
One combined launch handles `N*(Hq+Hkv)` rows with separate shared head weights.
The normalized Q/K values and their rotary products remain in registers.

The baseline rounds normalized values to BF16 **before** weight multiplication,
rounds SiLU **before** multiplying the up projection, and rounds rotary sine,
cosine and both products **before** their sum. The supplied fusions retain these
boundaries. With this toolchain, casts alone allowed a rotary BF16 multiply/add
to contract into FMA. The `round_bf16` conversion helper explicitly rounds each
product before addition; a bitwise regression with exact RMS sums guards this
boundary. Q/K head normalization uses four consecutive squares per lane and a
descending shuffle reduction to match the reference's 128-coordinate mean.
The `scaled_score` helper also preserves the FP32 rounding of `QK/sqrt(128)`
before subtracting the softmax maximum; contracting these into FMA changes that
boundary. Other FP32 reduction orders and transcendental approximations can still
differ. Fused attention changes softmax accumulation order and rounds only its
final output to BF16; there is no stored probability matrix. Freeze tolerances
before measurements, check full-vocabulary logits, and report maximum errors.

## CuTe layout and execution model

A layout maps logical coordinates to an element offset using shape and stride.
The host passes strided tensors through DLPack; last-coordinate stride is one.
A JIT launcher specializes compile-time widths and tile parameters; dynamic
shapes and strides permit reuse for growing cache lengths. A compiled function
must run on PyTorch's **current CUDA stream**, including a nondefault stream.
Compilation belongs before capture and before measured repeats. CuTe retains zero
strides as compile-time constants even in a dynamic layout: a broadcast position
tensor has a different launch argument layout from a non-broadcast tensor. Include
zero-stride patterns in the JIT cache key and test switching batch sizes in both
compilation orders; tensor rank/dtype alone is insufficient.

In the supplied reduction, thread `lane+32*warp` owns coordinates
`lane+32*i` of row `4*block+warp`. A five-stage butterfly shuffle sums 32 lane
partials. Every participating warp has a uniform valid-row predicate. A partial
block must not make only some lanes skip a shuffle. Attention assigns each warp
one query and holds four Q and four output coordinates per lane. Four warps give
a query tile of four rows; key tiles stream 32 rows, masking the final tile.
Only key positions through `P+t` are visited. GQA selects `kv_head=h//(Hq/Hkv)`;
there is no expanded KV tensor or score matrix in global memory.

The supplied attention stores normalized output `o=u/ell`. For each key tile it
first finds the tile maximum, then sums exponentials with a descending warp tree,
then accumulates weighted V. With `m'=max(m,m_tile)`, `a=exp(m-m')`, and
`ell'=a*ell+sum_tile exp(score-m')`, update
`o'=(a*ell/ell')*o + sum_tile [exp(score-m')/ell']*v`.
This is the same online-softmax merge derived above. It limits output rescaling
to tile boundaries and preserves FP32 probability normalization before the V
multiply. Three passes recompute QK instead of keeping a score fragment that
could spill to local memory. This deliberate teaching tradeoff increases QK
arithmetic and K reads; a register/shared-memory tile is a later optimization
experiment. All accumulation remains FP32.

Each score uses 128 multiply-adds: 256 FLOPs for QK. Weighting and accumulating
V uses another 256 FLOPs. For one layer, useful causal attention work is
`4*B*Hq*R*(P*T + T*(T+1)/2)`. Exponentials and normalization are omitted from
this Chapter 2 matmul ledger. The baseline actually computes a dense rectangle
before masking, while this kernel computes QK three times per visited key;
the plots use the **same useful causal work** for both methods, not executed FLOPs.
At `B=4,Hq=64,T=S=2048`, a single FP32 score matrix is 4 GiB per layer; score
and probability storage can coexist. The fused implementation avoids both, but
rereads KV per query and uses SIMT instructions. Storage reduction alone does
not imply fast prefill or tensor-core utilization.

No shared-memory communication is used in this first implementation, so no CTA
barrier is needed. If an exercise stages keys for reuse across warps, all
threads must finish writing before reads, and finish reading before overwriting:
place `cute.arch.sync_threads()` at both boundaries. Predication must not make
barriers divergent. Larger query/key tiles trade reuse against register/shared
memory pressure; verify this on hardware instead of equating occupancy with speed.

## Hardware scope and downstream contract

The verified toolchain is CUTLASS DSL **4.2.1**, with its matching
[v4.2.1 elementwise example](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/examples/python/CuTeDSL/ampere/elementwise_add.py),
on GB10 capability **12.1**, CUDA 13.0.2, driver 580.126.09. The course kernels
use supported SIMT instructions. The [CuTe quick start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
explains why release and example versions must agree; its current release may
require a newer toolkit than this verified combination. The roofline retains
273 GB/s from [DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
and labels 125 dense BF16 TFLOP/s as a **teaching assumption**, not an official
BF16 specification or calibrated sustained throughput.

`OptimizedQwen3` inherits Chapter 1's forward and checkpoint parameter layout.
It adds independently selectable operator methods. `decode=True` selects final
logits; a one-token input selects decode attention, independently of that flag.
KV is caller-owned `[B,Hkv,P,R]`, equal-length, unpadded, on the parameter device
and dtype; appending returns new storage. GEMMs and dynamic concatenation stay
on the original backend. Ragged lengths, paging and in-place ownership belong
to Chapter 4. Unsupported optimized configurations raise in strict mode or
produce counted, explicit reference fallbacks in exploratory mode. Import the
module directly; a previous notebook's live Python state is never required.
