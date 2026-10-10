# Chapter 2 · Roofline modeling and transformer inference performance

A performance model estimates how fast an LLM can run on a given hardware platform and identifies what limits its performance. By analyzing the computation and data movement required by each operator, we can estimate execution time, locate bottlenecks, and prioritize optimizations.

This chapter develops the foundations of LLM inference performance modeling. We introduce the GPU roofline model, estimate transformer FLOPs and memory traffic, and examine how prefill and decoding place different demands on hardware. We then connect these estimates to key metrics—including token throughput, time to first token (TTFT), and time between tokens (TBT)—to understand how model architecture, workload, and hardware shape inference performance.

## 1. What a hardware roofline means

TBD: better and more intuitive explaination of GPU architecture and roofline model. [Yi Wang's blog](https://wangkuiyi.github.io/roofline.html) for reference.

Three terms describe arithmetic work and throughput:

- **MAC (multiply-accumulate operation)** computes $a + b \times c$ and is a way to count operations. Multiplying an $m\times n$ matrix by an $n$-element vector requires $mn$ MACs; multiplying an $m\times k$ matrix by a $k\times n$ matrix (GEMM) requires $mnk$ MACs.
- **FLOP (floating-point operation)** counts one floating-point multiplication or addition as one operation. One floating-point MAC counts as two FLOPs, even when executed as a fused multiply-add.
- **FLOPS (floating-point operations per second)** measures compute throughput, written as FLOP/s in this chapter to distinguish it from FLOPs, an operation count.

We use $\mathrm{AI}$ for arithmetic intensity and reserve $I$ for the MLP intermediate width, following [Chapter 1](../01_reconstruct_qwen3/background.md#dimension-symbols-and-concrete-configurations).

For a workload, define:

| Symbol | Meaning | Unit |
|---|---|---|
| $F$ | Floating-point operations required by the chosen counting convention | FLOP |
| $Q$ | Bytes transferred across memory boundary | byte |
| $C$ | Compute throughput ceiling for the actual precision and execution units | FLOP/s |
| $\beta$ | Bandwidth ceiling across that memory boundary | byte/s |
| $\mathrm{AI}=F/Q$ | Arithmetic intensity: work performed per byte transferred | FLOP/byte |
| $P=F/t$ | Achieved arithmetic throughput over elapsed time $t$ | FLOP/s |

Decimal TFLOP/s means $10^{12}$ FLOP/s, and GB/s means $10^9$ byte/s. GiB denotes $2^{30}$ bytes.

Even ideal execution needs enough time for arithmetic and memory transfer:

$$
t_{compute}=\frac{F}{C},\qquad t_{memory}=\frac{Q}{\beta},\qquad
 t\geq\max\left(\frac{F}{C},\frac{Q}{\beta}\right).
$$

Assuming compute and transfers can overlap perfectly gives the roofline:

$$
\boxed{P\leq\min(C,\beta \mathrm{AI})},\qquad \boxed{\mathrm{AI}_* = \frac{C}{\beta}}.
$$

On logarithmic axes, the bandwidth limit is a rising line and the compute limit is horizontal. Their intersection, the **ridge point**, is the intensity needed to supply arithmetic units at their peak rate. Below that intensity, memory bandwidth sets the lower ceiling; above it, compute throughput sets the lower ceiling. A workload below both ceilings can instead be limited by small shapes, dependencies, instruction mix, cache behavior, dispatch, or synchronization. Roofline introduces these limits; it does not guarantee that an implementation reaches them. See [NVIDIA's roofline explanation](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/) and the [Scaling Book derivation](https://jax-ml.github.io/scaling-book/roofline/).

For an **illustrative** device with $C=100$ TFLOP/s and $\beta=200$ GB/s, the ridge is 500 FLOP/byte. At $\mathrm{AI}=1$, the ceiling is 0.2 TFLOP/s; doubling compute does not change it. At $\mathrm{AI}=1000$, the ceiling is 100 TFLOP/s; doubling bandwidth does not change it. These are explanatory numbers, not measurements of a course GPU.

### Why use it?

Roofline provides a scale check for latency, a hypothesis about the limiting resource, and a way to predict which change might help. More batching can reuse weights; fusion can remove intermediate traffic and launches; quantization can reduce bytes; a faster arithmetic unit helps only work that can feed it. It also makes a failed prediction useful: the gap asks which omitted cost or incorrect assumption needs measuring.

Choose the memory boundary explicitly: device DRAM, unified LPDDR, L2, and shared memory have different bandwidths and intensities. **Logical tensor bytes are not measured DRAM bytes.** Cache reuse can reduce physical transfers; tiled rereads, spills, cache copying, and materialization can increase them. A point based on estimated bytes is an analytical placement on a chart, not a hardware-counter measurement.

Precision matters just as much. Dense BF16 inference needs a dense BF16 compute ceiling, with the appropriate accumulation mode. Sparse FP4 marketing throughput is not that ceiling. DGX Spark's specifications list 273 GB/s LPDDR bandwidth and an FP4 AI throughput figure; the notebook keeps its exploratory BF16 compute assumption visibly separate from those published specifications. [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)

### From one GEMM to a whole model

For $X_{M\times K}W_{K\times N}=Y_{M\times N}$, with $e$ bytes per element:

$$
F_{GEMM}=2MKN,\qquad Q_{logical}=e(MK+KN+MN).
$$

If the weight matrix dominates traffic, $\mathrm{AI}\approx 2M/e$. With BF16, $e=2$ and $\mathrm{AI}\approx M$. A single-token decode with batch $B$ has $M=B$, whereas prefill has $M=BS$. This explains why batching and prefill often use arithmetic units more efficiently. It is an approximation: KV traffic and activation traffic eventually matter too.

Transformer operators execute largely in sequence. For serial components $j$, a stronger latency model is

$$
\widehat t_{forward}=\sum_j\left[\max\left(\frac{F_j}{C_{eff,j}},\frac{Q_j}{\beta_{eff,j}}\right)+\ell_j\right]+t_{host,critical},
$$

where $C_{eff,j}$ and $\beta_{eff,j}$ depend on shape, dtype and backend, and $\ell_j$ models non-overlapped overhead. Only add host work on the critical path. A single $\max(\sum F/C,\sum Q/\beta)$ can be much more optimistic than the sum of per-operator maxima: a memory-limited operator and a compute-limited operator still run one after the other. A trace determines actual overlap and fusion boundaries.

## 2. The decoder architecture being modeled

The [decoder architecture reference](https://www.ryhuang.blog/blogs/llm-arch/) provides the general FLOPs accounting methods for transformer architecture which decomposes into embeddings, attention projections, attention products, MLP, normalization, and output projection. Here we derive the inference-specific ledger using Qwen3's actual dimensions, grouped-query attention (GQA), and cache behavior. The reference's generic $D=H_qR$ simplification and training/backward formulas must not be applied automatically to Qwen3 inference.

### Qwen3 model architecture

A dense Qwen3 forward follows this sequence:

```text
Token IDs → embedding lookup
  → repeat L times:
      RMSNorm → Q/K/V projections → per-head Q/K RMSNorm → RoPE on Q/K
      → append K/V to cache → causal GQA → output projection → residual add
      → RMSNorm → gate/up projections → SiLU(gate) × up → down projection → residual add
  → final RMSNorm → vocabulary projection → select next token
```

GQA shares K/V heads across groups of query heads. Each query head still forms its own scores and weighted value sum. Qwen3 also normalizes each Q/K head, uses three bias-free SwiGLU projections, and has separate input embeddings and output-head weights.

For a hidden-state matrix $X\in\mathbb R^{M\times D}$, attention projects it into $Q\in\mathbb R^{M\times D_q}$ and $K,V'\in\mathbb R^{M\times D_{kv}}$. After reshaping into heads, query head $h$ uses KV head $g(h)=\lfloor h/(H_q/H_{kv})\rfloor$:

$$
Z_h=\operatorname{softmax}\left(\frac{\widetilde Q_h\widetilde K_{g(h)}^\top}{\sqrt R}+\text{causal mask}\right)V'_{g(h)}.
$$

The tildes denote Q/K normalization and rotary position encoding. Concatenating the $Z_h$ produces width $D_q$, and $W_O$ projects back to the residual width $D$. The MLP then expands into width $I$ and contracts back:

$$
\operatorname{SwiGLU}(X)=\left[\operatorname{SiLU}(XW_g)\odot XW_u\right]W_d,
\qquad \operatorname{SiLU}(x)=\frac{x}{1+e^{-x}}.
$$

Qwen3's normalization uses

$$
\operatorname{RMSNorm}(x)=\gamma\odot x\left(\frac1n\sum_{i=1}^n x_i^2+\epsilon\right)^{-1/2}.
$$

Here $n=D$ for residual-stream normalization and $n=R$ for each Q/K head. Residual additions preserve the shape $[M,D]$ across the layer. Dropout is disabled during evaluation. These operations explain why a decoder has both large matrix multiplies and many smaller memory/dispatch-sensitive steps.

### Model dimensions

These are the original dense Qwen3 checkpoints, not a MoE or quantized variant. The pinned configs are the source of the values below: [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) and [Qwen3-32B config](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json).

| Dimension | Symbol | Qwen3-8B | Qwen3-32B |
|---|---|---:|---:|
| Decoder layers | $L$ | 36 | 64 |
| Hidden width | $D$ | 4,096 | 5,120 |
| MLP intermediate width | $I$ | 12,288 | 25,600 |
| Query heads | $H_q$ | 32 | 64 |
| KV heads | $H_{kv}$ | 8 | 8 |
| Head dimension | $R$ | 128 | 128 |
| Query/output-attention width | $D_q=H_qR$ | 4,096 | 8,192 |
| Key/value width | $D_{kv}=H_{kv}R$ | 1,024 | 1,024 |
| Query heads per KV head | $G=H_q/H_{kv}$ | 4 | 8 |
| MLP expansion ratio | $I/D$ | 3 | 5 |
| Vocabulary size | $V$ | 151,936 | 151,936 |
| Tied embedding/head | — | No | No |

**For 32B, $D_q\ne D$.** Its query projection is $[M,5120][5120,8192]$, and its output projection is $[M,8192][8192,5120]$. Substituting $D^2$ for both weight shapes undercounts their FLOPs.

## 3. Count FLOPs using tokens and attention pairs

Let $B$ be the number of equal-length requests, $T$ the new tokens processed per request in this forward, and $P$ the already-cached prefix length. Define

$$
M=BT,\qquad A=B\left(PT+\frac{T(T+1)}{2}\right).
$$

$A$ counts valid causal query-key pairs across the batch, before multiplying by query heads. The first new query sees $P+1$ keys, the next sees $P+2$, and so on.

| Phase | New tokens $T$ | Cached prefix $P$ | Valid pairs $A$ |
|---|---:|---:|---|
| Prefill of an $S$-token prompt | $S$ | $0$ | $BS(S+1)/2$ |
| One decode call after an $S$-token prefix | $1$ | $S$ | $B(S+1)$ |
| Chunked prefill | $T$ | $P$ | $B[PT+T(T+1)/2]$ |

Throughout this chapter, **prefix length excludes the current decode input**. Its K/V are appended during that call, so attention sees $P+1$ positions (prefix plut current token itself). This convention prevents a one-token discrepancy between the math, cache length and benchmark rows. For ragged requests, sum the pair counts over requests; a padded implementation may execute extra work.

### Operator-by-operator FLOP ledger

The numeric columns substitute model dimensions but retain $M$ and $A$ so the same table works for any batch, prefill, decode or chunk. All rows are **per decoder layer** unless marked otherwise. Let $U$ be the number of positions passed to the vocabulary head across the batch; generation-only prefill uses $U=B$, while scoring all new tokens uses $U=M$.

| Operator | Equation / FLOP count | Qwen3-8B FLOPs | Qwen3-32B FLOPs |
|---|---|---:|---:|
| Q projection | $Q=XW_Q$, $2MDD_q$ | $33{,}554{,}432M$ | $83{,}886{,}080M$ |
| K projection | $K=XW_K$, $2MDD_{kv}$ | $8{,}388{,}608M$ | $10{,}485{,}760M$ |
| V projection | $V'=XW_V$, $2MDD_{kv}$ | $8{,}388{,}608M$ | $10{,}485{,}760M$ |
| Attention scores | $QK^\top$, $2AD_q$ | $8{,}192A$ | $16{,}384A$ |
| Attention weighted values | $\operatorname{softmax}(QK^\top/\sqrt R)V'$, matmul $2AD_q$ | $8{,}192A$ | $16{,}384A$ |
| Attention output projection | $OW_O$, $2MD_qD$ | $33{,}554{,}432M$ | $83{,}886{,}080M$ |
| MLP gate projection | $XW_g$, $2MDI$ | $100{,}663{,}296M$ | $262{,}144{,}000M$ |
| MLP up projection | $XW_u$, $2MDI$ | $100{,}663{,}296M$ | $262{,}144{,}000M$ |
| MLP down projection | $[\operatorname{SiLU}(XW_g)\odot XW_u]W_d$, matmul $2MID$ | $100{,}663{,}296M$ | $262{,}144{,}000M$ |
| **Dominant layer total** | $M(4DD_q+4DD_{kv}+6DI)+4AD_q$ | $385{,}875{,}968M+16{,}384A$ | $975{,}175{,}680M+32{,}768A$ |
| Vocabulary head, once per model | $H_{selected}W_{head}$, $2UDV$ | $1{,}244{,}659{,}712U$ | $1{,}555{,}824{,}640U$ |

For completeness, the remaining operators are listed below. Their work is smaller than the large matmuls, but their memory traffic and launch costs can matter greatly. These are **approximate scalar-operation equivalents**, not tensor-core FLOPs or instruction counts. For this table, exp/rsqrt/division each count as one operation, sign changes and comparisons are omitted, and RMSNorm on a length-$n$ vector costs $4n+2$: squares, reduction, mean, epsilon, reciprocal square root, normalization and learned scaling.

| Other operator | Approximate count | Qwen3-8B | Qwen3-32B |
|---|---|---:|---:|
| Two residual-stream RMSNorms per layer | $2M(4D+2)$ | $32{,}772M$ | $40{,}964M$ |
| Q per-head RMSNorm | $MH_q(4R+2)$ | $16{,}448M$ | $32{,}896M$ |
| K per-head RMSNorm | $MH_{kv}(4R+2)$ | $4{,}112M$ | $4{,}112M$ |
| RoPE application to Q/K | $3M(D_q+D_{kv})$ | $15{,}360M$ | $27{,}648M$ |
| Score scaling | $AH_q$ | $32A$ | $64A$ |
| Stable softmax | $\approx5AH_q$ | $\approx160A$ | $\approx320A$ |
| SiLU and gate multiplication | $\approx5MI$ | $\approx61{,}440M$ | $\approx128{,}000M$ |
| Two residual additions | $2MD$ | $8{,}192M$ | $10{,}240M$ |
| Final RMSNorm, once per model | $M(4D+2)$ | $16{,}386M$ | $20{,}482M$ |
| RoPE angle/sin/cos preparation, shared across layers | $\approx3MR/2$ with reused trig pairs | $\approx192M$ | $\approx192M$ |
| Embedding lookup | $0$ arithmetic FLOPs; gather $MD$ elements | $4{,}096M$ elements | $5{,}120M$ elements |
| Cache append/read, head reshape, causal masking | $0$ arithmetic FLOPs in this convention | Traffic/indexing | Traffic/indexing |
| Greedy token choice, once per emitted batch | $0$ arithmetic FLOPs; $B(V-1)$ comparisons | $151{,}935B$ comparisons | $151{,}935B$ comparisons |

The RoPE preparation estimate assumes identical mathematical angles can be reused; implementations may duplicate them or share positions across the batch. Softmax reductions, masking and fusion also change executed operations. Norms often use FP32 internally. These qualifications are why the notebook uses the **dominant matmul ledger** to report achieved FLOP/s, while timing the complete forward. Small arithmetic counts do not imply zero latency.

### Causal useful work versus dense executed work

The useful attention count includes only allowed pairs. An eager implementation can nevertheless compute the entire score rectangle before applying a mask. Its pair count is

$$
A_{dense}=BT(P+T),
$$

which becomes $BS^2$ for prefill. Thus attention matmuls cost $2BD_qS(S+1)$ under the causal convention versus $4BD_qS^2$ under the dense convention. Causal kernels can skip much of the masked region, but tiling, padding and internal recomputation still affect executed work. Decode has one query, so the two counts agree. Always label the convention when comparing FLOP/s across engines.

## 4. From one layer to a full Qwen3 forward

Define the dense linear-work coefficient

$$
K=4DD_q+4DD_{kv}+6DI.
$$

Then the notebook's model FLOPs are

$$
\boxed{F_{model}=L(MK+4AD_q)+2UDV}.
$$

The embedding lookup does not cost $2MDV$: it gathers selected rows. The output projection does cost $2UDV$. Computing only the last prompt position's logits is sufficient to choose the first generated token; computing every prompt position is required for some scoring workloads. The notebook explicitly requests `logits_to_keep=1` so measurement and formula agree.

For generation-only prefill and one cached decode call:

$$
\boxed{F_{prefill}(B,S)=L\left[BSK+2BD_qS(S+1)\right]+2BDV},
$$

$$
\boxed{F_{decode}(B,P)=L\left[BK+4BD_q(P+1)\right]+2BDV}.
$$

With **$B=1$, a 2,048-token prompt, one last-position vocabulary projection, and causal matmul counting**, the independently substituted totals are:

| Full-model work | Qwen3-8B | Qwen3-32B |
|---|---:|---:|
| Prefill $S=2048$ | 29,688,662,589,440 FLOPs = 29.6887 TFLOPs | 132,219,976,548,352 FLOPs = 132.2200 TFLOPs |
| First decode call $P=2048$; attends 2,049 keys | 16,344,743,936 FLOPs = 16.3447 GFLOPs | 68,264,132,608 FLOPs = 68.2641 GFLOPs |

The same workload gives the following **GFLOPs per operator** (one decoder layer, except the LM head). Values are rounded to six decimal places. Multiply the layer subtotal by $L$, then add the head once to recover the full-model totals above.

| Operator | 8B prefill | 8B decode | 32B prefill | 32B decode |
|---|---:|---:|---:|---:|
| Q projection | 68.719477 | 0.033554 | 171.798692 | 0.083886 |
| K projection | 17.179869 | 0.008389 | 21.474836 | 0.010486 |
| V projection | 17.179869 | 0.008389 | 21.474836 | 0.010486 |
| Attention scores | 17.188258 | 0.016785 | 34.376516 | 0.033571 |
| Attention weighted values | 17.188258 | 0.016785 | 34.376516 | 0.033571 |
| Output projection | 68.719477 | 0.033554 | 171.798692 | 0.083886 |
| Gate projection | 206.158430 | 0.100663 | 536.870912 | 0.262144 |
| Up projection | 206.158430 | 0.100663 | 536.870912 | 0.262144 |
| Down projection | 206.158430 | 0.100663 | 536.870912 | 0.262144 |
| **Layer matmul subtotal** | 824.650498 | 0.419447 | 2,065.912824 | 1.042317 |
| LM head (once per model) | 1.244660 | 1.244660 | 1.555825 | 1.555825 |

A projection's input length grows linearly with prefill tokens; causal attention grows quadratically. Cached decode processes only one new token per request, but its attention grows linearly with prefix length. KV caching avoids recomputing old token projections and MLPs; it does not eliminate reading old keys and values.

For $N$ generated tokens, prefill already supplies token 1. There are $N-1$ subsequent decode calls, with prefixes $S,S+1,\ldots,S+N-2$. Sum each call's FLOPs and time using its own prefix. The notebook's `DECODE_STEPS=8` therefore produces nine tokens including the first.

## 5. Weight traffic, KV traffic, and capacity
After calculating model FLOPs, this section analyze model weight and activation size. It's a estimate of how much data need to move between DRAM and GPU.

The layer's matrix parameter count is

$$
W_{layer}=2DD_q+2DD_{kv}+3DI.
$$

Two residual-stream norms add $2D$ weights; the Q/K norms add $2R$ weights shared across their respective heads. With untied embeddings and head, the parameter count is

$$
N_{params}=L(W_{layer}+2D+2R)+2DV+D.
$$

| Quantity derived from dimensions | Qwen3-8B | Qwen3-32B |
|---|---:|---:|
| Parameters | 8,190,735,360 | 32,762,123,264 |
| BF16 weights only, decimal GB | 16.3815 | 65.5242 |
| BF16 KV bytes per cached token per request, all layers | 147,456 | 262,144 |

The cache capacity for $B$ requests with $S$ cached tokens is

$$
Q_{KV,capacity}=2eLBSD_{kv}.
$$

Where $e$ is number of bytes per KV-cache elements, 2 bytes for BF16/FP16. The factor two is for K and V. GQA reduces this storage relative to full multi-head attention. Attention FLOPs still depend on $D_q$, so fewer KV heads do not proportionally reduce attention arithmetic.

For a first optimistic traffic proxy, assume all layer/head weights stream once per forward; embedding gathers read selected rows; old KV is read once; and new KV is written once:

$$
\begin{aligned}
Q_{weights}&=e\left[L(W_{layer}+2D+2R)+D+DV\right],\\
Q_{embed}&=eMD,\\
Q_{KV,read}&=2eLBPD_{kv},\\
Q_{KV,write}&=2eLBTD_{kv},\\
Q_{boundary}&=e(MD+UV),\\
Q_{proxy}&=Q_{weights}+Q_{embed}+Q_{KV,read}+Q_{KV,write}+Q_{boundary}.
\end{aligned}
$$

Here $Q_{boundary}$ accounts for a hidden-state output boundary and logits; this is deliberately a coarse compulsory-data model. In prefill, newly produced KV may feed attention on chip, so this model does not require an additional DRAM read of it. Real tiled kernels can reread KV. Dynamic caches can allocate and copy growing prefixes. Unfused activations, norms and residuals add traffic; eager attention may materialize score matrices. Neither capacity bytes nor this proxy is a measurement of actual transfers. Use $\mathrm{AI}_{proxy}=F/Q_{proxy}$ as a labeled hypothesis, then refine it with operator accounting and profiling. The [Scaling Book inference chapter](https://jax-ml.github.io/scaling-book/inference/) develops related weight/KV tradeoffs.

Weight capacity and weight traffic are different: input embeddings occupy memory even when only selected rows are accessed. Batch reuse amortizes a weight read across requests, while their independent KV caches scale with batch. Increasing batch can improve aggregate token throughput while increasing each request's token latency.

## 6. TTFT and TBT: model execution versus user experience

Let $t_{arrival}$ be request arrival and $t_i$ the time generated token $i$ becomes available to the user:

$$
\operatorname{TTFT}=t_1-t_{arrival},\qquad
\operatorname{TBT}_i=t_i-t_{i-1}\quad(i\geq2),
$$

$$
\overline{\operatorname{TBT}}=\frac{t_N-t_1}{N-1}\quad(N>1),\qquad
 t_{completion}-t_{arrival}=\operatorname{TTFT}+\sum_{i=2}^{N}\operatorname{TBT}_i.
$$

TTFT is **time to first token**. It can include queueing, tokenization, admission, transfers, prompt prefill, first-token selection and delivery. TBT is **time between tokens**, also called inter-token latency; it can include decode scheduling, forward work, selection and delivery. Prompt length mainly affects prefill work and KV state. Generated length determines how many subsequent token intervals occur. TTFT and TBT are complementary: a system can start quickly but generate slowly, or the reverse.

The lab deliberately measures narrower, reproducible boundaries:

| Recorded value | Boundary | Interpretation |
|---|---|---|
| `cuda_ms`, prefill | CUDA events around prompt forward, cache creation and final-position logits | Prefill forward interval; device idle gaps within the interval can be included |
| `cuda_ms`, decode | CUDA events around one cached forward | Decode forward interval at the recorded prefix length |
| `token_ready_ms`, prefill | Synchronized wall time from device-ready prompt to greedy token ready | Local TTFT; no queue, tokenizer, request transfer or network |
| `token_ready_ms`, decode | Synchronized wall time through one forward and greedy choice | Local TBT; instrumentation forces a per-token synchronization |

Use these labels rather than claiming an isolated forward benchmark measures production serving latency. CUDA execution is asynchronous; unsynchronized Python timing can measure launch submission alone. The lab warms the full prefill/decode path, uses inference mode, synchronizes appropriate boundaries, retains raw repeats, and rebuilds the cache for every independent run. It does not mix model loading or tokenization into forward time. [PyTorch CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)

For a prompt of $S$ tokens per sequence and batch $B$, prefill input throughput is $BS/t_{prefill}$ in tok/s, with $t_{prefill}$ in seconds. This counts processed prompt tokens, even though prefill produces only one first output token per sequence. Use the declared synchronized wall interval through first-token selection for this local metric.

For a fixed batch, each decode step emits $B$ tokens. Aggregate decode throughput is $B/t_{step}$; it is not the reciprocal of each request's TBT multiplied by prompt length. For $K$ decode calls at growing contexts, output throughput is $BK/\sum_{j=1}^{K}t_j$ and mean local TBT is $\sum_j t_j/K$. Exclude prefill and its first output token from both decode totals. Calculate aggregate token throughput as total emitted tokens divided by total time, and achieved FLOP/s as summed work divided by summed time, rather than averaging per-step rates.

Compare these metrics across a batch-size × initial-context-length grid. Hold context fixed to study batch scaling, and hold batch fixed to study context scaling. Prefill input tok/s and decode output tok/s describe different work. Summarize each repeat separately before taking medians and ranges; retain every decode prefix to show the growing-cache window.

### Analytical latency–throughput tradeoffs

At fixed prompt length $S$ tokens per sequence, vary batch size $B$. Using the
previous FLOP and traffic ledgers, define the predicted prefill time in seconds
as $\widehat t_{prefill}=\max(F_{prefill}/C,Q_{prefill}/\beta)$. Treating this
forward-only bound as an optimistic local TTFT gives the prefill chart coordinates

$$
(x,y)_{prefill}=\left(1000\widehat t_{prefill},
\frac{BS}{\widehat t_{prefill}}\right),
$$

where $x$ is milliseconds and $y$ is aggregate input tok/s. The batch processes
$BS$ prompt tokens, although its prefill logits supply only $B$ first output
tokens. If compute dominates and work grows linearly with batch at fixed $S$,
TTFT grows roughly with $B$ while input throughput approaches a plateau.

For a single decode step with $S$ cached tokens before appending, use
$\widehat t_{decode}=\max(F_{decode}/C,Q_{decode}/\beta)$, counting attention
over $S+1$ positions. **Interactivity** is the rate at which one user receives
output tokens. With one token per user per synchronous step, it is
$1/\widehat t_{decode}$; the batch produces $B$ tokens in that same time. Thus

$$
(x,y)_{decode}=\left(\frac{1}{\widehat t_{decode}},
\frac{B}{\widehat t_{decode}}\right),\qquad y=Bx.
$$

Here $x$ is output tok/s per user and $y$ is aggregate output tok/s. Increasing
batch amortizes shared weight traffic but increases per-request KV traffic and
total compute, so aggregate throughput can rise while interactivity falls.
For an **illustrative prediction**, a 50 ms decode step at $B=4$ gives
20 tok/s per user and 80 tok/s in total. A 1 s prefill at $S=2048$, $B=4$
gives 8,192 input tok/s and a predicted local TTFT of 1,000 ms.

These aggregate roofline predictions omit sequential-kernel effects, launch and
host overhead, token selection, queueing, and delivery. They use configured
hardware ceilings and a compulsory-byte proxy; they neither establish measured
performance nor guarantee memory capacity at the plotted batch sizes. The
[Section 2 notebook plots](code/lab.ipynb) label those assumptions and hold
context fixed so each point isolates a batch-size change.

## 7. Use the model to analyze an inference run

Follow the notebook's prediction → correctness → measurement → explanation sequence:

1. **Declare the workload.** Record checkpoint revision, dimensions, precision, batch, prompt length, generated-token count, cache, attention backend and logits positions. Choose causal or dense arithmetic counting.
2. **Configure hardware and calibrate.** Record GPU identity and the provenance of bandwidth/peak inputs. Measure streaming-copy traffic larger than cache, small-call intervals, and shape-specific GEMM rates. Preserve raw timing variation. Calibration measurements do not become advertised hardware peaks.
3. **Predict before observing.** Save operator counts, byte assumptions and predictions. Start with an aggregate bound, then sum sequential components for a more realistic predictor. Freeze six feasible held-out conditions spanning both models and phases before measuring them.
4. **Check then time.** Compare cached logits with a full-prefix oracle before accepting measurements. Run at least three independent uninstrumented repeats and retain each decode prefix. Capacity-limited points remain unmeasured; change the planned workload explicitly.
5. **Plot and diagnose.** Place measured $F/t$ at estimated $F/Q_{proxy}$, label assumptions, and report token throughput alongside local TTFT/TBT. Inspect prediction error $|\widehat t-t|/t$ and signed residuals. Min/max bars show observed spread, not confidence intervals.
6. **Test an intervention.** Use a separate trace to distinguish hypotheses. Freeze a new prediction and test a new condition; retain the original held-out score. Do not fit to a point and call it a held-out prediction.

| Observation | Candidate explanation | Discriminating measurement |
|---|---|---|
| Decode improves with batch | Weight reuse improves GEMM intensity | Same context, varying batch; inspect projection time and KV growth |
| Decode residual grows with prefix | KV reads, cache copies or attention efficiency | Attention/cache timings and transferred bytes versus context |
| Prefill is below the compute ceiling | Small/inefficient GEMMs, attention, scalar kernels or host gaps | Shape calibration plus timeline of dominant operators |
| Nearly constant latency gap | Dispatch, synchronization or omitted small operators | Launch counts and CPU/GPU gaps on a trace |
| Both achieved compute and bandwidth are low | Serialization, dependencies or an incomplete byte/work model | Critical-path trace before changing peak inputs |
| Serving TTFT rises but isolated prefill does not | Queueing or scheduling contention | Arrival-to-dispatch and token-delivery timestamps |

These are hypotheses, not diagnoses from a chart alone. [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) shows CPU/GPU dependencies; [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) provides kernel metrics and memory counters. Profiling replay changes timing, so keep profiling and ordinary latency measurements separate. Only compare counter bytes with FLOPs and timings covering the same work.

A model that misses the old 25% median-error target can still teach something if its failure is specific and the next experiment can distinguish causes. The [lab's completion checklist](code/lab.ipynb) retains calibration, held-out prediction, real-model coverage, and the profile-supported intervention. Use the [shared experiment protocol](../../shared/PROTOCOL.md) and [report rubric](../../shared/REPORT.md) for the full project.

## From performance modeling to kernel optimization

The next chapter uses the gap between these estimates and observations to
identify kernel optimization opportunities. The theoretical roofline describes
ideal limits; calibrated estimates describe expected behavior under stated
assumptions. A residual becomes actionable when a trace links it to a mechanism
such as repeated memory reads, materialized attention scores, poor small-shape
utilization, or launch overhead. The gap itself is not a promised speedup.

Chapter 3 tests a targeted kernel change in the same Qwen implementation and
compares its integrated effect with an Amdahl prediction. Chapter 4 then builds
scheduling and bounded KV memory around that measured model. This separates
isolated model/kernel latency from the queueing and ownership costs added by a
serving runtime.

## Reading path

| Read | Purpose |
|---|---|
| [Decoder architecture reference](https://www.ryhuang.blog/blogs/llm-arch/) | Reconstruct projections, attention, SwiGLU and vocabulary-head work; adapt generic dimensions to Qwen3 |
| [Roofline theory](https://jax-ml.github.io/scaling-book/roofline/) | Reproduce the roofline inequality, units and ridge point |
| [Transformer inference analysis](https://jax-ml.github.io/scaling-book/inference/) | Connect batching, weights, KV traffic and latency |
| [Qwen3 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3/modeling_qwen3.py) | Verify projection widths, Q/K norm and selected-position logits |
| [CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html) | Understand event recording and synchronization |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Investigate critical paths, physical traffic and profiling overhead |

Sources checked on 2026-09-12. Model configurations are pinned above; match API and profiler details to the versions recorded by each notebook run.

[Open the interactive lab](code/lab.ipynb) · [Previous: Reconstruct Qwen3](../01_reconstruct_qwen3/background.md) · [Course home](../../README.md) · [Next: Kernel optimization](../03_kernels/README.md)
