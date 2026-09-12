# LLM Inference Engineering

**Begin with [Chapter 0: From text to a serving system](chapters/00_introduction/README.md).** This conceptual introduction has no code or projects.

**Companion labs are now available:** see the [course chapter index](README.md)
and [laboratory setup](shared/SETUP.md). Each of the eight project folders contains
background, a step-by-step tutorial, executable teaching samples, and annotated
references. Chapter 1 consolidates theory/readings in `background.md` and practical
steps in [Lab 1](chapters/01_reconstruct_qwen3/code/lab.ipynb) and
[Lab 2](chapters/01_reconstruct_qwen3/code/lab2.ipynb), including its completion checklist.
The project scope and assessment requirements below remain the syllabus.

## A project-driven semester with Qwen3-8B and Qwen3-32B

**Course version:** 1.0 · September 11, 2026  
**Duration:** 16 weeks, approximately 15–18 focused hours per week  
**Primary laboratory:** your DGX Spark  
**Additional laboratory:** prepared rentals of connected H100 GPUs; B300 is an optional hardware comparison  
**Final product:** one evolving experimental inference engine, eight research reports, and the foundations of a textbook or blog series with an introduction and eight project chapters

The central question for the semester is:

> Given a model, workload, hardware configuration, and latency requirement, can I predict the bottleneck, implement an intervention, and explain the measured outcome?

Qwen3-8B is the everyday implementation and debugging model. Qwen3-32B tests whether your explanations survive a change in model size. During speculative decoding, 8B becomes the draft and 32B the target. This pairing is an experiment whose outcome must be measured.

This plan starts from your existing experience: PMPP, CuTe DSL elementwise and GEMM work, a Mini-SGLang replication, and familiarity with SGLang and Dynamo. It does not require repeating an introductory GPU programming course.

All timings, experiment sizes, assessment targets, and rental allocations below are proposed course choices. The architecture and hardware facts are sourced; performance results must come from your experiments. No benchmark has been run for this syllabus.

---

## 0. Introduction — architecture and the inference system

Begin Week 1 with [Chapter 0](chapters/00_introduction/README.md). Reserve
approximately 3–4 hours from the existing Week 1 reading allocation. This is a
conceptual chapter with equations, worked examples, and diagrams; it has no code,
project, assessment, or required submission. The eight projects and 16-week
schedule remain unchanged.

The reading proceeds from a token to the complete serving system:

1. **Transformer architecture:** autoregressive probability; encoder/decoder
   distinctions; tokenization and vocabulary tradeoffs; embeddings; residuals and
   RMSNorm; self-attention and GQA; rotary position encoding; SwiGLU MLPs; logits
   and sampling.
2. **LLM inference and serving:** prefill and decode; KV reuse and memory growth;
   request state; continuous batching and chunked prefill; vLLM and SGLang;
   paging, prefix reuse, latency, throughput, goodput, and speculative decoding.
3. **Distribution and performance modeling:** replicas, tensor and pipeline
   parallelism; prefill/decode disaggregation and state transfer; arithmetic
   intensity, critical paths, queueing, and capacity balance across kernel,
   model, server, and system boundaries.

The purpose is to establish the mathematical and systems intuition that later
projects make concrete. Chapter 0 uses the existing 8B/32B architecture examples
and introduces no additional model fixtures.

---

## 1. What you should be able to do afterward

By the end, you should be able to:

1. Reconstruct the Qwen3 forward computation, load published weights, and explain every major tensor shape.
2. Implement cached generation, request scheduling, continuous batching, and bounded KV allocation.
3. Predict memory requirements and estimate compute, memory, CPU, and communication costs.
4. Profile an unfamiliar result and test a specific explanation.
5. Implement and integrate a bounded CuTe DSL attention kernel; explain the design of an advanced GEMM.
6. Implement speculative verification and reason about correctness, acceptance, and break-even behavior.
7. Evaluate a real quantized configuration across quality, memory, latency, and serving capacity.
8. Implement basic tensor parallelism and compare it with replicas using equal hardware.
9. Transfer a request’s KV state between prefill and decode workers and quantify the costs.
10. Defend a serving design using reproducible evidence, including negative results.

**Success means a defensible explanation. A speedup is an experimental result, not a graduation requirement.**

### What “from scratch” means here

| Component | Your implementation | Existing building blocks and references |
|---|---|---|
| Model | Configuration, RMSNorm, RoPE, GQA, SwiGLU, decoder stack, cache semantics, weight mapping | Published Qwen weights, official tokenizer, PyTorch tensor operations and GEMM |
| Runtime | Request states, scheduler, block allocator, batching metadata, generation loop, metrics | CUDA/PyTorch execution; a FlashInfer or other supported attention adapter after the reference path works |
| Kernels | One bounded attention kernel and a measured modification to an existing GEMM exercise | CuTe DSL, vendor GEMM and attention libraries as comparisons |
| Speculation | Draft/verify/commit protocol, exact sampling reference, cache truncation and reconciliation | The two pretrained Qwen models; SGLang as a second implementation |
| Quantization | Quantize/dequantize reference, packing exercise, layer-error analysis | A maintained tool for AWQ or another selected production format; its actual execution backend |
| Distribution | Column/row sharding, placement choices, collective instrumentation, state handoff | PyTorch distributed and NCCL; Dynamo for comparative study |

You already have a Mini-SGLang replication. Reuse working infrastructure and record its provenance. Independently reconstruct the mechanisms you want to claim you understand, then integrate them into that repository. A second wholesale rewrite would spend time without necessarily adding depth. [Mini-SGLang source](https://github.com/sgl-project/mini-sglang)

The semester focuses on dense GQA inference. MLA, MoE, recurrent models, training, SFT/DPO, Kubernetes operators, production authentication, and fleet autoscaling are follow-up subjects. Section 14 defines bounded architecture extensions.

### Scope control

The required implementation is intentionally small: one scheduler policy plus one comparison, one cache layout plus one alternative, one deeply studied custom attention kernel, one real quantization path, TP=2, and one P/D handoff protocol.

Allow approximately 240–288 hours for the core. At 8–10 hours per week, use a 24–36 week schedule. Writing publication-ready chapters or building multiple state-of-the-art kernels would extend either schedule.

---

## 2. Know the two models precisely

Use these exact checkpoint families: **`Qwen/Qwen3-8B`** and **`Qwen/Qwen3-32B`**. They are dense Qwen3 decoder models. Their declared architecture is `Qwen3ForCausalLM`; this course uses conventional GQA and KV state.

| Configuration field | Qwen3-8B | Qwen3-32B |
|---|---:|---:|
| Decoder layers, \(L\) | 36 | 64 |
| Hidden width, \(D\) | 4,096 | 5,120 |
| MLP intermediate width, \(I\) | 12,288 | 25,600 |
| Query heads, \(H_q\) | 32 | 64 |
| KV heads, \(H_{kv}\) | 8 | 8 |
| Head dimension, \(d_h\) | 128 | 128 |
| Query projection output width, \(H_qd_h\) | 4,096 | 8,192 |
| Vocabulary size | 151,936 | 151,936 |
| Tied input/output embeddings | No | No |

Architecture values follow the [8B configuration](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) and [32B configuration](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json).

**A useful implementation trap:** in 32B, `hidden_size / num_attention_heads = 80`, but the head dimension is **128**. Read the explicit field. The attention output projection maps 8,192 features back to 5,120.

Qwen3 also applies learned RMS normalization to Q and K over the head dimension before RoPE. Include those weights. The MLP uses a SiLU gate multiplied by the up projection, followed by the down projection. Use the actual implementation to resolve normalization, positional encoding, and residual ordering. [Transformers Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)

Keep the core experiments near 8K total tokens or below; add 16K only when a question needs it. Allow space for both prompt and generated tokens in the engine's context setting. The model cards describe 32,768 native context and extended context using YaRN, while the configurations declare 40,960 maximum positions. Do not treat those as interchangeable claims or change RoPE scaling during a controlled comparison. [8B model card](https://huggingface.co/Qwen/Qwen3-8B), [32B model card](https://huggingface.co/Qwen/Qwen3-32B)

### First memory calculations

For cache element size \(b_{kv}\) bytes, unsharded KV storage is:

\[
M_{KV}=2L H_{kv}d_h b_{kv}\sum_{i=1}^{B}S_i.
\]

The factor 2 is for K and V. Each \(S_i\) is the cached length of a live request.

| Derived quantity | 8B | 32B |
|---|---:|---:|
| Parameter count from the declared architecture | 8,190,735,360 | 32,762,123,264 |
| BF16 parameter storage | 16.38 GB / 15.26 GiB | 65.52 GB / 61.02 GiB |
| BF16 KV bytes per token per request | 144 KiB | 256 KiB |
| BF16 KV at 2,048 tokens per request | 0.28125 GiB | 0.5 GiB |
| BF16 KV at 8,192 tokens per request | 1.125 GiB | 2 GiB |
| BF16 KV at 32,768 tokens per request | 4.5 GiB | 8 GiB |

These are calculations from the configuration and module structure, not allocator measurements. Validate parameter counts against checkpoint tensor metadata in Project 1. GB means \(10^9\) bytes; GiB means \(2^{30}\) bytes.

Both models together require approximately **81.91 GB / 76.28 GiB for BF16 parameters alone**. At 8K cached tokens, their two caches add **3.125 GiB per speculative request**, before workspaces, temporary verification positions, metadata, or loading overhead.

For block size \(P\), derive allocated KV bytes with:

\[
M_{KV,\mathrm{allocated}}=
2L H_{kv}d_h b_{kv}
\sum_i P\left\lceil \frac{S_i}{P}\right\rceil,
\]

before prefix sharing. Count physical shared blocks once. Allocation reservation can be much larger than the currently populated cache.

**Week 1 question:** how much of a measured memory discrepancy comes from parameter storage, live KV, reserved KV, temporary tensors, or duplicated loading buffers?

---

## 3. Laboratory and rental strategy

### Spark is the default

DGX Spark has 128 GB of unified LPDDR5x system memory with a stated 273 GB/s bandwidth. That supports useful memory-capacity experiments, but all 128 GB is not available to your model process. CPU allocations, the operating system, GPU workspaces, and loading copies also consume memory. Unified memory does not imply that every framework avoids copies. [NVIDIA Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

Use a working Spark-compatible environment, then pin its container digest and dependencies. NVIDIA provides an SGLang Spark recipe using Qwen3-8B, which is a suitable setup reference. Record the actual attention and GEMM backends chosen at runtime. [NVIDIA SGLang on Spark instructions](https://build.nvidia.com/spark/sglang/instructions)

Do not load a BF16 reference 32B model and your own complete BF16 32B model simultaneously. Compare them sequentially using saved logits, selected intermediate tensors, and identical token IDs. Stream checkpoint shards or use a low-memory loading strategy.

### Hardware determines the advanced kernel lab

| Hardware | CUDA compute capability | Main course use |
|---|---:|---|
| Spark / GB10 | 12.1 | Model, scheduler, cache, numerical experiments, selected supported kernels |
| H100 | 9.0 | Hopper GEMM/attention study and connected multi-GPU experiments |
| B300 | 10.3 | Optional data-center Blackwell kernel and scaling comparison |

The compute capabilities are listed in [NVIDIA's hardware table](https://developer.nvidia.com/cuda/gpus).

A kernel described as “Blackwell” may target a different instruction set and execution design from GB10. Check the exact architecture, toolkit, DSL revision, layout, and dtype. Run a tiny compile-and-correctness check before planning a large experiment.

The official FlashAttention repository includes a CuTe DSL implementation, FlashAttention-4, and a Hopper-specific FlashAttention-3 path. These are reading and benchmarking references; support for your exact GPU must be checked in the pinned release. [FlashAttention source](https://github.com/Dao-AILab/flash-attention), [CuTe DSL setup requirements](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)

Use H100 for the first advanced architecture-specific study. Add B300 after you have a specific comparison and working code. Kernel portability is a valid research question, but it should not block the model or scheduling projects.

### Prepared compute allocation

This is a planning allowance, not a rental quote or a measured runtime.

| Session | Allocation | GPU-hours |
|---|---:|---:|
| Single-GPU profiling, kernel and quantization checks | 1 GPU × 8 hours | 8 |
| Tensor-parallel and replica comparison | 2 connected GPUs × 8 hours | 16 |
| P/D transfer and serving comparison | 2 connected GPUs × 8 hours | 16 |
| Cross-node communication comparison | 2 GPUs across 2 nodes × 4 hours | 8 |
| Reruns and failures | Reserved allowance | 12 |
| **Core allowance** | | **60** |
| Optional B300 comparison | 1 GPU × 8 hours | **+8** |

A provider may bill an entire eight-GPU node even when you use two GPUs. Calculate cost from the actual billed allocation, duration, storage, and transfer charges.

Before starting a rental session, have a runnable experiment manifest, tiny correctness checks, expected tensor sizes, and a prioritized run list. Capture GPU SKU, topology, visible memory, driver/toolkit, and actual collective performance. “Two GPUs” does not specify whether the path is NVLink, PCIe, or a network.

A single 80 GB H100 is useful for a controlled 32B baseline, but the two BF16 models together leave little or no practical headroom. Do the first full-precision speculative pair experiment on Spark at batch 1. If a rental requires splitting the models across GPUs, measure that placement separately and account for both GPUs.

Cross-node work requires two real hosts. Two processes on Spark validate protocol logic but cannot establish multi-node performance. If that rental is unavailable, retain the explicit unmeasured extension and complete the same-node course.

---

## 4. The semester at a glance

| Weeks | Project | Research question | Required evidence |
|---|---|---|---|
| 1–2 | P1. Implement Qwen3 | Can one implementation reproduce both checkpoints? | Tensor inventory, weight mapping, cached/full equivalence, first memory accounting |
| 3–4 | P2. Build the runtime | How do scheduling and KV allocation change useful capacity? | Continuous batching, allocator invariants, latency/throughput and fragmentation curves |
| 5–6 | P3. Predict performance | Which costs explain prefill and decode across sizes? | Shape inventory, calibrated model, predictions on withheld workloads |
| 7–8 | P4. Optimize a kernel | Does a local kernel change improve model execution? | GEMM investigation, custom attention kernel, correctness, traces, Amdahl analysis |
| 9–10 | P5. Speculate with 8B → 32B | When does the draft pay for itself? | Exact verifier, state reconciliation, acceptance distribution, measured break-even |
| 11–12 | P6. Quantize and evaluate | Where does lower precision produce a useful tradeoff? | Real quantized execution, quality/memory/performance comparison, speculation interaction |
| 13–14 | P7. Distribute execution | When should a fixed GPU budget use TP or replicas? | TP implementation, collectives, fixed-resource comparison, scaling predictions |
| 15–16 | P8. Separate prefill/decode | When does isolation outweigh transfer and queueing? | KV handoff correctness, transfer model, fixed-resource P/D result, final design defense |

Each two-week project produces a 1,200–2,000 word research memo, at most a few essential figures, an experiment manifest, and a tagged code revision. These are chapter drafts. Polishing all eight into a finished textbook is a later editorial phase.

### Weekly rhythm

A workable 16-hour allocation is:

- 3 hours: assigned reading and a short derivation.
- 8 hours: implementation and correctness.
- 3 hours: measurement and analysis.
- 2 hours: writing and an oral explanation.

Before a benchmark, write the expected direction and likely cause. Afterward, preserve the original prediction and explain discrepancies. Each week, reconstruct one core mechanism without an AI-generated implementation and explain one result aloud.

---

## 5. One experimental protocol for every project

### Fixed workloads

Use these as initial fixtures. Lengths are token counts after applying the selected prompt template.

| Workload | Input tokens | Output tokens | Question |
|---|---:|---:|---|
| W1: Short interaction | 256 | 128 | Launch overhead and low-load latency |
| W2: Main comparison | 2,048 | 256 | Balanced serving behavior |
| W3: Long prefill | 8,192 | 128 | Attention and prefill interference |
| W4: Long generation | 256 | 1,024 | Sustained decode behavior |
| W5: Reused context | 2,048 shared + 256 unique | 256 | Prefix reuse and cache ownership |

For W3, either lower input length to leave room within an 8,192 total-length engine setting or explicitly raise that setting, for example to 8,448. Input length and total context length must be recorded separately.

Run broad sweeps on 8B. Repeat only selected decisive points on 32B: initially W1/W2/W3 at batch 1, then feasible batching points. Begin with batch sizes 1, 4, and 16, admitting only configurations within the memory budget.

Use synthetic token fixtures for isolated shape/timing experiments, with a fixed output length and explicitly documented EOS handling. Use real prompts for speculation acceptance and quality evaluation. Random token streams do not establish useful draft acceptance or model quality.

Keep `enable_thinking=False` for the core experiments and save the exact rendered input token IDs. Greedy generation is useful for correctness checks; quality evaluation must declare its decoding policy. Thinking-mode quality experiments are separate and should follow the model card’s guidance, which cautions against greedy thinking-mode decoding. [Qwen3 usage guidance](https://huggingface.co/Qwen/Qwen3-8B)

### Three kinds of measurement

1. **Kernel:** exact shapes, layout, precision, CUDA event time, repeat count.
2. **Model execution:** prefill time, decode-step time, GPU memory, CPU/GPU timeline.
3. **Serving:** arrival-to-first-token time, token emission timestamps, completion latency, offered load, success rate, and goodput.

Measure time to first token (TTFT) from request submission, including queueing. Measure inter-token latency (ITL) from successive emitted token timestamps. Also record server-side execution time so queueing and computation can be separated.

For speculation, tokens can be released in bursts. Report the distribution of client-visible gaps, per-request time per output token, verified tokens per cycle, and total completion time. Mean ITL alone can hide long pauses.

Define a serving goodput metric before comparisons:

\[
G=\frac{\text{successful completed requests satisfying the declared latency conditions}}
{\text{measurement duration}}.
\]

For example, a request can qualify if its TTFT is at most \(\tau_F\) and its within-request p95 ITL is at most \(\tau_I\). Report these thresholds numerically. Short outputs need a clearly defined token-gap statistic.

Choose the thresholds after the initial baseline, before tuning, and freeze them for each model/hardware/workload comparison. Separately report aggregate output tokens/s, failed requests, and latency percentiles. Count only emitted target tokens as useful output, excluding rejected drafts.

### Load and repetition

Use closed-loop fixed concurrency to study saturation, and open-loop arrivals to study queueing. Start open-loop rates at 25%, 50%, 75%, and 90% of one frozen baseline’s measured capacity. Use the same actual rates and arrival traces for all configurations.

Warm up until compilation and graph capture are complete. For model timing, use at least three independent repeats. For serving, aim initially for three traces of approximately 200 completed requests per condition; narrow the matrix if Spark runtime is prohibitive. Show spread and sample counts. Larger samples are needed for reliable extreme-tail claims; p99 from a small run is exploratory.

Use fixed prompts, seeds, generation settings, input/output lengths, and cache conditions. Run cold-prefix and warm-prefix cases separately. Do not put a per-token synchronization into production timing merely because a microbenchmark needs synchronization. Profiling runs are separate from uninstrumented timing runs.

These choices build on the distinctions in SGLang’s [benchmarking/profiling guide](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling) and [serving benchmark guide](https://docs.sglang.io/docs/developer_guide/bench_serving).

### Every result row needs provenance

Record:

```yaml
experiment_id: p3-w2-8b-bf16-b4
hypothesis: "Write this before measuring."
model_id: Qwen/Qwen3-8B
model_revision: "<resolved checkpoint commit>"
tokenizer_revision: "<resolved tokenizer commit>"
code_commit: "<your repository commit>"
reference_engine_commit: "<resolved engine commit>"
container_digest: "<resolved image digest>"
gpu_model: "<measured>"
compute_capability: "<measured>"
topology: "<single GPU / recorded links>"
dtype: bf16
attention_backend: "<actual selected backend>"
weight_backend: "<actual selected backend>"
kv_dtype: bf16
input_tokens: 2048
output_tokens: 256
batch_or_concurrency: 4
arrival_trace: "<fixture ID or null>"
cache_condition: cold_prefix
thinking: false
sampling: "<complete settings>"
seed: 42
warmup_policy: "<declared>"
repeats: 3
```

Add CUDA graphs, compilation, allocator reservation, KV block size, clocks/thermal observations, quantization metadata, and SLO thresholds when relevant. Preserve raw measurements and the script that generates each figure.

---

## 6. Project 1 — Reconstruct Qwen3 and prove the forward path

**Weeks 1–2 · Default: Spark · Main model: 8B; validation: 32B**

**Research question:** can a configuration-driven implementation reproduce both models, including cached decoding?

**Read first:** the two checkpoint configurations and model cards; the Qwen3 implementation; selected architecture/resource-accounting material from [Stanford CS336](https://cs336.stanford.edu/). Use the [Qwen3 technical report](https://arxiv.org/abs/2505.09388) for architectural context rather than reproducing its training pipeline.

### Week 1: implement the mathematical path

Implement RMSNorm with appropriate accumulation precision, RoPE with explicit absolute positions, Q/K normalization, GQA with causal masking, SwiGLU, residual blocks, final normalization, and the untied output head.

Use the first two real Qwen3-8B layers, with its embeddings, final norm, and vocabulary head, as the single course Qwen3 tiny configuration `(L,D,I,Hq,Hkv,R,V)=(2,4096,12288,32,8,128,151936)` for Chapter 1 correctness and DGX Spark timing in FP32. CPU execution is an optional untimed mathematical oracle. Keep the model dimensions fixed while varying workloads. Then move to real 8B and 32B checkpoints; subsequent chapters use those models. Compare each block with a simple mathematical oracle. Use a rectangular attention case and unequal query/KV head counts.

Then implement checkpoint loading. Produce a table mapping every checkpoint tensor name and shape to your module. Assert that required weights are consumed and shapes match. Avoid silently initializing an unmapped Q/K norm.

Derive the parameter count. For the declared bias-free structure, a useful audit is:

\[
P=L\left(2DH_qd_h+2DH_{kv}d_h+3DI+2D+2d_h\right)+2VD+D.
\]

Explain which terms correspond to attention matrices, MLP matrices, normalizations, and the two vocabulary matrices. Verify this count using checkpoint metadata.

### Week 2: add cache semantics and validate the full model

Add explicit per-request position and cache length. Compare:

- Full-prefix recomputation with one-token incremental decoding.
- A whole prefill with prefill split into unequal chunks.
- A single request with the same request in a mixed-length batch.
- Selected layers and logits against the reference 8B model.
- A short 32B prefill and continuation using the same implementation.

A cached multi-token query needs a mask offset by the existing prefix. For a query at position \(p+j\), allowed key positions satisfy \(k\le p+j\). Test this explicitly; rectangular causal masks are a common source of apparently plausible but incorrect outputs.

Use full-vocabulary logits for selected positions, not just a readable sample answer. Save intermediate tensors sequentially so two 32B model copies need not coexist.

### Experiments

For S = 128, 512, and 2,048, compare cached decoding with recomputation. Predict complexity and memory first. Use the reference backend to establish empirical numerical differences for BF16.

On tiny FP32 fixtures, begin with tolerances such as `rtol=1e-4, atol=1e-5` and investigate violations. Full BF16 comparisons require an empirically justified tolerance; do not inflate it merely to pass. Inspect the first divergent layer and the top-token logit margin. Floating-point reduction differences can change an argmax near a tie.

Measure parameter storage, peak loading memory, and live/reserved device memory separately. Check the derived cache slope against several lengths.

### Completion criterion

Both models load without shape-specific patches; cache/chunk/batch equivalence checks pass under documented numerical criteria; memory discrepancies have an explanation.

**Chapter draft:** “Reconstructing Qwen3: what actually happens during one token.”

**Oral defense:** why can the 32B query projection be wider than its residual stream, and why is its KV cache less than four times the 8B cache?

**Extension:** 16K context validation without changing positional encoding.

---

## 7. Project 2 — Build a runtime around bounded KV memory

**Weeks 3–4 · Spark · Broad experiments: 8B; selected checks: 32B**

**Research question:** which scheduling and memory decisions change the amount of work the system can complete within a latency requirement?

**Read first:** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) for iteration-level scheduling; [PagedAttention](https://arxiv.org/abs/2309.06180) for allocation/sharing; Mini-SGLang’s scheduler and cache code as the source comparison. Read the cache section of the [SGLang paper](https://arxiv.org/abs/2312.07104) when adding prefix reuse.

### Week 3: make the state machine explicit

Implement request states for waiting, prefill, decode, completed, and cancelled. Track token IDs, cache handles, absolute positions, generated length, and timing.

Begin with static batches and a contiguous KV buffer. Then add a physical block pool and per-request block tables. Admission must reserve enough working space for the next scheduled step; do not rely on an OOM exception as the scheduling policy.

Build a scheduler that selects work under both a token budget and a free-block budget. Add continuously changing decode batches. Define how a newly admitted prefill shares an iteration with existing decoding requests.

Use a simple gather-to-contiguous attention adapter for correctness if necessary. Count and profile its gather cost. Replace it with a supported paged-attention backend for a meaningful efficient runtime baseline. Owning a block table does not by itself make memory access efficient.

### Week 4: investigate one interference problem

Add chunked prefill and compare unchunked execution with two chunk sizes, initially 256 and 1,024 tokens. Keep the scheduling policy explicit: for example, reserve decode work first, then fill the remaining token budget with prefill chunks.

Add cancellation and release behavior. Add whole-block exact-prefix reuse with reference counts and read-only shared blocks. A full radix-tree implementation is an extension. If shared partial blocks can be extended, implement copy-on-write or keep that case unsupported and explicit.

Run an arrival trace mixing short interactions with long-prefill requests. The hypothesis is that long prefills can delay decode iterations; chunking may reduce that delay while changing prefill efficiency.

### Correctness and measurements

Check that batching, chunk size, and prefix reuse preserve the intended logits under your numerical criteria. Verify that cancelled/completed requests release their blocks, live requests never share writable blocks accidentally, and cache exhaustion produces bounded waiting or a declared rejection.

Plot throughput versus TTFT/ITL, useful versus allocated KV bytes, and the time spent in prefill/decode/gather/scheduling.

Repeat one representative comparison on 32B. Keep model quality out of the scheduler comparison by changing only runtime behavior.

### Completion criterion

The runtime handles interleaved request lengths and cancellation without stale cache state, and you can explain one scheduling tradeoff with a trace and a workload curve.

**Chapter draft:** “From token generation to an inference scheduler.”

**Oral defense:** why might smaller prefill chunks improve ITL while hurting total throughput?

**Extension:** CUDA graph buckets or more sophisticated prefix-cache eviction, after correctness is stable.

---

## 8. Project 3 — Build a model that predicts performance

**Weeks 5–6 · Spark; optional prepared H100 baseline**

**Research question:** how much performance can you explain from tensor shapes and measured hardware behavior before timing the whole model?

**Read first:** [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/), selected [transformer math](https://jax-ml.github.io/scaling-book/transformers/), and [transformer inference](https://jax-ml.github.io/scaling-book/inference/). Use [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) for the timeline and [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) for a few important kernels.

### Week 5: count operations and calibrate components

Record actual GEMM shapes as \([M,K]\times[K,N]\), including whether projections are fused. For decode, M is the scheduled token count; for prefill it is the flattened batch/chunk token count.

| Projection | 8B: K → N | 32B: K → N |
|---|---:|---:|
| Q | 4,096 → 4,096 | 5,120 → 8,192 |
| K or V | 4,096 → 1,024 | 5,120 → 1,024 |
| Attention output | 4,096 → 4,096 | 8,192 → 5,120 |
| Gate or up | 4,096 → 12,288 | 5,120 → 25,600 |
| Down | 12,288 → 4,096 | 25,600 → 5,120 |
| Vocabulary projection | 4,096 → 151,936 | 5,120 → 151,936 |

These are derived logical shapes. Stored layouts and fused kernels may differ.

For a GEMM, start with \(2MKN\) FLOPs. Derive the attention FLOPs from actual query-key pairs. For one sequence of length S, causal prefill attention has approximately \(2LH_qd_hS(S+1)\) FLOPs for QK and AV together, excluding softmax. One decode token attending to S cached positions has approximately \(4LH_qd_hS\) FLOPs. GQA reduces unique KV storage; its query heads still do attention arithmetic.

Calibrate a streaming bandwidth benchmark, representative GEMMs at M = 1, 4, 16, 128, and 512, and a small set of attention shapes. Use dense compute throughput for the actual precision. Spark’s advertised sparse FP4 performance is not BF16 compute throughput.

### Week 6: predict, then measure withheld workloads

For each sequential kernel j, start with:

\[
t_j\approx \max(F_j/C_{\mathrm{eff},j},\,Q_j/B_{\mathrm{eff},j})
+t_{\mathrm{launch},j}.
\]

Sum the relevant kernel costs, then add measured CPU scheduling and synchronization costs. Refine using the actual execution timeline. A single global maximum over the entire model can hide sequential attention and MLP costs.

Maintain separate prefill, decode, and queueing models. Add a small replay/simulation model only after the service-time model works; a complicated simulator is unnecessary here.

Fit on selected 8B shapes. Withhold an intermediate batch size and a context length. Predict them before measuring. Then predict selected 32B points from its dimensions. If an unseen shape requires a separate microbenchmark, label that as additional calibration.

### A useful initial estimate

For small-batch decode, assume most matrix weights are streamed once per step. Input embedding lookup touches selected rows, while the output vocabulary projection reads its matrix. Excluding the full input embedding matrix gives approximately 15.14 GB of streamed BF16 parameters for 8B and 63.97 GB for 32B.

At the advertised 273 GB/s, this gives idealized weight-read terms of approximately **55 ms** and **234 ms** respectively. These are starting estimates under the stated streaming assumption, not benchmark predictions or strict universal lower bounds. They omit KV reads, activations, compute, launch overhead, imperfect bandwidth, and cache effects.

At long context, unique KV reads grow with batch and length. For a simple implementation, model decode traffic as streamed weights plus relevant KV and activation traffic. Account for duplicate KV reads if the kernel fails to reuse data across query heads.

### Completion criterion

Publish predicted versus observed prefill/decode time on at least six withheld feasible points spanning the pair. Aim initially for median relative error below 25%; this is a diagnostic target. If you miss it, identify where the model fails and propose the next discriminating measurement.

**Chapter draft:** “Predicting inference latency before pressing Run.”

**Oral defense:** how can a model use little compute yet achieve poor memory bandwidth?

**Extension:** repeat a prepared subset on H100 and test which calibrated terms need changing.

---

## 9. Project 4 — Connect CuTe kernels to model performance

**Weeks 7–8 · Spark-compatible kernels; H100 for the selected Hopper study**

**Research question:** does the operation you optimized matter to the whole inference workload?

**Read first:** [FlashAttention](https://arxiv.org/abs/2205.14135) for tiled attention and online normalization, [FlashAttention-2](https://arxiv.org/abs/2307.08691) for work partitioning, and the [CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html). For the H100 investigation, study [FlashAttention-3](https://arxiv.org/abs/2407.08608) and one relevant official implementation. Use the [FlashInfer paper](https://arxiv.org/abs/2501.01005) and [source](https://github.com/flashinfer-ai/flashinfer) to connect attention layout with serving.

### Week 7: deepen GEMM using the model’s shapes

Take one GEMM from your existing CuTe work. Benchmark model-derived shapes at M = 1, 4, 16, 128, and 512. Compare with the actual vendor/backend implementation used in the runtime.

Study one advanced implementation for the chosen GPU: data movement, tile layout, staging, synchronization, and epilogue. On Hopper, this can include TMA, warp-group MMA, and producer/consumer work division. Follow instructions supported by that architecture; do not assume the same design runs on Spark.

Make one bounded change to your implementation: tile size, stage count, memory layout, or epilogue fusion. Predict whether it helps large prefill batches, small decode batches, or neither. Explain memory traffic and occupancy/resource effects using a small number of relevant counters.

Stop after one explained modification. Reimplementing every advanced GEMM algorithm is outside the core.

### Week 8: implement attention with a clear boundary

The recommended deep implementation is a **forward-only GQA decode kernel** with head dimension 128, BF16 inputs/cache, and FP32 accumulation.

Begin with contiguous KV, one query token per request, fixed head counts, and ragged sequence lengths. Implement tiled reads and stable online softmax. Add context splitting and combine partial statistics if profiling suggests insufficient parallelism.

For partial attention results with maxima \(m_a\), normalizers \(\ell_a\), and unnormalized outputs \(u_a\), derive a numerically stable merge using a shared maximum. Verify that changing the split does not change the mathematical result.

Also implement a small PyTorch tiled-prefill reference that applies the same online-softmax idea without materializing the full score matrix. This provides the FlashAttention derivation. A competitive CuTe prefill kernel and backward attention are extensions.

Integrate the decode kernel through your runtime’s attention interface. If your kernel only accepts contiguous KV, measure its gather cost and show both kernel-only and integrated timing. A paged-KV extension is useful only after the simpler kernel is correct.

If your measured bottleneck is prefill and you strongly prefer that specialization, choose a bounded forward-only CuTe prefill kernel instead. Keep the same correctness and integration requirements, and retain decode attention as a measured library comparison.

### Correctness and experiments

Check S = 127, 128, 129, 2,048, and 8,192; batches 1 and 4; extreme logits; and every partial-tile mask. Compare to an FP32 reference on manageable shapes. Run a memory/race checker appropriate to the chosen toolchain where available.

Benchmark matched dtype, head grouping, masks, and layouts. For bandwidth calculations, distinguish logical minimum bytes from repeated physical reads. Do not count GQA’s shared KV as if every query head stored a separate cache.

Measure the fraction f of model time originally spent in the optimized operation. If the kernel speedup is s, compare the measured model speedup with:

\[
S_{\mathrm{Amdahl}}=\frac{1}{(1-f)+f/s}.
\]

A small or negative integrated improvement is useful evidence when the adapter, launch cost, or unchanged operations dominate.

### Completion criterion

One model-derived GEMM change is explained, one custom attention path is numerically validated, and its integrated result is measured on a declared workload. A production-library performance gap is acceptable if you can explain the evidence and remaining uncertainty.

**Chapter draft:** “Why a faster attention kernel may barely change token latency.”

**Oral defense:** why do prefill and decode attention need different parallelization choices?

**Extension:** paged addressing, more GQA reuse, a full CuTe prefill path, or a B300 port. Pick one.

---

## 10. Project 5 — Speculative decoding with 8B drafting for 32B

**Weeks 9–10 · Start on Spark with batch 1 and short contexts**

**Research question:** is the 8B draft sufficiently cheap and sufficiently accurate to improve 32B generation?

The main pair remains unchanged: **draft = Qwen3-8B; target = Qwen3-32B**. Do not assume that a fourfold difference in total parameters guarantees a good draft. Sequential draft passes, vocabulary projections, cache maintenance, and acceptance all matter.

Use conventional separate-model speculative sampling. Native MTP or an EAGLE head is not part of the implementation specified here. SGLang currently documents a `STANDALONE` draft-model path, so it provides a useful production comparison after your own verifier works. [SGLang speculative decoding documentation](https://docs.sglang.io/docs/advanced_features/speculative_decoding)

**Read first:** [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) and [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318). Concentrate on the verifier and its correctness assumptions.

### Week 9: implement verification before optimizing it

Check the pinned tokenizers’ full token-to-ID mapping, merges, added/special tokens, and EOS handling. Matching vocabulary sizes alone is insufficient. Both models must see the same token sequence; render the prompt once and share its IDs.

First implement a tiny finite-vocabulary stochastic oracle. Let \(q_i\) be the draft distribution and \(p_i\) the target distribution at the same accepted history. For draft token \(x_i\sim q_i\), accept with:

\[
a_i=\min\left(1,\frac{p_i(x_i)}{q_i(x_i)}\right).
\]

At the first rejection, sample from normalized \(\max(p_i-q_i,0)\). If all k drafts are accepted, sample one extra token from the target’s next distribution. Implement the all-accepted, immediate-rejection, EOS, and maximum-length boundaries.

The probability vectors must be the actual distributions after the declared temperature, truncation, and other logits transformations. Keep repetition/history-dependent processors out of the first implementation. Add one only after you can evaluate it consistently for each hypothetical prefix.

Add a separate greedy verifier that accepts the longest draft prefix matching target argmax decisions. Greedy matching and stochastic rejection sampling have different correctness checks.

Then implement target block verification with correct shifted-logit indexing. Retain a clear convention for the last emitted token that has not yet been processed into the cache. This “pending token” convention determines whether the next target call consumes k or k+1 input positions.

### Cache reconciliation is part of the algorithm

The draft and target have different layer counts and hidden dimensions; their KV tensors are separate and cannot be reused interchangeably.

Write down each cache length before drafting, after target verification, and after commitment. Rejected tentative positions must become unreachable. Accepted outputs plus any correction/bonus token must be processed as needed before the next cycle.

Use logical truncation for contiguous caches and a correct release/ownership procedure for paged caches. Add shared-prefix or copy-on-write interactions only after the unshared case passes. Truncating a length counter is insufficient if later reads can still address rejected entries.

A concrete adversarial fixture: draft four tokens, accept two, reject the third, emit a correction, and continue for several cycles. Compare the final logits with fresh recomputation at the committed history.

### Week 10: investigate break-even behavior

Use real prompts in three domains: prose, code, and structured extraction. Start with 30 prompts per domain, then expand only where variation prevents a conclusion.

Sweep k = 1, 2, 4, and 8 at batch 1. Record:

- Draft time, including any catch-up processing.
- Target verification time.
- Commit/rollback/sampling overhead.
- Accepted-prefix length distribution and emitted target tokens per cycle.
- Target-only decode time at comparable contexts.
- Extra memory, TTFT, total completion latency, and client-visible token gaps.

Include the draft’s prompt-prefill cost in full-request latency. Present steady-state decode speedup separately.

### The performance model

Let \(A\) be the number of committed output tokens in a cycle, \(t_D(k)\) the complete draft-side time, \(t_V(k)\) target verification time, \(t_O\) other cycle overhead, and \(t_T\) a comparable target-only one-token step:

\[
S_{\mathrm{decode}}\approx
\frac{\mathbb E[A]t_T}{t_D(k)+t_V(k)+t_O}.
\]

Estimate \(\mathbb E[A]\) directly from the accepted-prefix distribution. As a simplified exercise only, with constant independent acceptance probability \(\alpha\), no stopping boundary, and one correction/bonus token:

\[
\mathbb E[A]=1+\alpha+\cdots+\alpha^k.
\]

For an invented worked example with \(k=4\), draft-step cost \(0.25t_T\), verification \(1.2t_T\), and other overhead \(0.05t_T\), predicted speedup is about 1.49× at \(\alpha=0.8\), but 0.86× at \(\alpha=0.5\). These values illustrate the equation; they are not measurements of the Qwen pair.

Cache state and acceptance are history dependent, so the real distribution is more informative than one aggregate acceptance rate.

### Correctness and completion criterion

Use the finite-vocabulary oracle to test empirical output distributions against the target, with a declared sampling-error bound. Test multiple contexts and rejection locations. For real-model greedy tests, compare to target-only generation under the same numerical path, investigating ties and numerical changes.

An exact speculative sampler preserves the target distribution in the mathematical model; using the same random seed does not require the same sampled token sequence because RNG consumption changes.

Finish with a recommendation to enable or disable this pair for a stated workload and k. The course succeeds if the pair loses and you can explain why.

**Chapter draft:** “When an 8B draft can—or cannot—accelerate a 32B target.”

**Oral defense:** why can higher draft accuracy still lead to lower speedup?

**Extension:** after completing the required pair, compare a cheaper n-gram draft or an additional small Qwen model. This is optional and should answer a failure mode discovered in the main experiment.

---

## 11. Project 6 — Quantization as a measured quality/performance choice

**Weeks 11–12 · Spark for numerics; Spark or H100 for verified packed execution**

**Research question:** under what workload does lower precision improve useful serving capacity, and what changes in quality?

**Default route:** groupwise INT4 weight-only inference with BF16 activations, commonly described as W4A16; study AWQ as the main calibrated method. Implement numerical foundations yourself, then use a maintained tool to produce an actual compatible checkpoint.

**Read first:** MIT 6.5940’s quantization lectures, the [AWQ paper](https://arxiv.org/abs/2306.00978), and the selected backend’s documentation. Use [LLM Compressor](https://github.com/vllm-project/llm-compressor) as a tooling reference and [SGLang’s quantization guide](https://docs.sglang.io/docs/advanced_features/quantization) to verify the load format and execution path. [MIT course material](https://hanlab.mit.edu/courses/2024-fall-65940)

### Week 11: separate numerical representation from efficient execution

Implement per-tensor, per-channel, and groupwise quantize/dequantize references. Cover scale choice, zero point where applicable, clipping, rounding, signed ranges, and packing two 4-bit values into a byte.

For each group g, a simple affine reference is:

\[
q=\operatorname{clip}(\operatorname{round}(x/s_g)+z_g,q_{\min},q_{\max}),
\qquad \hat{x}=s_g(q-z_g).
\]

Compare group sizes 32 and 128 on representative attention and MLP weights. Measure weight error and layer-output error on real activations. Explain why a small weight error does not uniquely determine model-output error.

For a numerical experiment, dequantizing to BF16 before a matrix multiply is acceptable. For a serving speed claim, inspect the real packed-weight execution path and account for unpacking, conversion, scaling, and repacking. A smaller file or a fake-quantized graph does not establish faster inference.

Produce one actual calibrated 8B checkpoint, then apply the same recipe to 32B. Use approximately 128 disjoint calibration sequences of up to 512 tokens as a starting budget. Record their provenance and prompt formatting; do not calibrate on evaluation examples.

Time-box backend setup. If the selected W4A16 path is unsupported on GB10 or the pinned Arm environment, run the packed-kernel comparison on the prepared H100 environment. A supported FP8 path is a reasonable alternative main format if W4 setup remains blocked; document the change and study its specific scaling and compute path.

### Week 12: evaluate quality and serving behavior

Choose one fixed teacher-forced text corpus and one task with an objective score. Use a local, pinned evaluation harness and fixed prompt templates. Approximately 512 scored task examples is a starting budget; a noisy or borderline result requires a wider interval or more data.

For a compact course evaluation, use fixed held-out text for mean negative log-likelihood, a pinned GSM8K subset for task accuracy, and a small structured-extraction fixture for format correctness. GSM8K is only one task, and its results should not be described as general model quality. Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the [GSM8K source](https://github.com/openai/grade-school-math) as references.

Before running, choose a quality budget. An example course target is at most 1% relative increase in mean negative log-likelihood and at most two percentage points of task-accuracy loss. These are proposed engineering thresholds, not standards. Use paired examples and uncertainty estimates. A 512-example test may not resolve a two-point difference; label that result inconclusive.

Compare BF16 and the selected quantized format within each model at:

1. The same batch and context: execution cost.
2. The largest feasible batch under the same latency conditions: capacity.
3. The same offered load and fixed SLOs: serving goodput.

Keep cache dtype fixed for this comparison. Weight quantization does not automatically reduce KV storage. A 32B W4 checkpoint still has 2 GiB of BF16 KV at 8K tokens per request unless KV precision changes.

Report serialized bytes, loaded packed-weight bytes, reserved memory, peak temporary memory, TTFT, completion latency, goodput, and quality. Mixed precision, scales, padding, and workspaces make real storage larger than the ideal parameter-count × 4-bit calculation.

### Required interaction with speculation

Run a small three-condition comparison:

| Condition | Target | Draft | Purpose |
|---|---|---|---|
| Baseline | BF16 32B | BF16 8B | Reuse Project 5 |
| Cheaper draft | BF16 32B | Quantized 8B | Does lower draft cost compensate for changed acceptance? |
| Quantized target | Quantized 32B | Best available 8B draft | Does target acceleration reduce the remaining opportunity for speculation? |

For the second condition, a correct verifier still targets the BF16 32B distribution when it uses the quantized draft’s actual probabilities. In the third, the verifier targets the quantized 32B distribution; losslessness does not restore the original BF16 model.

Use a 2×2 subset—BF16/quantized target and speculation off/on—if it makes the interaction easier to interpret. Do not assume separately measured speedups multiply.

### Completion criterion

One real quantized path is evaluated on both models, its actual execution is identified, and a limited workload-specific recommendation is supported by quality and performance evidence.

**Chapter draft:** “Quantization: from packed bits to a serving decision.”

**Oral defense:** why can reducing weight bytes improve maximum concurrency but barely change low-load TTFT?

**Extensions:** FP8 weights/activations; [GPTQ](https://arxiv.org/abs/2210.17323); [SmoothQuant](https://arxiv.org/abs/2211.10438); or a KV-cache precision experiment inspired by [KIVI](https://arxiv.org/abs/2402.02750). Complete one primary format before choosing an extension.

---

## 12. Project 7 — Implement tensor parallelism and predict scaling

**Weeks 13–14 · Two connected H100 GPUs; bounded cross-node extension**

**Research question:** when should two GPUs serve one request together, and when should they serve independent requests?

**Read first:** the tensor-parallel construction in [Megatron-LM](https://arxiv.org/abs/1909.08053), a concrete implementation such as [TorchTitan’s tensor-parallel source](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/tensor_parallel.py), and [NCCL tests](https://github.com/NVIDIA/nccl-tests). The selected source includes training machinery; extract the forward-pass placement and communication relevant to inference.

### Week 13: shard a layer, then the decoder

Start with one layer from the real Qwen3-8B checkpoint and deterministic inputs. In PyTorch’s stored weight layout, distinguish splitting output features from splitting input features.

Implement:

- Q/K/V projections sharded by complete heads.
- Local attention on each rank’s query and KV heads.
- Attention output projection that produces partial residual-width outputs, followed by a sum.
- Gate/up projections sharded along the intermediate width.
- Local SwiGLU and a down projection that produces partial residual-width outputs, followed by a sum.

Use `torch.distributed` and NCCL for communication. Replicate the embedding and vocabulary head initially to simplify correctness; include their replicated memory in the capacity model. Shard them only as an extension.

At TP=2, both models’ eight KV heads divide cleanly. Each rank can own four KV heads and half the KV cache. Scaling beyond the number of KV heads requires another placement choice; cache memory does not divide indefinitely with TP degree.

Validate a single 8B layer, then full 8B prefill/decode against TP=1. Finally validate selected 32B inputs. Document reduction-order differences and run the established numerical checks.

### Week 14: compare two uses of the same hardware

For each model, compare:

1. One TP=1 instance on one GPU: a scaling reference.
2. One TP=2 instance on two GPUs.
3. Two independent TP=1 replicas with a declared routing policy.

The third comparison is valid only at workloads that fit each replica. If TP=2 enables a context/batch that TP=1 cannot fit, report that separately as a capacity advantage.

Use the same two GPUs, precision, prompt trace, SLOs, and useful output accounting for TP-versus-replica decisions. Fix or explicitly sweep the per-instance batch limit. Report low-load latency and maximum goodput as separate outcomes.

Measure collective sizes in the actual traces. In the simple construction above, there are two residual-width reductions per decoder layer: 72 for 8B and 128 for 32B per forward step, before other implementation-specific communication.

### A distributed performance model

For n ranks and an m-byte all-reduce, a ring approximation is:

\[
t_{\mathrm{AR}}(m,n)\approx
2(n-1)\alpha+\frac{2(n-1)}{n}\frac{m}{B_{\mathrm{link}}}.
\]

Here \(\alpha\) is a per-phase latency term and \(B_{\mathrm{link}}\) an effective bandwidth. Fit them from measurements in the relevant size range. NCCL may select other algorithms; treat the ring equation as a starting model and compare it with observed behavior.

For a BF16 residual tensor, a common reduction payload is \(m=BD\times2\) bytes at decode and \(m=MD\times2\) bytes for a prefill chunk of M tokens. For 32B at batch 1, this is only 10 KiB per reduction. Measure small collectives; a large-message bandwidth result does not explain token-by-token latency.

Build the step model from local kernel times plus communication on the critical path. Use Nsight to determine whether overlap occurs; do not add all communication if it overlaps, or omit it because an operation was launched asynchronously.

Predict TP=2 latency from the single-GPU/component model before seeing full TP results. Report scaling efficiency \(t_1/(nt_n)\) for matched work, while keeping throughput and capacity comparisons separate.

### Cross-node experiment

Repeat a limited TP=2 comparison across two real nodes: one small-batch decode case and one prefill case. Record NICs, transport, topology, message sizes, and measured bandwidth/latency. Repeat the collective microbenchmarks on that path.

Same-node and cross-node results are separate claims. A throttled local link or simulated network can support a sensitivity analysis but cannot substitute for a real network measurement.

### Completion criterion

Your TP layer/model is correct, your scaling prediction is compared with measurements, and you can choose TP=2 or two replicas for a specific workload.

**Chapter draft:** “One model on two GPUs, or two copies on one GPU each?”

**Oral defense:** why can TP reduce memory pressure yet worsen a short request’s latency?

**Extensions:** TP=4, vocabulary parallelism, or pipeline parallelism. Pipeline parallelism introduces a distinct scheduling/bubble question and should have its own experiment.

---

## 13. Project 8 — Transfer KV state and evaluate prefill/decode separation

**Weeks 15–16 · Two GPUs; use the topology already characterized**

**Research question:** when does separating prefill from decode improve goodput enough to justify state transfer and resource partitioning?

**Read first:** [DistServe](https://arxiv.org/abs/2401.09670) and [Dynamo’s disaggregated-serving design](https://docs.dynamo.nvidia.com/dynamo/v-0-9-0/design-docs/disaggregated-serving). The latter is a versioned reference; pin the deployed code and use its matching recipe.

### Week 15: build a minimal, correct handoff

Implement one prefill worker and one decode worker using the same model revision, precision, and cache layout. Begin with a synchronous transfer and TP=1 on each side.

The exported state must specify model/format identity, request token history, processed cache length, absolute positions, K/V tensors or physical-block payloads, and any pending sampled token. Include sufficient sampler state for the chosen stochastic protocol. Do not transfer raw pointers or source-local block IDs as if they were meaningful on the destination.

An explicit convention can be:

1. Prefill processes S prompt tokens and obtains logits.
2. A first token is sampled; caches still represent the S processed prompt tokens.
3. Transfer those caches, position/history metadata, and the pending token.
4. Decode processes the pending token, appends its KV entries, and produces the next logits.

Decide which worker emits the first token and measure the resulting first-to-second-token gap. A quick first token followed by a long handoff pause can look good in TTFT while giving poor streaming behavior.

Allocate destination blocks, copy the payload, acknowledge readiness, and release the source state only after completion. Count the temporary duplication of KV in the memory model.

Validate continuation against colocated generation. Include multiple prompt lengths, page-boundary lengths, EOS, and cancellation during transfer. Limit the first protocol to equal TP degree and layout; changing either requires a reshard/repack step.

### Transfer modeling

For a BF16 32B request at 8,192 cached tokens, the logical KV payload is 2 GiB. At an invented 200 Gbit/s payload link, serialization alone is:

\[
t_{\mathrm{wire}}=\frac{2\times2^{30}}{200\times10^9/8}
\approx 85.9\ \mathrm{ms}.
\]

This is a sensitivity example, not a measured property of your rental. Actual time also includes transport latency, packing, staging, registration, coordination, contention, and any layout conversion.

Fit:

\[
t_{\mathrm{handoff}}\approx
t_{\mathrm{setup}}+M_{KV}/B_{\mathrm{effective}}
+t_{\mathrm{pack}}+t_{\mathrm{coord}}.
\]

Measure transfer in isolation and under serving load. Compute bytes transferred per request and per second. Lower-precision weights do not reduce the transfer unless KV precision or the transferred representation also changes.

### Week 16: compare complete serving systems

With the same two GPUs, compare two colocated replicas against one prefill GPU plus one decode GPU. Also compare the best colocated TP arrangement discovered in Project 7 when feasible.

Use W1–W4 and a fixed mixture of W1 and W3. Start at low load, then increase offered load using identical traces. Sweep the prefill/decode allocation only if you have additional GPUs; a 1P/1D experiment cannot establish the best ratio at large scale.

Report queueing at each stage, utilization, transferred bytes, completion latency, TTFT, token-gap distributions, and goodput. Include the underutilized-stage case. A design can reduce interference and still lose because the split strands capacity.

A simple queueing bound is \(\lambda<\min(\mu_P,\mu_D)\) for stable stage capacities under a fixed workload mix; those capacities depend on batching and sequence lengths. Measure the service curves rather than assuming fixed request costs.

Your runtime’s transfer prototype establishes mechanism and correctness. If time remains, run one matching Dynamo/SGLang deployment and trace the transfer path to understand the production implementation. Successful deployment alone does not replace the experiment.

### Final decision report

Write a deployment recommendation for each model under one declared workload and hardware budget. State:

- The selected precision, batching policy, and cache strategy.
- Whether speculation is enabled and its draft length.
- Whether the hardware runs TP or replicas.
- Whether prefill/decode separation is justified.
- The measured latency, goodput, quality, and memory constraints.
- The workload change most likely to invalidate the recommendation.

Use your own measurements and label projections. Keep production guarantees outside the scope of a semester prototype.

### Completion criterion

The KV handoff preserves continuation, the transfer model is compared with measurements, and the final recommendation accounts for the full hardware allocation.

**Chapter draft:** “When separating prefill and decode is worth moving the cache.”

**Oral defense:** which measurement would convince you to turn disaggregation off?

**Extensions:** transfer/computation overlap, NIXL transport, KV quantization in transit, unequal TP degrees, or an additional network topology. Each changes the model and needs a separate comparison.

---

## 14. Reading guide and existing courses

Closely related courses already exist. Use selected portions as companions to this project sequence; completing all their assignments would exceed the semester.

| Resource | Assigned use | Material to defer |
|---|---|---|
| [CMU 11-868 / LLM Systems, Spring 2025 syllabus](https://llmsystem.github.io/llmsystem2025spring/docs/Syllabus/) | Serving, quantization, attention, speculation, and P/D lectures | Broad framework/training topics unless a project needs them |
| [Stanford CS336, Spring 2026](https://cs336.stanford.edu/) | Architecture/resource accounting; selected Basics and Systems assignment exercises | Full tokenizer, pretraining, data pipeline, and alignment assignments |
| [MIT 6.5940, Fall 2024](https://hanlab.mit.edu/courses/2024-fall-65940) | Quantization lectures and related exercises | NAS, pruning, and unrelated edge-model labs |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Rooflines, transformer math, inference and sharding derivations | Training-scale examples that do not answer your inference question |

This course’s particular structure is an independent design: the same two checkpoints, one evolving runtime, predictions across model sizes, and an explicit chapter-writing process.

### What to read in each project

Read one conceptual source and one relevant code path closely. Scan the other references only to answer a question discovered during implementation.

| Project | Essential conceptual reading | Code/documentation focus | Reading question |
|---|---|---|---|
| P1 | Qwen3 architecture; CS336 architecture/resource accounting | Qwen configs and attention/MLP/cache implementation | Which dimensions and state are actually different between 8B and 32B? |
| P2 | Orca scheduling; PagedAttention allocation | Mini-SGLang scheduler/cache; SGLang cache section if needed | Who owns each block at every request transition? |
| P3 | Scaling Book rooflines and inference | Nsight timelines; SGLang benchmark definitions | Which term can explain the observed curve? |
| P4 | FlashAttention online softmax; FA2 work division | One CuTe GEMM and one attention implementation for your GPU | Where are values reused and where is work exposed? |
| P5 | One full speculative-sampling algorithm/proof; compare the second paper | SGLang STANDALONE verifier/cache path | Which distribution and cache prefix does each position represent? |
| P6 | MIT quantization; AWQ method and limitations | One quantizer and its packed-weight execution backend | Which error is introduced, and where is execution overhead paid? |
| P7 | Megatron tensor-parallel construction | TorchTitan placement; NCCL tests and your traces | Which dimensions stay local and which outputs require communication? |
| P8 | DistServe architecture/evaluation | Dynamo handoff sequence; your transfer implementation | Which interference is removed, and which cost is introduced? |

All referenced web pages were checked while preparing this plan on September 11, 2026. Repository and documentation URLs can evolve. Resolve model revisions, source commits, and container digests when beginning each project. A source reference verifies the reading target, not compatibility or performance on your machine.

### Optional architecture continuation

After the dense-model capstone, take a separate 4–6 week comparative module:

| Topic | Bounded implementation | Systems question |
|---|---|---|
| MLA | Small random-weight expanded-K/V and compressed-cache paths | How does compressed state exchange cache traffic for computation? |
| MoE | Routing, token grouping, expert evaluation, weighted combination | How do imbalance and small expert batches affect execution? |
| Expert parallelism | Two-rank token dispatch/combine on the small MoE | When does all-to-all communication dominate useful expert work? |

These are mathematical and systems exercises. Random-weight modules do not establish model quality. Choose an actual MLA/MoE checkpoint only when starting that module and verify its architecture then.

---

## 15. Turn the work into a textbook or blog series

Organize the student engine repository by mechanism and experiment. The following
are suggested implementation paths, not a claim that a complete engine is supplied.
This course directory now provides a conceptual Chapter 0 and eight project chapters under `chapters/`
and common setup, protocol, and report materials under `shared/`; keep your evolving
engine and experiment artifacts alongside them or in your existing engine repository.

| Path | Purpose |
|---|---|
| `model/` | Configuration, layers, weight loading |
| `runtime/` | Scheduler, requests, cache allocator, model runner |
| `kernels/` | CuTe implementations and backend adapters |
| `speculation/` | Verification and cache reconciliation |
| `quantization/` | Numerical references and conversion recipes |
| `distributed/` | TP layers and KV transfer |
| `experiments/` | Immutable workload/config manifests and run scripts |
| `analysis/` | Performance models and plot generation |
| `tests/` | Mechanism-level correctness and numerical comparisons |
| `chapters/` | Research memos and later edited chapters |

Do not commit large weights or private data. Keep source attribution for reused Mini-SGLang or other code and make your changes identifiable.

### Project chapter template (Chapters 1–8)

1. **Question and scope.** Specify model, workload, hardware, and the decision being investigated.
2. **Mechanism.** Explain the relevant computation or state transition, with tensor shapes.
3. **Prediction.** Give the equation or hypothesis written before measurement.
4. **Implementation.** Show only the code needed to understand the mechanism.
5. **Correctness.** Explain the oracle, comparisons, tolerances, and boundary cases.
6. **Experiment.** Provide a compact reproducibility manifest.
7. **Results.** Include the decisive plots and uncertainty/sample counts.
8. **Explanation.** Connect profiler evidence with the prediction.
9. **Limits and failed hypotheses.** Preserve results that did not support the proposed optimization.
10. **Decision and next experiment.** State when the result is useful and what might overturn it.

Label plots as measured, modeled, or illustrative. A useful chapter usually needs two or three figures: a prediction-versus-measurement plot, a tradeoff curve, and possibly one annotated trace. Avoid presenting a broad benchmark grid without an explanation.

### Suggested chapter sequence

| Chapter | Working title | Main figure |
|---|---|---|
| 0 | From text to a serving system (conceptual reading) | Transformer and request-flow diagrams; no project |
| 1 | Reconstructing Qwen3 | Predicted versus measured memory by context |
| 2 | An inference scheduler from first principles | Goodput and token latency under mixed arrivals |
| 3 | Predicting inference latency | Predicted versus observed prefill/decode times |
| 4 | From a kernel optimization to model speed | Kernel speedup versus integrated speedup |
| 5 | Speculating with an 8B draft | Speedup versus accepted tokens per cycle |
| 6 | Quantization as a serving choice | Quality versus memory/goodput |
| 7 | Tensor parallelism or replicas | Goodput under fixed latency conditions |
| 8 | The price of moving KV | P/D benefit versus transfer and load |

### Assessment rubric

| Dimension | Weight | Evidence |
|---|---:|---|
| Correctness | 30% | Numerical and state invariants; reproducible oracle |
| Explanation and prediction | 25% | Derived model and withheld-condition predictions |
| Experimental method | 20% | Controlled variables, uncertainty, complete resource accounting |
| Implementation understanding | 15% | Readable mechanism and oral reconstruction |
| Writing and reproducibility | 10% | A reader can follow the argument and rerun it |

A well-explained negative result can receive full credit. A large speedup with a broken verifier, changed workload, or hidden extra GPU cannot.

At the end, give yourself a 30-minute technical defense: trace a request, estimate its memory, explain one failed optimization, and choose a deployment configuration for a changed workload. Use project evidence to support what you claim in interviews. Distinguish personal experiments from production systems you operated at work.

---

## 16. Your first five study sessions

Each session is approximately 2–3 hours. Spread the remaining Week 1 time across reading and debugging.

| Session | Action | Concrete output |
|---|---|---|
| 1 | Pin both model/config/tokenizer revisions and the working Spark environment; run 8B reference prefill/decode | Environment manifest, saved token fixture, first timing |
| 2 | Audit dimensions and checkpoint metadata; derive parameter/KV sizes | Shape inventory and memory calculation |
| 3 | Implement RMSNorm, SwiGLU, and the Q/K normalization path on a tiny FP32 model | Layer comparisons and first-divergence diagnostics |
| 4 | Implement RoPE, GQA, and offset causal masking with explicit cache lengths | Full-versus-chunked and cached-versus-recomputed checks |
| 5 | Assemble the tiny decoder, start 8B weight mapping, and write the first hypothesis | Mapping report and a one-page experiment note |

**First hypothesis:** at batch 1, cached decoding removes repeated prefix computation, but total step time eventually rises with context because the attention kernel reads more KV state.

**First milestone:** a correct 8B attention block and a reproducible baseline. Validate the complete 8B model and a short 32B continuation in Week 2.

Keep the semester’s question in view: can you explain why the system behaves this way, and can your explanation predict what happens next?
