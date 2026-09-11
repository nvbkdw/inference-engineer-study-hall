# Tutorial: reconstruct, validate, then measure

## 1. Freeze the reference (2 hours)

Use [the executable checkpoint workflow](checkpoint_workflow.md) for the concrete
snapshot audit, sequential reference/custom runs, and saved-logit comparison.
Run its tiny sharded checkpoint first, then apply the same commands to the pinned
real snapshots. The remaining steps below are the full project requirements.

Complete [setup](../../shared/SETUP.md), resolve both checkpoint/tokenizer revisions,
and save one rendered real prompt's IDs. Start a manifest before loading weights.
Use one engine backend and one attention implementation for the initial baseline.
Run the reference with greedy, non-thinking generation. Save selected prefill and
continuation logits on CPU. Keep the checkpoint and Transformers code revisions
separate: model revision does not pin implementation code.

**Checkpoint:** a reference artifact includes token IDs, positions, logits, dtype,
backend, seed, and software identity. No performance claim is needed yet.

## 2. Audit dimensions and memory (3 hours)

```bash
python chapters/01_reconstruct_qwen3/code/lab.py --config models/qwen3-8b/config.json
python chapters/01_reconstruct_qwen3/code/lab.py --config models/qwen3-32b/config.json
```

Expect parameter counts 8,190,735,360 and 32,762,123,264. Derive them independently
from the background equation. List Q/K/V/O, gate/up/down, normalization, embedding,
and head shapes. Inspect safetensors metadata to count real checkpoint elements;
the config calculation alone is not a weight audit.

For every tensor, create a CSV row `checkpoint_name,engine_name,shape,numel,status`.
Add BF16 weights, KV at 128/512/2048 tokens, anticipated loading copies, and workspace
headroom to a memory budget. Predict the slope of memory versus context.

## 3. Reconstruct the tiny math (5 hours)

```bash
python chapters/01_reconstruct_qwen3/code/lab.py
```

Read `RMSNorm`, `rope`, and `Block.forward`. The sample runs random weights on CPU,
with ordinary PyTorch operations, concatenated caches, and repeated KV heads.
It verifies full versus incremental and unequal-chunk logits. It is a correctness
oracle; allocation and dense attention make its timings unsuitable as a fast baseline.

Reimplement RMSNorm and SwiGLU in your engine without copying the sample. Compare
outputs on fixed tensors. Next add head reshaping, Q/K norm, RoPE, and attention.
Hand-write the p=3,t=2 mask before implementing it. Compare Q/K before and after
RoPE, attention output, and residual output to isolate failures.

**Exercise:** intentionally derive head_dim from hidden_size and observe which
fixture catches the error. Restore correctness before proceeding.

## 4. Load real weights (5 hours)

Use the mapping CSV to implement an explicit name mapping. `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` are distinct from Q/K
normalization weights. Validate shapes before copying; reject missing or unexpected
trainable weights. Consume shards sequentially and discard loading buffers promptly.

The sample's names are pedagogical and do not load a Hugging Face state dict
directly. Implement the mapping in your engine. Require exact coverage rather than
`strict=False`. Compare one 8B block first, then the stack and logits. Run reference
and custom 32B sequentially using saved CPU artifacts to stay within memory.

## 5. Make cache behavior independent of batching (5 hours)

Add request-specific processed lengths and cache storage. Validate full prefill,
chunks `[3,1,7]`, eleven one-token calls, and a continuation after prefill. Then run
the same request alone and in a mixed-length batch. The sample uses equal-length
batches; mixed-length metadata/masking is your required extension.

Test empty initial cache, a one-token prompt, a chunk ending exactly at capacity,
and an attempted context overflow. Include prompts whose lengths differ by one.
Compare selected complete 8B and short 32B outputs to the saved reference.

**Checkpoint:** one config-driven implementation handles both models without
special-case projection dimensions. Every cached read is below processed length.

## 6. Measure the cache hypothesis (6 hours)

Begin the measurement workflow on DGX Spark using
[the standalone CUDA experiment](standalone.md). Its `code/experiment.py`
uses CUDA events and records the actual GPU identity. CPU `lab.py` checks
are optional correctness preparation; they supply no timing baseline.
The remaining steps below extend the reference workload to the full project.

At S=128,512,2048 and B=1, generate a fixed number of new tokens using cached
decode and repeated full-prefix computation. Predict which terms disappear and
which grow with S. Time uninstrumented runs after warmup; profile a separate run.
Record prompt processing, per-step decode, parameter storage, peak loading memory,
live memory, and reserved memory. Do not include tokenization in a kernel time.

Use `sum(t.numel()*t.element_size())` for logical tensor bytes and allocator tools
for live/reserved device memory. On unified memory, also record host process and
system memory. Explain differences instead of equating allocator reservation with
live KV. Repeat a short 32B continuation and representative memory points.

## 7. Decide and submit (4 hours)

Plot modeled and measured memory versus cached tokens, and cached versus
recomputed decode time. Explain the intercept and slope. Propose one optimization
supported by evidence, such as avoiding cache concatenation or redundant head
copies, then measure a bounded change.

Submit the mapping CSV, numerical comparison report, both model validation
artifacts, memory inventory, and [research memo](../../shared/REPORT.md).
Completion requires real-checkpoint validation as well as the tiny checks.

**Defense:** Why is 32B's query stream wider than its residual stream? Why does
its cache not grow fourfold relative to 8B? What would cause plausible text but
incorrect chunked-prefill logits?
