# References for the introduction

These readings supply architectural definitions and primary descriptions of
serving mechanisms. They are optional deeper reading, with no implementation or
project attached. Documentation links were checked on September 11, 2026;
software capabilities can change. Published benchmark numbers belong to their
original hardware and workloads.

## Architecture and mathematical intuition

| Reading | What to focus on |
|---|---|
| [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) | Section 3's attention equations and encoder/decoder distinction; the original block differs from modern Qwen3 |
| [Hugging Face: Tokenization algorithms](https://huggingface.co/docs/transformers/main/tokenizer_summary) | Subword and byte-level representations; the tradeoff between vocabulary and sequence length |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | Scale normalization and the learned channel multiplier |
| [RoFormer](https://arxiv.org/html/2104.09864v5) | Rotation matrices and the relative-position identity in attention |
| [GQA](https://arxiv.org/abs/2305.13245) | The continuum from multi-head to shared key/value heads |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | Multiplicative gating and the SwiGLU definition |
| [Qwen3 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3) | Terminology for the course's concrete architecture |
| [Qwen3-8B configuration](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) and [Qwen3-32B configuration](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json) | Actual checkpoint dimensions; the Chapter 1 symbol table translates the fields |
| [Generation parameters](https://huggingface.co/docs/transformers/main/main_classes/text_generation) | Temperature, top-k, top-p, and stopping conditions |

## Inference and serving mechanisms

| Reading | What to focus on |
|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) | The distinction between a logical attention matrix and the data actually written to memory |
| [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) | Logical token positions, physical blocks, fragmentation, and sharing |
| [vLLM documentation](https://docs.vllm.ai/en/latest/) | How the serving engine brings execution and memory management together |
| [SGLang paper](https://arxiv.org/html/2312.07104v2) and [documentation](https://docs.sglang.io/) | RadixAttention and prefix-state reuse within a serving runtime |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | Draft proposals, target verification, and distribution-preserving correction |

## Distribution and performance

| Reading | What to focus on |
|---|---|
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | Forward-pass matrix partitioning and where partial results must combine |
| [DistServe](https://arxiv.org/html/2401.09670v2) | Prefill/decode interference, separate worker allocation, and goodput objectives |
| [SGLang P/D disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) | Prefill and decode roles and the existence of an explicit transfer path; deployment commands are beyond Chapter 0 |
| [Berkeley Lab: Roofline intuition](https://cs-newsarchive.lbl.gov/news/2017/roofline-model-boosts-manycore-code-optimization-efforts/) | Arithmetic intensity and competing compute/memory limits |
| [Roofline publications](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/ppan-roofline-publications/) | The original Williams–Waterman–Patterson model and later extensions |
| [MIT: Queueing models](https://web.mit.edu/1.041/spring2023/lectures/L8-queuing-models-2023sp.pdf) | Little's law, consistent boundaries, and steady-state assumptions |

[Chapter guide](README.md) · [Next chapter](../01_reconstruct_qwen3/background.md)
