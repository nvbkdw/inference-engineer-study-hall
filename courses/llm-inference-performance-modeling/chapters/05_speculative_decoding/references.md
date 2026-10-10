# Reading guide

| Priority | Primary source | Assigned use |
|---|---|---|
| Required | [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) | Read the complete algorithm and distribution-preservation argument |
| Comparison | [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | Map its notation to p, q, accepted prefix, and correction |
| Implementation | [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding) | Locate the standalone draft-model path and match its version/configuration |

Opened during preparation on 2026-09-11. Derive one rejection example by hand.
Paper speedups are results of their workloads and hardware; they are not expected
results for Qwen3-8B drafting Qwen3-32B.
