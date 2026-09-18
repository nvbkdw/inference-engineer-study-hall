# Reading guide

| Priority | Primary source | Assigned use |
|---|---|---|
| Foundation | [MIT 6.5940, Fall 2024](https://hanlab.mit.edu/courses/2024-fall-65940) | Quantization lectures/labs: scale, clipping, and representation |
| Required | [AWQ paper](https://arxiv.org/abs/2306.00978) | Explain why activation information changes useful weight treatment |
| Conversion | [LLM Compressor](https://github.com/vllm-project/llm-compressor) | Select a release-matched calibration recipe; inspect saved format metadata |
| Execution | [SGLang quantization guide](https://docs.sglang.io/docs/advanced_features/quantization) | Verify actual format/backend/hardware support |
| Evaluation | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Freeze local task configuration and example-level outputs |
| Dataset | [GSM8K source](https://github.com/openai/grade-school-math) | Inspect task structure, splits, and answer format |

Opened during preparation on 2026-09-11. Backend support is release specific.
Reading exercise: identify every byte of metadata needed to reconstruct a weight
group, then identify the kernel that consumes those bytes during inference.
