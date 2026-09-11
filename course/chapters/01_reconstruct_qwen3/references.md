# Reading guide

Read the configurations and one forward path closely; use the others to resolve
specific questions. Source links were opened during preparation on 2026-09-11;
moving branches are reading targets, so pin commits for experiments.

| Priority | Primary source | What to extract |
|---|---|---|
| Required | [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) and [32B config](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json) | Build the dimension and parameter table; inspect explicit head_dim |
| Required | [Transformers Qwen3 source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | Trace Q/K norm, rotary ordering, residuals, and the causal attention path |
| Companion | [Stanford CS336](https://cs336.stanford.edu/) | Architecture/resource-accounting lectures and Basics exercises |
| Tooling | [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) | Pin snapshots and avoid downloading multiple changing revisions |

Reading exercise: annotate every line of the attention forward pass with its
logical shape. Which values can be retained across tokens, and why?
