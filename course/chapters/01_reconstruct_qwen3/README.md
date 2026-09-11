# Chapter 1: Reconstructing Qwen3

**P1 · Weeks 1–2 · 30–36 hours**  
**Laboratory:** DGX Spark measurements; optional CPU mathematical oracle; 8B daily and 32B validation

Can one configuration-driven implementation reproduce both checkpoints and their cached generation?

## Learning outcomes

- Derive projection shapes, parameter count, and KV bytes from configuration.
- Implement Q/K normalization, RoPE, GQA, SwiGLU, and correct offset masks.
- Validate full, incremental, chunked, and mixed-batch behavior against a reference.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

Tensor algebra, PyTorch, stable softmax. Complete the shared setup first. Follow the [shared setup](../../shared/SETUP.md) and
[experiment protocol](../../shared/PROTOCOL.md). Read the background, work through
the tutorial in order, and consult the annotated references at each milestone.

| Material | Purpose |
|---|---|
| [Background](background.md) | Definitions, first-principles derivations, and failure modes |
| [Step-by-step tutorial](tutorial.md) | Commands, implementation sequence, experiments, and checkpoints |
| [Code sample](code/lab.py) | Small runnable reference; scope and limitations are documented in the source |
| [Standalone experiment](code/experiment.py) | Reproducible sweep with predictions, raw CSV, summaries, and figures |
| [Checkpoint workflow](checkpoint_workflow.md) | Strict shard loading, offline HF oracle, and real-snapshot command sequence |
| [Measurable goals](assessment.md) | Acceptance conditions and required evidence for each milestone |
| [Online references](references.md) | Assigned readings, code paths, and reading questions |
| [Report and rubric](../../shared/REPORT.md) | Submission structure and common assessment criteria |

## Required evidence

Parameter/weight mapping; real 8B and short 32B validation; predicted versus measured memory.

- [ ] Every checkpoint tensor is mapped and shape-checked.
- [ ] Tiny FP32 and real-checkpoint comparisons pass documented criteria.
- [ ] Memory intercept/slope and cached-versus-recomputed timing are explained.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Course home](../../README.md) · [Next chapter](../02_runtime_and_kv/README.md)
