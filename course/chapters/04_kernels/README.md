# Chapter 4: From a kernel optimization to model speed

**P4 · Weeks 7–8 · 30–36 hours**  
**Laboratory:** DGX Spark attention measurements; optional CPU oracle; H100 for Hopper study

Does a local kernel improvement change useful model performance?

## Learning outcomes

- Reason about GEMM reuse, resource use, and one bounded design change.
- Derive online softmax and implement a bounded CuTe GQA attention path.
- Connect kernel, adapter, and model speed through Amdahl's law.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

P3 bottleneck evidence plus basic CUDA/CuTe GEMM experience. Follow the [shared setup](../../shared/SETUP.md) and
[experiment protocol](../../shared/PROTOCOL.md). Read the background, work through
the tutorial in order, and consult the annotated references at each milestone.

| Material | Purpose |
|---|---|
| [Background](background.md) | Definitions, first-principles derivations, and failure modes |
| [Step-by-step tutorial](tutorial.md) | Commands, implementation sequence, experiments, and checkpoints |
| [Code sample](code/lab.py) | Small runnable reference; scope and limitations are documented in the source |
| [Standalone experiment](code/experiment.py) | Reproducible sweep with predictions, raw CSV, summaries, and figures |
| [Measurable goals](assessment.md) | Acceptance conditions and required evidence for each milestone |
| [Online references](references.md) | Assigned readings, code paths, and reading questions |
| [Report and rubric](../../shared/REPORT.md) | Submission structure and common assessment criteria |

## Required evidence

One GEMM change; custom CuTe source; attention correctness; integrated timing and Amdahl comparison.

- [ ] Partial tiles, ragged lengths, and extreme logits are validated.
- [ ] One actual custom GPU attention kernel is implemented and checked.
- [ ] Gather/adapter costs and end-to-end impact are included.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../03_performance_model/README.md) · [Course home](../../README.md) · [Next chapter](../05_speculative_decoding/README.md)
