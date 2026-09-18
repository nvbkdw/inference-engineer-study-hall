# Chapter 3: From a performance gap to kernel optimization

**P3 · Weeks 5–6 · 30–36 hours**
**Laboratory:** DGX Spark attention measurements; optional CPU oracle; H100 for Hopper study

Which gap between Chapter 2's estimated and observed performance can a kernel
change explain and reduce, and does that change improve whole-model latency?

## Learning outcomes

- Distinguish a theoretical bound from a calibrated estimate and diagnose their gaps to measured latency.
- Reason about GEMM reuse, resource use, and one bounded design change.
- Derive online softmax and implement a bounded CuTe GQA attention path.
- Connect kernel, adapter, and model speed through Amdahl's law.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

Bring Chapter 2's frozen predictions, raw model timings, and a separate profile,
plus basic CUDA/CuTe GEMM experience. Reuse the Chapter 1 model through
[Chapter 2's importable model module](../02_performance_model/code/model.py);
no previous notebook kernel state is needed. This chapter contributes reusable
kernel implementations and attention adapters for Chapter 4's runtime.
Follow the [shared setup](../../shared/SETUP.md) and
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

One diagnosed Chapter 2 performance gap; a frozen optimization prediction; one GEMM change; custom CuTe source; attention correctness; integrated timing and Amdahl comparison.

- [ ] Partial tiles, ragged lengths, and extreme logits are validated.
- [ ] One actual custom GPU attention kernel is implemented and checked.
- [ ] Gather/adapter costs and end-to-end impact are included.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../02_performance_model/background.md) · [Course home](../../README.md) · [Next chapter](../04_runtime_and_kv/README.md)
