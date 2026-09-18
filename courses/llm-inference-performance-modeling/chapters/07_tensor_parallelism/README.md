# Chapter 7: Tensor parallelism or replicas

**P7 · Weeks 13–14 · 30–36 hours**  
**Laboratory:** Two connected DGX Sparks or two physical rental GPUs; optional untimed CPU sharding oracle

How should a fixed two-GPU budget serve a declared workload?

## Learning outcomes

- Derive QKV/MLP sharding and local versus reduced tensors.
- Validate a TP=2 model and measure actual collectives.
- Choose TP or replicas using matched resource and SLO accounting.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

P1 model, P2 model/calibration, basic process communication. Follow the [shared setup](../../shared/SETUP.md) and
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

Sharding diagrams; full-model checks; collective curves; predicted versus observed scaling; placement decision.

- [ ] TP layers from 8B, full 8B, and selected 32B cases pass.
- [ ] Small-message communication and replicated memory are accounted for.
- [ ] Replica comparisons are feasible and use the same two GPUs/arrivals.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../06_quantization/README.md) · [Course home](../../README.md) · [Next chapter](../08_prefill_decode/README.md)
