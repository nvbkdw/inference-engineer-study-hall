# Chapter 8: The price of moving KV

**P8 · Weeks 15–16 · 30–36 hours**  
**Laboratory:** Two connected DGX Sparks or two physical rental GPUs; optional untimed state oracle

When does phase isolation outweigh transfer, queueing, and stranded capacity?

## Learning outcomes

- Define and validate a request-state handoff including a pending token.
- Model and measure transfer and stage-specific queues.
- Defend an integrated serving design using evidence from all eight projects.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

P2 ownership, P3 modeling, and P7 topology; results from P5/P6 for final decisions. Follow the [shared setup](../../shared/SETUP.md) and
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

Handoff protocol/checks; withheld transfer prediction; fixed-resource system comparison; final design defense.

- [ ] Real-model continuation and cancellation during handoff are validated.
- [ ] Transfer and duplicated state are measured at relevant payload sizes.
- [ ] P/D is compared with two replicas and feasible TP at equal hardware budget.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../07_tensor_parallelism/README.md) · [Course home](../../README.md)
