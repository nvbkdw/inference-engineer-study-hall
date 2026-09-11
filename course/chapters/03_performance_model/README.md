# Chapter 3: Predicting inference latency

**P3 · Weeks 5–6 · 30–36 hours**  
**Laboratory:** Spark; optional prepared H100 profiling

Which measured hardware and component costs predict unseen prefill/decode workloads?

## Learning outcomes

- Count FLOPs, logical traffic, and attention pairs with units.
- Calibrate shape-specific rates and sum costs on the critical path.
- Freeze predictions and explain residuals on held-out conditions.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

P1 shapes and P2 execution/serving boundaries. Follow the [shared setup](../../shared/SETUP.md) and
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

Operation ledger; calibration table; at least six withheld predictions; profile-supported intervention.

- [ ] Hardware inputs are measured or clearly illustrative.
- [ ] Six feasible held-out points span both models and both phases.
- [ ] Residuals and one optimization decision have profiler evidence.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../02_runtime_and_kv/README.md) · [Course home](../../README.md) · [Next chapter](../04_kernels/README.md)
