# Chapter 5: Speculating with an 8B draft

**P5 · Weeks 9–10 · 30–36 hours**  
**Laboratory:** DGX Spark 8B + 32B placement; untimed stochastic oracle

When does draft cost pay for accepted target tokens?

## Learning outcomes

- Prove and implement exact rejection sampling separately from greedy matching.
- Align target block logits and reconcile both model caches.
- Measure acceptance distributions and full-request break-even behavior.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

P1 correct caches, P2 state ownership, and P3 timing boundaries. Follow the [shared setup](../../shared/SETUP.md) and
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

Finite-vocabulary proof/checks; forced-rejection cache traces; real-prompt acceptance and speedup curves.

- [ ] Tokenizers and transformed sampling distributions are consistent.
- [ ] Rejection, bonus, EOS, and length boundaries preserve committed state.
- [ ] The enable/disable recommendation includes both models and prompt costs.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../04_kernels/README.md) · [Course home](../../README.md) · [Next chapter](../06_quantization/README.md)
