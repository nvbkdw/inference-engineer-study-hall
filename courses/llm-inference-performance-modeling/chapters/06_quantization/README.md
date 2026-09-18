# Chapter 6: Quantization as a serving choice

**P6 · Weeks 11–12 · 30–36 hours**  
**Laboratory:** DGX Spark numerical/reconstruction measurements; supported packed backend on Spark or H100

When does smaller representation improve capacity within a measured quality budget?

## Learning outcomes

- Derive quantization, clipping, scale, and packing costs.
- Distinguish numerical reconstruction from actual packed-weight execution.
- Evaluate paired quality, memory, latency, goodput, and speculation interaction.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

P2 cost model and P5 baseline; probability and basic statistical intervals. Follow the [shared setup](../../shared/SETUP.md) and
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

Packing tests; calibrated 8B/32B checkpoints and recipe; quality intervals; serving/speculation comparisons.

- [ ] Calibration and evaluation data are disjoint and pinned.
- [ ] Real packed execution is identified in modules/traces.
- [ ] Both model sizes and all three speculation precision conditions are evaluated.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../05_speculative_decoding/README.md) · [Course home](../../README.md) · [Next chapter](../07_tensor_parallelism/README.md)
