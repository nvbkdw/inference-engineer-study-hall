# Chapter 2: An inference scheduler from first principles

**P2 · Weeks 3–4 · 30–36 hours**  
**Laboratory:** DGX Spark serving; untimed allocator/simulation exercise

How do scheduling and bounded KV allocation change useful serving capacity?

## Learning outcomes

- Implement request states, page ownership, admission, and continuous batching.
- Explain chunking, prefix reuse, fragmentation, and cancellation semantics.
- Measure TTFT, token gaps, and goodput under identical arrivals.

## Before you begin

For an independent first experiment, start with the [standalone lab](standalone.md).
It needs no previous engine implementation and produces saved data and figures.
Use the [measurable goals](assessment.md) to track entry and full-project completion.

P1's validated model and cache semantics. Follow the [shared setup](../../shared/SETUP.md) and
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

State/ownership diagrams; allocator gates; latency/goodput and fragmentation curves; a 32B spot check.

- [ ] Exhaustion is bounded and reservation is atomic.
- [ ] Cancellation and shared full prefixes preserve ownership and logits.
- [ ] Chunking and an efficient paged adapter have measured integrated behavior.

Passing the small sample is an initial correctness milestone. The tutorial defines
the full model/GPU experiments students must implement and measure. CPU or
illustrative outputs must not be reported as Qwen serving benchmarks.

## Navigation

[Previous chapter](../01_reconstruct_qwen3/README.md) · [Course home](../../README.md) · [Next chapter](../03_performance_model/README.md)
