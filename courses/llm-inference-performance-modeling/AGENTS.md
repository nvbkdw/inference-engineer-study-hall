# Repository guidelines

## Course purpose

This is a code repository for an LLM inference engineering educational courses. It combines
theory for understanding and optimizing performance with practical labs that
build the inference software stack from scratch and from first principles.
Students develop one evolving implementation across the kernel, runtime,
serving, and distributed-system layers.

For each topic, connect the mathematical model to an implementation, a measurable
performance prediction, and an experiment that tests the prediction. Explain the
reasoning behind an optimization and the conditions under which it helps.

Write in a clean, concise, educational style using simple sentences and familiar words. 
Focus each sentence on one main idea and connect ideas logically. When introducing a new concept, 
briefly explain what it means and why it matters, starting with intuition before technical detail. 
Define unfamiliar terms and acronyms, and use concrete examples or analogies when helpful. 
Build explanations step by step, make key assumptions and cause-and-effect relationships clear, 
and match the depth to the reader’s background. Remove filler, repetition, and unnecessary jargon 
while preserving technical accuracy and enough reasoning for the reader to understand how and why things work.

## Chapter structure

Each chapter has two primary learning artifacts:

- `background.md`: the theory, definitions, mathematical derivations, architecture,
  assumptions, performance analysis, and supporting references.
- `code/lab.ipynb`: the practical implementation exercises, correctness checks,
  benchmark experiments, plots, and interpretation of results.

Keep theory in the background and the implementation/benchmark workflow in the
notebook. Link them where an exercise applies a particular derivation. Keep
instructions, exercises, and completion criteria in the notebook instead of
creating additional tutorial or assessment pages that duplicate it.

Some chapters still have an older layout. Use the two-artifact structure when
creating or consolidating a chapter; migrate existing content within the scope
of the requested work and update affected navigation links.

## Progressive implementation

Chapters are progressive. Modules implemented as exercises in one chapter carry
over to the next chapter as its foundation.

- State the prerequisite concepts and implementations at the start of each lab.
- Identify the new capability the chapter adds and the reusable modules it
  contributes to later chapters.
- Import and extend earlier implementations instead of copying them or building
  an unrelated engine for each chapter. Keep one source of truth for shared logic.
- Give reusable code explicit interfaces for configuration, tensor shapes, dtype,
  device, state ownership, and cache behavior. Document changes that affect later
  chapters and update dependent exercises when necessary.
- Keep reusable implementations accessible through modules under a chapter's
  `code/` directory or `shared/` as appropriate. If an implementation is defined
  in a notebook, provide an explicit import/export mechanism; later chapters must
  not depend on a previous notebook's live kernel state.

## Theory and teaching style

Derive results from tensor shapes, operations, data movement, and hardware limits.
Define symbols and units before using them, show intermediate reasoning, and
include concrete worked examples. Explain factors such as two FLOPs per
multiply-add and the two attention matrix products rather than presenting only
the final formula.

Keep notation consistent across chapters. Use Chapter 1's model-dimension
conventions, including `I` for MLP intermediate width. Use `AI` for arithmetic
intensity to avoid ambiguity. State counting conventions, approximation scope,
and cache lengths before and after appending tokens.

Use the course's Qwen3 configurations for concrete examples. Verify dimensions
against the configuration rather than assuming architectural identities such as
hidden width equaling query-head count times head dimension. Cite sources for
architecture and hardware specifications, while showing the derivation locally.

## Labs and benchmarks

Make students implement and understand the mechanism being taught. Existing
frameworks and inference engines can supply correctness oracles and performance
baselines; label those roles explicitly. Benchmarks should exercise the student's
implementation as it develops across chapters.

Organize labs around **prediction → implementation → correctness → measurement
→ explanation**. Keep exercises focused, explain expected inputs and outputs,
and connect observed bottlenecks to the theory. A justified negative result is
valid evidence.

Notebooks should run in a documented order from a fresh kernel, with visible
configuration and explicit dependencies. Keep correctness checks separate from
timing. Record model revision, hardware, precision, workload dimensions, backend,
warmup, repeats, and timing boundaries. Save predictions before measurements,
retain raw observations, and distinguish theoretical estimates, illustrative
inputs, and measured results. Label hardware-peak assumptions and the scope of
MFU, TTFT, and TBT calculations.

Use synchronization appropriate to asynchronous GPU execution. Do not present CPU
checks, synthetic workloads, or unexecuted code as real-model GPU measurements.
Document unavailable hardware or checkpoints as unmeasured scope.

## Repository workflow

Use `pyproject.toml` and [shared/SETUP.md](shared/SETUP.md) for the environment,
[shared/PROTOCOL.md](shared/PROTOCOL.md) for experiments, and existing course code
as the starting point. Preserve unrelated work in the checkout.

Run checks appropriate to the change. For implementation changes, verify
correctness and affected downstream modules; execute relevant notebook paths
from a fresh kernel when feasible. For documentation-only changes, check
notation, equations, and links without rerunning expensive model benchmarks.

From the course directory, the regression suite is
`python -m unittest discover -s tests -v`. Launch a chapter notebook with
`jupyter lab chapters/<chapter>/code/lab.ipynb`. Report which checks ran and any
remaining limitations accurately.
