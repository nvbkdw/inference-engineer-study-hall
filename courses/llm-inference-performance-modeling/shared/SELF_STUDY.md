# Independent study guide

The course is designed for both a 16-week class and a student working alone.
Begin with [Chapter 0](../chapters/00_introduction/README.md), a conceptual introduction with no installation, code, projects, or submissions. Its readings connect transformer math to the complete inference system.
Each project chapter (1–8) has a complete entry experiment, followed by a full project that
develops the same evolving Qwen inference engine. You can start an entry experiment
without completing earlier implementations; the full semester follows chapters
in order. Read the relevant background before executing the sample.

## Continue to a project

| Project | Entry lab | Measurable goals | Entry experiment scope |
|---|---|---|---|
| P1 | [Cache correctness and memory](../chapters/01_reconstruct_qwen3/code/lab.ipynb) | [P1 goals](../chapters/01_reconstruct_qwen3/code/lab.ipynb) | Qwen3 tiny, shared with Chapter 1 correctness: FP32 CUDA decode time and logical cache bytes on Spark |
| P2 | [Held-out prediction](../chapters/02_performance_model/code/lab.ipynb) | [P2 goals](../chapters/02_performance_model/code/lab.ipynb) | Interactive roofline, GEMM calibration, frozen predictions, real Qwen3 prefill/decode latency and tok/s across batch/context, and plots |
| P3 | [Profile](../chapters/03_kernels/code/lab.ipynb) → [implement](../chapters/03_kernels/code/lab2.ipynb) | [Matched sweep and goals](../chapters/03_kernels/code/lab3.ipynb) | Real 8B/32B BF16 profiles, CuTe fusions/attention, ablations, matched speedup and roofline evidence |
| P4 | [Bounded scheduling](../chapters/04_runtime_and_kv/standalone.md) | [P4 goals](../chapters/04_runtime_and_kv/assessment.md) | Real allocator logic with explicitly simulated service times |
| P5 | [Speculative sampling](../chapters/05_speculative_decoding/standalone.md) | [P5 goals](../chapters/05_speculative_decoding/assessment.md) | Finite-vocabulary distributions and an invented timing model |
| P6 | [Quantization tradeoffs](../chapters/06_quantization/standalone.md) | [P6 goals](../chapters/06_quantization/assessment.md) | Actual random-layer error/storage and Spark CUDA reconstruction cost |
| P7 | [Reduction costs](../chapters/07_tensor_parallelism/standalone.md) | [P7 goals](../chapters/07_tensor_parallelism/assessment.md) | Actual two-GPU NCCL all-reduce, three held-out payload sizes |
| P8 | [Handoff costs](../chapters/08_prefill_decode/standalone.md) | [P8 goals](../chapters/08_prefill_decode/assessment.md) | Actual two-GPU NCCL payload plus acknowledgment |

Follow [SETUP.md](SETUP.md) once. The measured entry experiments use DGX Spark, CUDA-enabled PyTorch, NumPy,
and Matplotlib. P1/P2/P3/P6 need one Spark. P1 first downloads the real weights used by its two-layer practice checkpoint. The performance notebook’s calibration needs no weights; its full-model sections use pinned 8B/32B checkpoints. P3 uses the same pinned 8B/32B checkpoints; P6 component samples need no checkpoint download.
P4/P5 are explicitly untimed model exercises alongside the full Spark labs.
P7/P8 require two connected Sparks or the syllabus's two-GPU rental. Two local
CPU/Gloo ranks on one Spark are not a measurement substitute.
The full projects retain the syllabus's Spark/connected-GPU resources and Qwen3
8B/32B validation. A machine without those resources can complete optional correctness/modeling work
and derivations, but must mark the corresponding full GPU milestones unmeasured.

For the first real-model baseline, use [P1 Lab 1](../chapters/01_reconstruct_qwen3/code/lab.ipynb)
to extract and verify the first two layers of pinned real Qwen3-8B weights. Continue
with [P1 Lab 2](../chapters/01_reconstruct_qwen3/code/lab2.ipynb) to download full
8B/32B checkpoints, inspect metadata and weights, and generate logits with custom blocks.

## A repeatable study session

1. **Predict, 15 minutes.** Write a short hypothesis and derive at least one unit-
   checked quantity. Keep this note unchanged after seeing the result.
2. **Trace, 20 minutes.** Read the local code sample and follow one tensor, request,
   probability vector, or message through its state changes.
3. **Run, 10–20 minutes.** Execute the documented correctness check and experiment.
   Use a new output directory for each run. Inspect `summary.json` before graphing.
4. **Explain, 20 minutes.** Reproduce one CSV result by hand. Read figures using
   their actual boundary: modeled bytes, CUDA workload time, simulation, or communication.
5. **Perturb, 20 minutes.** Change one assumption/input as instructed, predict its
   effect first, and preserve both original and changed-condition results.
6. **Assess, 15 minutes.** Use the chapter's checklist. Write the 400–600 word entry
   note and answer the self-check before starting the full project tutorial.

These estimates cover the entry experiment, not the full two-week assignment.
The full project uses the syllabus's 30–36 hours and 1,200–2,000 word research memo.

## Read an experiment directory

`manifest.json` identifies scope, repeats, software, and source hashes.
`prediction.json` records a hypothesis or calibrated numeric predictions written
before matching measurement. `results.csv` retains individual observations;
`summary.json` records checks and derived metrics. Additional CSVs preserve
calibration, requests, histograms, or memory accounting as needed. SVG figures
are scalable exports; PNG copies are convenient for inspection.

Points in supplied figures are medians and whiskers are observed ranges. They are
not confidence intervals. Between-run noise, different request traces, and random
weight fixtures have different interpretations. Use the [experimental protocol](PROTOCOL.md)
when making a full statistical performance/quality claim.

Do not optimize for a predetermined speedup. If a tiled reference is slower,
explain its extra dispatch and memory behavior. If a prediction misses the target,
preserve the miss and identify a new measurement. If a simulated policy wins,
check whether the conclusion survives a different service curve before proposing
it for a real server.

## What counts as completion?

The entry note demonstrates that you can run and interpret the mechanism. The full
chapter requires its checklist's real-model, hardware, correctness, and
measurement evidence. The final course outcome is the complete engine experiment
portfolio and a defensible design under a declared workload/hardware/SLO budget.

At the final defense, trace one request through the model, scheduler, cache,
verification/quantization choices, distribution, and optional handoff. Estimate
its bytes and critical-path time, then revise the design when the workload changes.
