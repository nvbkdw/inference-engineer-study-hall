# Performance Modeling of LLM Inference

This college-level, lab-driven course develops a first-principles understanding of LLM inference performance. Students build and study one evolving inference engine, learning to reason at the **kernel, server, and system levels**.

For every project, ask three questions:

1. What performance does the analytical model predict, and where is the bottleneck?
2. What does the real implementation measure, and why does it differ from the prediction?
3. Which change should remove the bottleneck, and does the measured result support it?

The course is a **16-week research project using Qwen3-8B and Qwen3-32B**, with a conceptual introduction followed by eight project chapters, guided tutorials, runnable teaching examples, annotated readings, experiment protocols, and completion criteria.

**Start with [Chapter 0: From text to a serving system](chapters/00_introduction/README.md).** It introduces transformer mathematics, cached generation, inference servers, speculation, distributed serving, and performance modeling through readings and diagrams. It has no code or projects and fits within the opening Week 1 reading time.

[Detailed syllabus](qwen3_inference_engineering_course.md) · [Laboratory setup](shared/SETUP.md) · [Independent study guide](shared/SELF_STUDY.md) · [Experiment protocol](shared/PROTOCOL.md) · [Report template and rubric](shared/REPORT.md)

The plan assumes **15–18 focused hours per week**, familiarity with Python/PyTorch, linear algebra, probability, and basic GPU programming. Prior CuTe DSL study and a Mini-SGLang implementation are useful; the setup guide identifies preparation work for students who need it.

**Model progression: one Qwen3 tiny configuration for Chapter 1 correctness and Spark timing, then real Qwen3-8B and Qwen3-32B checkpoints.** Tiny retains real Qwen3-8B layers 0–1, embeddings, final norm, and vocabulary head, with `(L,D,I,Hq,Hkv,R,V)=(2,4096,12288,32,8,128,151936)` and is confined to Chapter 1.

**8B is your daily development model; 32B tests whether your performance explanations hold at larger scale.** Both use the same configuration-driven GQA implementation, though their dimensions differ. During speculative decoding, 8B becomes the draft and 32B the target. [8B configuration](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json), [32B configuration](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)

| Timing    | Chapter                                                                         | Main result                                                        |
| --------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Week 1 opening** | [0. Introduction](chapters/00_introduction/README.md) | A conceptual map from tokens and transformer operators to distributed serving; reading only |
| **1–2**   | [1. Reconstruct Qwen3](chapters/01_reconstruct_qwen3/background.md) | Numerical validation on both models; memory accounting |
| **3–4**   | [2. Build scheduling and bounded KV memory](chapters/02_runtime_and_kv/README.md) | Explain latency, throughput, and memory tradeoffs |
| **5–6**   | [3. Predict inference performance](chapters/03_performance_model/README.md) | Predict prefill/decode time on workloads withheld from calibration |
| **7–8**   | [4. Optimize GEMM and attention](chapters/04_kernels/README.md) | Explain kernel performance and its effect on model execution |
| **9–10**  | [5. Speculate with 8B → 32B](chapters/05_speculative_decoding/README.md) | Correct verification/rollback and a measured break-even analysis |
| **11–12** | [6. Quantize and evaluate quality](chapters/06_quantization/README.md) | Compare memory, latency, goodput, and interaction with speculation |
| **13–14** | [7. Implement TP and compare replicas](chapters/07_tensor_parallelism/README.md) | Choose how two GPUs should serve a fixed workload |
| **15–16** | [8. Transfer KV and evaluate P/D](chapters/08_prefill_decode/README.md) | Evaluate disaggregation and defend a final serving design |

Chapter 1 has one [theory and reading guide](chapters/01_reconstruct_qwen3/background.md) and two interactive notebooks: [Lab 1, reconstruct and measure tiny](chapters/01_reconstruct_qwen3/code/lab.ipynb), then [Lab 2, run full 8B/32B weights](chapters/01_reconstruct_qwen3/code/lab2.ipynb). Instructions and acceptance goals live in the notebooks; reusable utilities remain in `code/`. Chapters 2–8 have separate overview, background, standalone lab, full tutorial, assessment, and reference pages. The projects share model conventions and an experiment protocol. Chapter 0 contains only conceptual readings, equations, diagrams, and references.

**Studying alone:** read Chapter 0 first, then Chapter 1's background and Lab 1. Later chapters begin with their standalone labs. Every project saves assumptions/predictions, raw CSV data, a summary, and SVG/PNG figures. These entry experiments need no previous engine implementation. Measured experiments run on DGX Spark with CUDA events; P2/P5 retain clearly labeled untimed models. P7/P8 require two physical GPUs. Continue to the full project and its checklist for real Qwen/GPU validation. The [independent study guide](shared/SELF_STUDY.md) lists all eight entry points and explains which results are measured or simulated.

The code provides small executable references for core mechanisms. Students extend these into the full engine, CuTe kernels, real quantization path, and serving experiments described in the tutorials. CPU checks and illustrative calculations are labeled separately from required GPU and real-checkpoint validation.

The [project dependency manifest](pyproject.toml) includes all supplied-script and checkpoint dependencies. The setup guide uses `uv sync --python 3.12` in a dedicated `.venv-spark`; add `--extra kernels` for the P4 CuTe DSL toolchain.

After activating a Spark-compatible CUDA environment using [setup](shared/SETUP.md), run the first measurement and optional correctness checks:

```bash
python chapters/03_performance_model/code/experiment.py --device cuda:0 --out results/spark-first --repeats 3
python -m unittest discover -s tests -v
```

P1 model and checkpoint tests require the prepared two-layer checkpoint; tests skip that scope when its weights are absent. Tensor checks require PyTorch; Spark measurement checks also need CUDA and Matplotlib. The optional checkpoint workflow tests use Transformers, safetensors, and Accelerate. Missing dependency groups are explicitly skipped. Chapters 7 and 8 measure NCCL on two connected Sparks or two physical rental GPUs. CPU/Gloo is optional correctness-only work. See [validation notes](shared/VALIDATION.md) for what was actually checked while preparing this material.

Three choices keep the semester focused:

* **32B appears in Week 2.** You validate the shared implementation early, then repeat selected experiments instead of duplicating every sweep.
* **Speculation is a hypothesis.** The 8B draft may cost too much for its acceptance rate. An explained slowdown is a valid result. SGLang’s documented standalone draft-model path provides a comparison implementation. [Speculative decoding guide](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
* **Kernel work has a stopping point.** Study and modify one GEMM, then deeply investigate one attention kernel. Measure its integrated effect before expanding scope.

Most work stays on **Spark**. The guide proposes a **60 GPU-hour rental allowance** for prepared profiling, tensor-parallel, P/D, and cross-node experiments. B300 is an optional additional comparison.

Each assignment ends with the same evidence: **prediction → correctness → measurement → explanation → decision**. The document also includes a chapter template and your first five study sessions.

Your first milestone is **a correct 8B attention block, a reproducible reference baseline, and a tensor/memory inventory**. By the end of Week 2, the same implementation should run a validated short 32B continuation.
