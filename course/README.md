# Performance Modeling of LLM Inference

This college-level, lab-driven course develops a first-principles understanding of LLM inference performance. Students build and study one evolving inference engine, learning to reason at the **kernel, server, and system levels**.

For every project, ask three questions:

1. What performance does the analytical model predict, and where is the bottleneck?
2. What does the real implementation measure, and why does it differ from the prediction?
3. Which change should remove the bottleneck, and does the measured result support it?

The course is a **16-week research project using Qwen3-8B and Qwen3-32B**, with eight project chapters, guided tutorials, runnable teaching examples, annotated readings, experiment protocols, and completion criteria.

[Detailed syllabus](qwen3_inference_engineering_course.md) · [Laboratory setup](shared/SETUP.md) · [Independent study guide](shared/SELF_STUDY.md) · [Experiment protocol](shared/PROTOCOL.md) · [Report template and rubric](shared/REPORT.md)

The plan assumes **15–18 focused hours per week**, familiarity with Python/PyTorch, linear algebra, probability, and basic GPU programming. Prior CuTe DSL study and a Mini-SGLang implementation are useful; the setup guide identifies preparation work for students who need it.

**8B is your daily development model; 32B tests whether your performance explanations hold at larger scale.** Both use the same configuration-driven GQA implementation, though their dimensions differ. During speculative decoding, 8B becomes the draft and 32B the target. [8B configuration](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json), [32B configuration](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)

| Weeks     | Project                                                                         | Main result                                                        |
| --------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **1–2**   | [1. Reconstruct Qwen3](chapters/01_reconstruct_qwen3/README.md) | Numerical validation on both models; memory accounting |
| **3–4**   | [2. Build scheduling and bounded KV memory](chapters/02_runtime_and_kv/README.md) | Explain latency, throughput, and memory tradeoffs |
| **5–6**   | [3. Predict inference performance](chapters/03_performance_model/README.md) | Predict prefill/decode time on workloads withheld from calibration |
| **7–8**   | [4. Optimize GEMM and attention](chapters/04_kernels/README.md) | Explain kernel performance and its effect on model execution |
| **9–10**  | [5. Speculate with 8B → 32B](chapters/05_speculative_decoding/README.md) | Correct verification/rollback and a measured break-even analysis |
| **11–12** | [6. Quantize and evaluate quality](chapters/06_quantization/README.md) | Compare memory, latency, goodput, and interaction with speculation |
| **13–14** | [7. Implement TP and compare replicas](chapters/07_tensor_parallelism/README.md) | Choose how two GPUs should serve a fixed workload |
| **15–16** | [8. Transfer KV and evaluate P/D](chapters/08_prefill_decode/README.md) | Evaluate disaggregation and defend a final serving design |

Each chapter folder contains a learning guide (`README.md`), mathematical background (`background.md`), an independent entry lab (`standalone.md`), a full step-by-step project (`tutorial.md`), measurable acceptance goals (`assessment.md`), annotated online material (`references.md`), and code in `code/`. The chapters carry forward the same model, state conventions, and experiment protocol.

**Studying alone:** start with a chapter's standalone lab. Every project has a complete command sequence that saves assumptions/predictions, raw CSV data, a summary, and SVG/PNG figures. These entry experiments need no previous engine implementation. Measured experiments run on DGX Spark with CUDA events; P2/P5 retain clearly labeled untimed models. P7/P8 require two physical GPUs. Then follow the full tutorial and its measurable-goal checklist for the real Qwen/GPU project. The [independent study guide](shared/SELF_STUDY.md) lists all eight entry points and explains which results are measured or simulated.

The code provides small executable references for core mechanisms. Students extend these into the full engine, CuTe kernels, real quantization path, and serving experiments described in the tutorials. CPU checks and illustrative calculations are labeled separately from required GPU and real-checkpoint validation.

The [project dependency manifest](pyproject.toml) includes all supplied-script and checkpoint dependencies. The setup guide uses `uv sync --python 3.12` in a dedicated `.venv-spark`; add `--extra kernels` for the P4 CuTe DSL toolchain.

After activating a Spark-compatible CUDA environment using [setup](shared/SETUP.md), run the first measurement and optional correctness checks:

```bash
python chapters/03_performance_model/code/experiment.py --device cuda:0 --out results/spark-first --repeats 3
python -m unittest discover -s tests -v
```

The unit suite needs no pretrained model downloads. Tensor checks require PyTorch; Spark measurement checks also need CUDA and Matplotlib. The optional checkpoint workflow tests use Transformers, safetensors, and Accelerate. Missing dependency groups are explicitly skipped. Chapters 7 and 8 measure NCCL on two connected Sparks or two physical rental GPUs. CPU/Gloo is optional correctness-only work. See [validation notes](shared/VALIDATION.md) for what was actually checked while preparing this material.

Three choices keep the semester focused:

* **32B appears in Week 2.** You validate the shared implementation early, then repeat selected experiments instead of duplicating every sweep.
* **Speculation is a hypothesis.** The 8B draft may cost too much for its acceptance rate. An explained slowdown is a valid result. SGLang’s documented standalone draft-model path provides a comparison implementation. [Speculative decoding guide](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
* **Kernel work has a stopping point.** Study and modify one GEMM, then deeply investigate one attention kernel. Measure its integrated effect before expanding scope.

Most work stays on **Spark**. The guide proposes a **60 GPU-hour rental allowance** for prepared profiling, tensor-parallel, P/D, and cross-node experiments. B300 is an optional additional comparison.

Each assignment ends with the same evidence: **prediction → correctness → measurement → explanation → decision**. The document also includes a chapter template and your first five study sessions.

Your first milestone is **a correct 8B attention block, a reproducible reference baseline, and a tensor/memory inventory**. By the end of Week 2, the same implementation should run a validated short 32B continuation.
