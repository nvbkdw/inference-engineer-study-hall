# Course delivery and platform audit

Historical project IDs below are preserved with their original validation records: P2 runtime is now Chapter 4, P3 performance modeling is now Chapter 2, and P4 kernels is now Chapter 3. Current commands and navigation use the new chapter numbers.

A reading-only [Chapter 0](../chapters/00_introduction/README.md) introduces architecture, serving, and performance modeling without code or projects. It fits within the opening Week 1 reading allocation.

The eight-project, 16-week course structure is retained: each project chapter supplies
background, guided labs, executable samples, measurable acceptance criteria, and
annotated references. Chapter 1 consolidates theory and readings in `background.md`,
with tiny and full-model practical work in `lab.ipynb` and `lab2.ipynb`; the other
projects retain separate walkthrough and assessment pages. The user's platform
correction makes **DGX Spark the default for performance measurement**.

| Requirement | Current evidence |
|---|---|
| Kernel/model measurements on Spark | P1/P3/P4/P6 create CUDA tensors and use warmed CUDA-event timing; actual GB10 runs passed |
| Correct GPU assumptions | P3 uses BF16 Qwen-derived 4096x4096 GEMMs; FP32 numerical references explicitly disable TF32; actual hardware/build recorded |
| CPU scope | Optional untimed mathematical/state checks; `--device cpu` rejected by measured experiment entry points |
| Simulation scope | P2/P5 explicitly record no hardware timing and retain labeled modeled costs |
| Distributed scope | P7/P8 default to NCCL, require distinct physical GPUs, record rank hardware, and document one rank per connected Spark |
| Single-Spark honesty | Single-GPU guard tested; two-GPU measurement remains unperformed until that hardware session |
| Lab navigation | Root index, setup, study guide, Chapter 1 notebooks, and Chapters 2–8 walkthroughs/checklists describe the Spark path |
| Artifact/reproducibility checks | Manifests, predictions, raw CSV, summary, figures, and output-preservation checks are exercised by the test suite |
| Full-model custom forward | Lab 2 audited full 8B/32B checkpoints and generated five finite full-vocabulary vectors per model on Spark; full-model oracle comparison remains unperformed |
| Full project scope | Real 8B/32B integration, CuTe, packed quantization, full serving, TP, and P/D remain the syllabus's student deliverables |

See [VALIDATION.md](VALIDATION.md) for executed tests, environment versions, and
unmeasured scopes. The optional offline checkpoint test validates the strict
loader against a tiny HF model; it does not establish real-model GPU performance.
