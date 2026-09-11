# Research memo and assessment

Submit 1,200–2,000 words, 2–3 decisive figures, raw results, a filled manifest,
the analysis script, and a tagged engine revision for each chapter.

1. **Question:** model, workload, hardware budget, and decision.
2. **Derivation:** tensor shapes, units, assumptions, and a pre-measurement prediction.
3. **Mechanism:** pseudocode or a small excerpt explaining your implementation.
4. **Correctness:** oracle, numerical criteria, adversarial cases, and failures.
5. **Experiment:** controls, samples, warmup, metadata, and uncertainty method.
6. **Results:** prediction/measurement plot and the intervention's tradeoff curve.
7. **Explanation:** profile evidence, residuals, failed hypotheses, and model limits.
8. **Decision:** enable/disable or select a configuration; what could overturn it?

| Dimension | Weight | Full-credit evidence |
|---|---:|---|
| Correctness | 30% | Passes chapter gates; oracle and tolerances are justified |
| Prediction and explanation | 25% | Derived model predicts unseen conditions or explains its failure |
| Experimental method | 20% | Matched work, uncertainty, resource and failure accounting |
| Implementation understanding | 15% | Student reconstructs the mechanism and defends choices orally |
| Writing and reproducibility | 10% | Another student can follow and rerun the experiment |

Instructors should grade the reasoning behind a negative result equally with a
speedup. Do not award optimization credit when correctness, workload, or hardware
allocation changed silently. Use the oral question in each chapter; ask students
to rederive one key mechanism without viewing the sample code.
