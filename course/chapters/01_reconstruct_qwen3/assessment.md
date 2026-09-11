# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: cache bytes | Exact agreement at S=128,256,512,1024,2048; default FP32 CUDA slope 4096 bytes/token | memory.csv and slope derivation |
| Entry: cache correctness | All five CUDA next-logit comparisons pass rtol=1e-4/atol=1e-5; original unequal-chunk checks pass | summary.json and lab output |
| Full: weight inventory | 100% trainable checkpoint tensors mapped for both models; zero unexplained missing/unexpected tensors | 8B/32B mapping CSV and shape audit |
| Full: numerical equivalence | Both models: selected full-vocabulary logits; 8B: full, incremental, unequal chunks, mixed batch; document BF16 criteria before acceptance | saved reference/custom comparisons |
| Full: performance model | S=128,512,2048 with >=3 repeats; cached/recomputed timing and logical/live/reserved memory separately | raw timings, memory curve, residual explanation |
| Decision | One measured intervention and a 1,200–2,000 word memo; explain both intercept and cache slope | report, figures, manifest, code revision |

Performance directions are hypotheses, not mandatory speedups. A negative result
can meet the goals when correctness passes, controls are matched, measurements
are reproducible, and the explanation is supported. Unperformed hardware work
must be marked unmeasured, never inferred from a toy check.

Apply the [common rubric](../../shared/REPORT.md): correctness 30%, prediction and
explanation 25%, experimental method 20%, implementation understanding 15%, and
writing/reproducibility 10%. At least three independent repeats are required for
full timing comparisons; record raw observations and uncertainty. Numerical
acceptance for BF16 must be justified against the declared oracle and frozen
before accepting a result. For quality intervals, record pass/fail/inconclusive
rather than hiding uncertainty with a point estimate.

Before the oral defense, reproduce one equation without the code, identify one
failed hypothesis, and explain the scope of each figure: measured, modeled, or
illustrative. Use the chapter tutorial's defense questions and retain the exact
source revision and experiment manifest with your submission.

