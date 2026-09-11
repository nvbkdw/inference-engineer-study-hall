# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: numerical oracle | All 12 CUDA FP32 method/length cases pass rtol=1e-4/atol=1e-5; partial-tile/extreme-score unit checks pass | summary.json and test output |
| Entry: storage derivation | Reproduce dense/tiled temporary sizes; default S=512/tile32 ratio equals 16 | score_storage.csv and derivation |
| Full: GEMM investigation | One bounded source change at M=1,4,16,128,512, matched dtype/backend; correctness before timing | source diff and raw timing/counter table |
| Full: CuTe correctness | S=127,128,129,2048,8192; B=1,4; ragged lengths; both head configurations; justified BF16 criteria | GPU comparison and memory/race-check records |
| Full: integrated performance | Kernel-only, adapter-inclusive, and full-model timing with >=3 repeats; Amdahl prediction from measured baseline fraction | trace and three timing boundaries |
| Decision | Explain a measured improvement or gap, with the exact shape/hardware scope | memo and kernel integration decision |

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

