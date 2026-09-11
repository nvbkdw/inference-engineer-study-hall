# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: held-out chronology | CUDA BF16 calibration M=1/16/512, K=N=4096; prediction.json persisted before all 6 held-out measurements | calibration.csv, prediction.json, code path |
| Entry: error calculation | Recompute six per-point relative errors and their median; explain the two largest residuals | summary.json and a residual note |
| Full: work inventory | List actual 8B/32B projection shapes and sequential operators, including omitted/fused costs | operation ledger with FLOPs and bytes |
| Full: predictive coverage | >=6 feasible withheld points span both models and prefill/decode; >=3 repeats each | frozen predictions and raw timings |
| Full: modeling target | Aim for median relative error <=25%; if missed, identify a specific failure and a discriminating measurement | error plot and profiler-backed diagnosis |
| Decision | Predict and test one intervention; retain original held-out score and use new points for a revised model | memo and new-condition comparison |

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

