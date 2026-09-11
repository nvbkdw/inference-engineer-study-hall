# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: sharding identity | Untimed sharding oracle passes; NCCL MLP passes on two physical GPUs | lab output from simulated and real distributed paths |
| Entry: communication prediction | 3 calibration and 3 withheld sizes on two distinct CUDA GPUs with NCCL, >=3 repeats; all reduced elements equal 3 | calibration.csv, prediction.json, summary.json |
| Full: model validation | Single attention/MLP layer, full tiny decoder, 8B prefill/decode, selected 32B inputs all pass declared numerical criteria | TP=1/2 logit comparisons |
| Full: accounting | Report actual collectives/payloads and replicated vs sharded weight/KV bytes; investigate differences from 2 reductions/layer | trace and per-rank inventory |
| Full: equal-resource comparison | One TP=2 versus two feasible TP=1 replicas on the same 2 GPUs; fixed traces/SLOs; >=3 repeats | latency/goodput/memory curves |
| Decision | Select TP or replicas for both models; distinguish capacity-only advantages and unmeasured cross-node extensions | placement memo and scaling prediction |

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

