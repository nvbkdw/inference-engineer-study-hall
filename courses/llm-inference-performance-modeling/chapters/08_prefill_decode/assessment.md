# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: independent state | Clone/continuation checks pass; reject wrong revision and inconsistent processed length | lab/test output |
| Entry: transfer prediction | 3 calibration and 3 withheld payload sizes on two distinct CUDA GPUs with NCCL; >=3 repeats; every received payload matches before the next trial | raw transfer data, prediction.json, summary.json |
| Full: real continuation | 8B lengths 15/16/17/256/2048 plus feasible long context and selected 32B; EOS/output-budget boundaries | colocated vs transferred logit/continuation artifacts |
| Full: ownership protocol | Cancellation during reservation, transfer, and installation; duplicate acknowledgments never double-free; source retained until ready | request epoch/ownership traces and failure checks |
| Full: system comparison | >=3 repeats on matched 2 GPUs for replicas, 1P/1D, feasible TP; W1–W4 and fixed W1/W3 mixture; first-to-second gap included | stage queues, transfer bytes, latency and goodput data |
| Decision | For both models justify precision, batching/cache, speculation, TP/replicas, and P/D; include one workload change that reverses a choice | final design memo and 30-minute defense |

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

