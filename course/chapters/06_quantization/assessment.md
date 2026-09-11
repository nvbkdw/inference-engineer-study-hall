# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: representation | All signed nibble roundtrips and zero-group/rounding checks pass; reproduce byte counts for all 5 group sizes | lab output and results.csv |
| Entry: error interpretation | Compare both activation conditions on 3 paired weight fixtures; explain delta_Y rather than inferring model quality | error.svg and paired calculations |
| Full: calibration | Actual 8B and 32B checkpoints from one pinned compatible recipe; start with 128 sequences of <=512 tokens, disjoint from evaluation | recipe, calibration IDs, checkpoint format metadata |
| Full: real execution | Identify loaded packed storage and actual kernel; report weights/scales/workspace/KV separately | backend logs, trace, memory table |
| Full: quality and serving | Paired held-out NLL and initial 512 task examples; frozen quality budget; fixed-work, capacity, and offered-load comparisons | example-level scores, uncertainty, serving data |
| Decision | Evaluate all 3 target/draft precision conditions with matching target-only baselines; choose precision/speculation using quality and goodput | memo and uncertainty-aware decision |

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

