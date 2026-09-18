# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: sampling law | Exact finite-vocabulary tests pass; empirical deviations remain inside the recorded bound at all 36 conditions | lab output and summary.json |
| Entry: cost reconstruction | Recompute E[A] and speedup for at least one winning and one losing condition if present; label all costs invented | histogram calculation and break-even figure |
| Full: tokenizer/greedy checks | Match full tokenization semantics; greedy target-only equivalence for all rejection positions and all-accepted bonus | tokenizer audit and verifier fixtures |
| Full: state boundaries | Pass forced k=4/accept2/reject3 continuation across multiple cycles, page boundary, EOS, and output limit | cache-length/page traces and fresh-logit comparisons |
| Full: measured matrix | k=1,2,4,8 at B=1; >=30 prompts/domain for prose/code/extraction; include draft prefill/catch-up and both caches | real-prompt manifest, per-cycle and per-request rows |
| Decision | Enable/disable the pair for a stated workload based on E[A], full-request latency, and useful output | memo and measured break-even curve |

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

