# Measurable goals and acceptance evidence

Use this table as a checklist before beginning and at submission. **Entry** goals
can be completed with the supplied [standalone lab](standalone.md). **Full** goals
cover the original syllabus project and require its model/GPU/serving work.
Completing entry goals alone does not complete the full project.

| Goal | Measurable acceptance condition | Required evidence |
|---|---|---|
| Entry: conservation | 90 completions for every one of 9 trace/policy conditions; zero owned pages after drain | results.csv and summary.json |
| Entry: metric reconstruction | Recompute at least one goodput row from request SLO decisions and observation duration | calculation with units and exact matching CSV row |
| Full: allocation invariants | Pass page-boundary 15/16/17, exhaustion, cancellation in 3 states, and shared-prefix owner release tests | invariant tests and transition traces |
| Full: controlled experiments | Unchunked plus 256/1024 chunks; page sizes 8/16/32 in a declared feasible matrix; >=3 arrival traces | manifest and request/token timestamp rows |
| Full: model behavior | 8B batching/chunk/prefix logits agree; one decisive 32B comparison; efficient paged adapter profiled with gather baseline | numerical artifacts and trace |
| Decision | Select a policy from latency/goodput and fragmentation curves; explain one tradeoff without changing workload/SLOs | memo and reproducible figures |

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

