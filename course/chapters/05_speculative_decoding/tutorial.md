# Tutorial: verify the sampler before timing the draft

## 1. Check identity and capacity (3 hours)

Use 8B as draft, 32B as target. Compare the pinned token-to-ID mappings, merges,
added tokens, special tokens, and EOS rules; vocabulary size equality alone is
insufficient. Render prompts once and supply the same IDs to both models. Disable
thinking for core comparisons and start with no history-dependent logits processors.

Calculate combined BF16 weights (~76.28 GiB), both live caches, temporary verified
positions, workspaces, and loading overhead. Start B=1 on the syllabus's Spark
configuration only after checking actual free capacity. If placement spans two
GPUs, count both GPUs and communication in the comparison.

## 2. Prove the finite-vocabulary sampler (4 hours)

```bash
python chapters/05_speculative_decoding/code/lab.py
```

Inspect `output_distribution`: it analytically adds accepted and residual mass.
`cycle` is a history-dependent sequential oracle. It does not accelerate a model;
its target calls establish the answer a batched verifier must reproduce.

Reimplement the one-position rule using p=(.7,.2,.1), q=(.2,.6,.2). Enumerate the
output distribution before Monte Carlo sampling. Test disjoint supports, p=q,
immediate rejection, EOS, all accepted, and zero remaining output budget. The
sample's Monte Carlo threshold is a Hoeffding union bound with stated sample count
and failure probability; explain why a statistical test can occasionally fail.

Extend the fixture so target/draft probabilities depend on the previous token.
Compare empirical distributions at multiple histories. Keep an exact enumeration
check to distinguish a logic error from random variation.

## 3. Implement greedy target block verification (5 hours)

Build a target-only greedy baseline using the same numerical path. Choose the
pending-token convention from the background and log input positions, output-logit
rows, tentative lengths, and committed lengths for every cycle. Process a proposed
block in one target call using the correct rectangular mask.

For pending u and proposals `[a,b,c,d]`, write which target output row scores each
proposal and the bonus. Test a deliberately wrong draft that forces each possible
rejection location. If target block and target one-token results disagree near a
logit tie, inspect numeric differences before blaming the verifier.

## 4. Reconcile both caches (5 hours)

Force four proposals, accept two, reject the third, emit a correction, then continue
for several cycles. Truncate target/draft logical state and free rejected private
pages. Make stale positions unreachable through both length masks and page tables.
Process accepted/correction/bonus tokens as needed to catch the draft up.

After every commit, compare selected next logits with fresh target computation at
the committed history. Include rejection across a page boundary and EOS within the
accepted prefix. Use fixed stopping rules: never emit extra tokens merely because
the verification block already computed them. Prefix sharing stays disabled until
the unshared ownership path passes.

**Checkpoint:** cache reconciliation remains correct over multiple cycles, not
only at the first block. Greedy outputs match the target-only numerical reference.

## 5. Add stochastic verification (4 hours)

Compute p and q after the declared temperature/top-k/top-p transforms for each
hypothetical accepted history. Use the finite-vocabulary algorithm as the oracle.
Store enough draft probabilities to form the residual at the first rejection.
Begin without repetition penalties; adding them requires evaluating each proposed
prefix consistently. Demonstrate that the sampler preserves the chosen target
distribution, which is an implementation/numerics claim rather than text quality.

## 6. Measure break-even on real prompts (7 hours)

Use 30 prompts each from prose, code, and structured extraction. Preserve provenance
and exact IDs. Sweep k=1,2,4,8 at B=1. For every cycle record accepted prefix, useful
emitted count, draft time (including catch-up), verification time, and other overhead.
Record target-only step time at comparable contexts. Include both models' prefill,
memory, TTFT, completion latency, and emitted-token timestamps per request.

Compute E[A] from actual cycle outcomes; do not substitute a pooled acceptance
fraction into the independence formula and call it measured expectation. Plot
predicted versus observed decode speedup and full-request speedup by domain/k.
The sample's printed 0.86x/1.49x examples are invented equation demonstrations.
Run one compatible SGLang standalone-draft comparison using its pinned guide and
record actual backend settings, rather than mixing different draft algorithms.

## 7. Submit and decide (4 hours)

Provide stochastic oracle evidence, greedy real-model checks, cache traces for the
forced rejection case, acceptance histograms, break-even plots, and a recommendation
to enable or disable this pair for a declared workload. An explained slowdown is
a complete outcome. No runtime speedup is established by the finite-vocabulary code.

**Defense:** Why can a more accurate draft be slower overall? Which target row
scores the first proposal, and what exactly remains in each cache after rejection?
