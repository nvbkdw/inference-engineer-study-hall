# Background: exact speculation and its break-even point

Autoregressive decoding normally runs one target step per emitted token. A cheap
draft proposes several tokens; a target verifies them with a short block forward.
The value comes from making that block cheaper than several sequential target
steps. Draft cost, acceptance, rollback, memory, and prompt setup all matter.

## Distribution correctness

At an accepted history h, let q(x|h) be the draft distribution and p(x|h) the target.
A proposal x sampled from q is accepted with `min(1,p(x)/q(x))`. The probability
mass contributed by accepted proposals at x is `min(p(x),q(x))`. The missing mass
is `max(p(x)-q(x),0)`. On rejection, sample from that positive residual normalized
over the vocabulary. Accepted mass plus correction mass is p.

This one-position argument applies conditionally after each accepted prefix.
Stop verification at the first rejection; later proposals were generated under
the wrong history. If all k proposals are accepted, emit a bonus sample from the
target's distribution after those k tokens. These rules follow
[Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) and
[Chen et al.](https://arxiv.org/abs/2302.01318).

Use the actual distributions after temperature/truncation. A proposed token must
have positive q probability, but target-only support may appear in the residual.
If p=q, rejection has zero probability; a zero-mass residual must not be normalized.
Identical random seeds do not require identical output strings across algorithms:
they consume RNG draws differently. Test distributions, not seed-level equality.

Greedy verification is a separate mode: accept the longest draft prefix matching
target argmax at each corresponding history; emit target argmax at the mismatch,
or a bonus target argmax if all match. It is easier to debug cache reconciliation
with greedy fixtures, but it does not test stochastic rejection sampling.

## Shifted logits and pending tokens

Choose the invariant: the last emitted token is pending; the cache contains all
earlier processed tokens. If processed length is p and draft proposals are d1..dk,
target input is `[pending,d1,...,dk]`. Output row 0 scores d1, row 1 scores d2,
and row k scores the bonus. This is a k+1 input call under this convention.

After verification the tentative target cache has p+k+1 positions. If a proposals
are accepted and a correction is emitted, keep p+1+a processed positions and leave
the correction pending. When all are accepted, keep p+1+k and leave the bonus
pending. Stop at EOS/output limit even if more verified tokens exist. The first
cycle after prompt prefill may instead reuse saved prompt logits; document that
special initialization and its indexing.

The draft cache needs its own reconciliation and catch-up. A final proposed token
may still be pending on the draft side. Neither model can reuse the other's KV:
their layer counts and weights differ. The equality to preserve is committed token
history, not equality of tensor shape or cache length at every intermediate moment.

## Predict the economics

Let A be useful emitted target tokens per cycle, tD complete draft time including
catch-up, tV verification, tO sampling/commit overhead, and tT target-only step time:

```text
speedup_decode ≈ E[A] * tT / (tD + tV + tO)
break even iff E[A] * tT > tD + tV + tO
```

For independent constant acceptance alpha, no stopping boundaries, and one
correction/bonus, `E[A]=sum(alpha**j for j in 0..k)`. Real acceptance is conditional
and domain dependent, so measure the accepted-prefix histogram. Include draft
prefill in full-request latency and count both models' weights/caches. Report
client-visible bursts and pauses; dividing cycle time by emitted tokens can hide
long streaming gaps.
