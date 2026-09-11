# Tutorial: build a bounded runtime

## 1. Establish a static baseline (3 hours)

Use the P1 model with greedy decoding, a contiguous cache, and fixed batches.
Create W1/W2 request fixtures from the [protocol](../../shared/PROTOCOL.md). Log
arrival, scheduled work, processed length, and each emitted token timestamp.
Predict memory at batch 1/4/16 before admitting work.

Write down the state transition table and the invariant `cache length = processed
tokens`, which can differ from history length by one pending token.

## 2. Implement a physical block pool (5 hours)

```bash
python chapters/02_runtime_and_kv/code/lab.py
```

Inspect `Pool.append`, `share_prefix`, and `release`. The sample manages ownership
metadata; it allocates no KV tensors and runs no model. Its exhaustion case leaves
both the table and length unchanged. Reconstruct this atomic reservation behavior
in your runtime, then connect physical slots to GPU K/V storage.

Use P=16 initially. Implement logical-page translation and a correctness adapter
that gathers populated KV into contiguous tensors. Compare logits to P1 after each
step. Add lengths 15,16,17 and an exhausted-pool scenario. Count every copied byte.
Before timing efficient serving, integrate one supported paged attention backend
and retain the gather path as an oracle.

**Checkpoint:** distinct private pages never alias; counts equal actual owners;
failed reservation changes no state; only populated positions can be read.

## 3. Schedule one iteration at a time (5 hours)

Represent waiting requests separately from active prefill/decode work. Start with
FCFS admission and decode-first iteration selection. `select_work` illustrates a
token budget; add a free-page budget and rollback of reservations if an entire
planned work item cannot run. One iteration follows:

```text
collect arrivals/cancellations
select decode tokens and prefill chunks
reserve required KV pages and temporary capacity
construct IDs, absolute positions, block tables, and ragged lengths
run model; commit processed lengths; sample/emit outputs
release terminal request state after in-flight work completes
```

Replace completed requests between iterations. If decode demand exceeds the token
budget, use round-robin selection or reject excess admission; document the rule.
Include a maximum wait or aging rule so repeated decode priority cannot silently
starve prefill. Verify the same request alone and amid changing batch membership.

## 4. Add chunking and cancellation (5 hours)

Compare unchunked prefill with chunks 256 and 1024 under the same token budget.
Predict the longest decode pause from a prefill chunk's service time. Cancel a
request while waiting, between prefill chunks, and after decode dispatch. Free
state exactly once after the GPU finishes reading it.

For repeated cancellation tests, fill the pool, cancel half the requests, admit
new requests, and check that no new request sees old tokens. Set a cache budget
small enough to force waiting/rejection deterministically without relying on OOM.

## 5. Share complete prefixes (4 hours)

Implement exact-token whole-page reuse for W5. Use immutable shared full blocks
and private suffix blocks. Include model identity, positions, and precision in
the prefix key. Test identical prefixes followed by divergent suffixes, owner
cancellation, and different prefixes of the same length.

P2's sample exercises refcounts but does not validate content hashes or copy KV.
Those are student extensions. Do not enable shared partial-page append until
copy-on-write has its own correctness tests.

## 6. Replay a controlled mixture (6 hours)

Mix W1 and W3 at a fixed declared proportion, for example 80%/20%. Freeze three
seeded open-loop arrival traces from a measured baseline capacity; replay the
same traces for all policies. Keep output lengths and EOS handling fixed. Also
run closed-loop concurrency to locate saturation. Use the common goodput rules.

Sweep page sizes 8/16/32 and chunk sizes only in a small, feasible matrix. Plot
allocated versus useful KV, goodput versus offered load, and within-request p95
ITL versus chunk size. Annotate one timeline with prefill, decode, gather, and CPU
scheduling. Repeat one decisive W2/mixed comparison on 32B.

**Troubleshooting:** monotonically growing reserved memory may be caching allocator
behavior; growing ownership counts after terminal requests indicates a leak.
Improved average ITL with worsening p95 suggests uneven service or starvation.

## 7. Submit and defend (4 hours)

Submit the request state diagram, ownership invariants, cancellation/prefix checks,
raw timestamp traces, memory curve, and one explained policy decision. Required
completion includes continuous batching, bounded admission, chunking, complete-page
reuse, and a measured efficient attention adapter.

**Defense:** Why can smaller chunks improve ITL while reducing throughput? At what
point can a cancelled request's physical blocks safely be reused?
