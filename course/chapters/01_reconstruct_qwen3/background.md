# Background: a token's computation and state

Let B be batch size, T new tokens, S cached tokens after this call, D residual
width, I MLP width, Hq query heads, Hkv KV heads, R head dimension, L layers, and
V vocabulary size. Stored linear weights are `[out_features, in_features]`.

## Derive the decoder block

For each token, RMSNorm computes `x / sqrt(mean(x²) + eps) * gamma` over its last
dimension. Accumulate the mean square in FP32. Normalization rescales a vector;
it does not subtract its mean. In a pre-norm residual block, attention receives
normalized input while the skip connection retains the original residual.

Starting from `X[B,T,D]`, Q has shape `[B,T,Hq*R]`, K/V `[B,T,Hkv*R]`.
Reshape to heads; normalize Q and K over R with learned weights. Apply RoPE to Q/K
using absolute token positions. RoPE rotates pairs of coordinates, allowing dot
products to depend on relative position. Use the checkpoint's split-half pairing
and frequency convention. Cache rotated K and unrotated V. These details are
specified by the [Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py).

Each group of `Hq/Hkv` query heads shares K/V. A dense oracle repeats KV heads;
an efficient kernel accesses shared storage directly. Scores are `Q Kᵀ / sqrt(R)`.
Mask future keys before softmax, then multiply probabilities by V. Concatenate
query-head outputs and project `[Hq*R,D]` back into the residual stream.

The MLP is `down(silu(gate(z)) * up(z))` with two `[D,I]` projections and one
`[I,D]` projection. Both attention and MLP add their result to a residual. A final
RMSNorm and vocabulary projection produce logits; softmax converts logits to a
distribution only when needed for sampling or evaluation.

## The architecture trap

8B uses `(L,D,I,Hq,Hkv,R)=(36,4096,12288,32,8,128)`; 32B uses
`(64,5120,25600,64,8,128)`. Both have V=151936 and untied embeddings. In 32B,
Q's width is 8192 while D is 5120. Never infer R as D/Hq.
[8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json),
[32B config](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json).

For the bias-free dense structure, count parameters:

```text
P = L(2 D Hq R + 2 D Hkv R + 3 D I + 2 D + 2 R) + 2 V D + D
```

The first two terms count Q/O and K/V matrices; then MLP, two block norms, Q/K
norms, two vocabulary matrices, and final norm. The tiny sample intentionally
uses `D != Hq*R` and verifies the formula against actual parameter tensors.

## Cache semantics before speed

After processing p tokens, the cache represents exactly positions `0..p-1`.
For a new chunk of t tokens, query j has absolute position `p+j`; key k is visible
iff `k <= p+j`. For p=3 and t=2 the rows of the allowed mask are:

```text
1 1 1 1 0
1 1 1 1 1
```

A top-left triangular mask would hide valid prefix keys. A sampled next token
is pending: it is in the output history but has no KV until processed.
That distinction will govern speculation and P/D handoff later.

For element size b bytes and request lengths Si,
`M_KV = 2 L Hkv R b sum(Si)`. BF16 yields 147456 bytes/token for 8B and
262144 for 32B. At 2048 tokens these are 0.28125 and 0.5 GiB per request.
Weights are constant in context length; live KV grows linearly. Cached decode
avoids prefix projections but still reads growing attention state.

## Numerical reasoning

A readable continuation is weak evidence of correctness. Compare full-vocabulary
logits and locate the first divergent layer. Report maximum absolute error and
relative L2 error; inspect near-tied top logits separately. Start tiny FP32 at
`rtol=1e-4, atol=1e-5`. Establish BF16 tolerance empirically against the chosen
reference/backend, rather than widening it until a comparison passes.
