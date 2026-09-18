# Background: representation, numerical error, and execution

Quantization replaces a set of floating-point values with a smaller codebook plus
metadata. Memory can fall without latency falling: the backend may unpack slowly,
reconstruct full weights, use an inefficient kernel, or leave KV as the bottleneck.
Treat serialized size, loaded representation, and executed math as separate facts.

## Construct a quantizer

For group g and scale s>0, an affine quantizer is
`q=clip(round(x/s)+z,qmin,qmax)` and reconstruction is `x_hat=s*(q-z)`.
Scale determines spacing; clipping limits the representable range. Rounding
introduces at most s/2 error for values inside that range; clipping can exceed it.
Handle all-zero groups without dividing by zero.

The sample uses symmetric signed INT4 values -7..7, scale `max(abs(group))/7`, and
round-to-nearest with PyTorch's tie convention. It reserves -8 in quantization,
but supports all -8..7 codes in its packing exercise. Packing stores q+8 in a
nibble, low nibble first. This teaching layout is not a production AWQ ABI.

For weight `[N,K]`, per-tensor scale uses one group; per-output-channel scale uses
each row; groupwise scale partitions each row along K. Smaller groups adapt to
local range but cost more metadata. With group size G and s_b bytes/scale,
ideal storage is `N*K/2 + (N*K/G)*s_b`, before zero points, padding, unquantized
layers, and workspaces. The byte-packing sample uses FP32 scales for inspection.

## Weight error is not output error

For linear output Y=XWᵀ and quantized W_hat=W+E, error is `X Eᵀ`. Thus the same
weight error E can matter very differently for different activation directions.
Outliers in activation channels can amplify otherwise small errors. The
[AWQ paper](https://arxiv.org/abs/2306.00978) motivates activation-aware treatment;
the numerical min/max reference does not implement AWQ calibration/search.

Layer-output error, model NLL, and task accuracy answer different questions.
Teacher-forced mean NLL is total negative log probability of held-out target tokens
divided by the number of scored tokens. Shift logits/labels correctly and exclude
padding; a perplexity ratio is not numerically the same as a relative NLL increase.
For accuracy, preserve paired examples and report the difference in percentage
points with uncertainty.

## Where speed can come from

Weight-only W4A16 can reduce streamed weight bytes and allow more concurrent KV.
But dequantization work, scale reads, kernel shape efficiency, and compute type
determine latency. Executing `dequantize(q) @ x` as an ordinary BF16 GEMM demonstrates
numerics, not packed inference. Inspect the actual kernel and loaded representation.

Keep KV dtype fixed during the weight-only comparison. With unchanged KV,
memory saved on weights increases available request capacity but does not reduce
bytes per cached token or P/D handoff size. Long-context performance may therefore
improve less than short-context decode.

## Quality and speculation together

Calibration examples choose quantizer parameters; evaluation examples test the
result and must be disjoint. Declare allowable quality change before tuning.
A 512-example task evaluation may not resolve a two-percentage-point difference;
an interval overlapping the budget is inconclusive, not a pass.

Quantizing the draft changes both draft cost and acceptance. Quantizing the target
changes the distribution the exact verifier preserves. Exact speculation against
a quantized target preserves that target's distribution under its numerical path;
it does not restore the original BF16 target. Compare every speculative condition
to a matching target-only precision baseline.
