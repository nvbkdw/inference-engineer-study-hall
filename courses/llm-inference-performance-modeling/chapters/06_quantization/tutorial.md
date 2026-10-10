# Tutorial: make a quality-constrained serving decision

## 1. Implement quantization numerics (4 hours)

```bash
python chapters/06_quantization/code/lab.py
```

Inspect `quantize`, `dequantize`, `pack`, and `unpack`. Reconstruct affine and
symmetric quantization on a hand-written row containing zero, positive/negative
values, and an outlier. Test every nibble code, all-zero groups, exact endpoints,
and a rounding tie. The sample requires K divisible by group size and an even
element count; reject unsupported shapes or add explicitly masked padding.

Extend the sample with per-tensor and per-channel scales. Compare G=32 and 128.
Record code bytes, scale bytes, zero-point bytes if used, and padding separately.
Its printed output errors come from random tensors and establish no Qwen quality.

## 2. Connect error to real activations (4 hours)

Capture a few attention/MLP inputs from the P1 reference on real calibration text.
Quantize representative weight rows and compute weight relative L2, maximum error,
and layer-output relative L2. Compare equal-size weight errors in different channels.
Explain the difference using `delta_Y = X E.T` and activation magnitudes.

For numerical experiments, reconstruct BF16 weights before GEMM if convenient;
label these runs numerical-only. Predict ideal storage and whether the P2
bottleneck could benefit from lower weight traffic.

## 3. Produce an actual calibrated checkpoint (7 hours)

Select one maintained AWQ/groupwise W4A16 recipe compatible with your pinned
inference backend. Start from [LLM Compressor](https://github.com/vllm-project/llm-compressor)
and the [backend quantization guide](https://docs.sglang.io/docs/advanced_features/quantization).
Record quantizer commit, exact recipe file, execution backend, group size, excluded
layers, scale/zero-point formats, and calibration seed. Use the recipe from the
same release rather than assuming all AWQ checkpoints share a packing layout.

Create a manifest of about 128 calibration sequences with at most 512 tokens each,
including dataset revision, IDs, formatting, and checksums. Keep evaluation IDs
disjoint. Run conversion first for 8B, then 32B with the same recipe. Save conversion
logs and resolved configuration beside the resulting checkpoint.

Load the output in the selected server, run a tiny numerical check, inspect
quantized modules and the profiler's actual GEMM kernel names, and measure loaded
memory. Confirm packed weights reach the execution backend. A smaller file or
fake-quantized tensor is insufficient evidence. Time-box platform setup; follow
the syllabus's H100 fallback or document a supported FP8 alternative if W4A16 is
unavailable. A format change requires a new numerical/storage derivation.

## 4. Freeze and run quality evaluation (5 hours)

Select held-out text for teacher-forced mean NLL, a pinned 512-example GSM8K subset,
and a compact extraction fixture with exact schema checks. Freeze IDs, prompt
template, decoding/stopping rules, answer parser, and the evaluation harness commit.
Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as
the local evaluation adapter; retain the exact task configuration and command.

Declare a budget, e.g. <=1% relative NLL increase and <=2 percentage points accuracy
loss, as a course choice. Evaluate BF16 and quantized outputs on paired examples;
bootstrap paired score differences or use a justified paired interval. Report
pass/fail/inconclusive relative to the budget, and expand samples when necessary.
NLL uses shifted next-token labels and token-weighted aggregation, excluding padding.

## 5. Measure execution and serving capacity (6 hours)

Begin the measurement workflow on DGX Spark using
[the standalone CUDA experiment](standalone.md). Its `code/experiment.py`
uses CUDA events and records the actual GPU identity. CPU `lab.py` checks
are optional correctness preparation; they supply no timing baseline.
The remaining steps below extend the reference workload to the full project.

For both model sizes, compare BF16 and W4A16 at fixed batch/context, at the largest
feasible batch meeting the frozen SLOs, and under identical offered-load traces.
Keep KV dtype, prompts, output policy, and cache condition fixed. Start with W1/W2,
then use W3 to test the changing weight/KV balance.

Record serialized bytes, loaded packed bytes, scales, reserved memory, loading
peak, workspace, TTFT, completion latency, goodput, and quality. Use P2 to predict
the speedup from reduced weight traffic, adding unpack/scale/compute costs supported
by the trace. Explain why the observed result differs from an ideal fourfold byte
reduction. Repeat selected points on 32B rather than duplicating the entire sweep.

## 6. Revisit speculation (4 hours)

Run BF16 target/BF16 draft, BF16 target/quantized draft, and quantized target with
the best available draft. Add target-only baselines for each target precision.
Use the same real prompt subset and draft lengths as P5. Compare acceptance,
draft/catch-up time, verification time, E[A], and full-request latency. Determine
whether faster target steps leave less opportunity for speculation.

## 7. Submit (3 hours)

Provide quantization/packing checks, real calibration provenance for both models,
backend execution evidence, paired quality intervals, memory/performance curves,
and speculation interaction. A full-quality evaluation or packed-GPU benchmark
has not been completed merely because `lab.py` passes.

**Defense:** Why does a four-bit file not imply fourfold speedup? What distribution
does exact speculative sampling preserve after the target is quantized?
