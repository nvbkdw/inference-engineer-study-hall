# From a local checkpoint to comparable logits

This workflow turns P1's reference-baseline and weight-mapping steps into concrete
commands. It also supplies a local baseline artifact for later chapters. Run the
small checkpoint first on Spark; use the same workflow for the pinned real snapshots after
checking memory capacity. It does not require a network connection once snapshots
and Python packages are installed.

## 1. Install the course dependencies

The model tools are included in [pyproject.toml](../../pyproject.toml). Complete
the [Spark environment setup](../../shared/SETUP.md) first. In the dedicated
course environment, synchronize the dependencies from the course root:

```bash
UV_PROJECT_ENVIRONMENT=.venv-spark uv sync
```

The authoring validation used Transformers 5.17.0 and safetensors 0.8.0. Record the
installed Accelerate version as well. On a GPU, retain the working PyTorch/CUDA
stack and select model-tool versions compatible with it; don't replace that stack
with CPU wheels. `device_map` loading uses Accelerate to place weights directly.

## 2. Create an offline practice checkpoint

```bash
python chapters/01_reconstruct_qwen3/code/make_fixture.py --out results/p1-loader/tiny
```

This creates a two-layer HF Qwen3 model with random weights, intentionally
`hidden_size != query_heads*head_dim`, and multiple safetensors shards. Its vocabulary
has 101 entries; it is a shape/loader fixture, not a usable language model. The
saved `tokens.json` contains seven prompt IDs and three forced continuation IDs.

## 3. Audit checkpoint headers before allocating weights

```bash
python chapters/01_reconstruct_qwen3/code/run_checkpoint.py --snapshot results/p1-loader/tiny --tokens results/p1-loader/tiny/tokens.json --model-revision local-tiny-v1 --backend custom --audit-only --out results/p1-loader/audit
```

Inspect `inventory.json`: every source tensor must map to exactly one expected
parameter with the right shape. The loader checks shard-index agreement, duplicates,
missing and unexpected tensors, dtype, and total parameter count. It constructs
the destination model on the meta device for this audit; no full weight storage is
allocated. `memory_prediction.json` gives parameter and final logical cache bytes.

The supported custom path is dense, bias-free Qwen3 with SiLU, untied vocabulary
weights, full attention, and default unscaled RoPE. It rejects quantized or scaled-
RoPE configurations rather than silently applying an incorrect implementation.

## 4. Run reference and custom implementations sequentially

```bash
python chapters/01_reconstruct_qwen3/code/run_checkpoint.py --snapshot results/p1-loader/tiny --tokens results/p1-loader/tiny/tokens.json --model-revision local-tiny-v1 --backend transformers --out results/p1-loader/reference
python chapters/01_reconstruct_qwen3/code/run_checkpoint.py --snapshot results/p1-loader/tiny --tokens results/p1-loader/tiny/tokens.json --model-revision local-tiny-v1 --backend custom --out results/p1-loader/custom
python chapters/01_reconstruct_qwen3/code/compare_logits.py --reference results/p1-loader/reference --candidate results/p1-loader/custom --rtol 0.0001 --atol 0.00001
```

The runner defaults to `--device cuda:0`. Use `--device cpu` only for an explicitly optional untimed correctness comparison, never as the performance baseline. Each runner loads one model, performs prompt prefill, then consumes each forced
continuation token with its cache. It saves four complete vocabulary vectors:
after the prompt and after each supplied token. Comparing at identical histories
avoids confusing a near-tie argmax difference with later divergent input histories.
The comparison checks model/config/fixture identity and dtype before numerical
tolerances. Expect four positions, vocabulary 101, and `correctness: passed`.

The custom loader reads one tensor at a time from safetensors and copies it into
preallocated destination storage. It maps the checkpoint's Q/K norms as well as
all projections. The custom model computes only the last-position vocabulary head
in this workflow, although every new token is processed through the decoder.

The runner's wall times are explicitly **cold diagnostic timings**, including
synchronization at each recorded boundary. They are not warmed serving or kernel
benchmarks. Use P3's timing protocol for performance claims. CUDA runs also record
allocated, reserved, and peak allocated device memory; host/loading copies still
need process/system accounting on unified-memory hardware.

## 5. Prepare a real text fixture

Download and pin `Qwen/Qwen3-8B` and `Qwen/Qwen3-32B` using the commands in
[setup](../../shared/SETUP.md). Render one non-thinking prompt and tokenize a short
forced continuation. For example, save the following as `results/make_tokens.py`
and run it after downloading the 8B snapshot:

```python
import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/qwen3-8b", local_files_only=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain why a KV cache speeds up decoding."}],
    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
)
continuation = tokenizer.encode("A KV cache retains", add_special_tokens=False)
Path("results/qwen_tokens.json").write_text(json.dumps({
    "kind": "real text, non-thinking, teacher-forced fixture",
    "prompt_ids": prompt, "continuation_ids": continuation,
}))
```

Keep the source text, tokenizer revision, template settings, and saved IDs in the
full manifest. Before sharing these IDs between model sizes, audit the pinned
tokenizers' mapping and special-token semantics as P5 requires.

## 6. Run the real 8B comparison, then the selected 32B comparison

First run `--audit-only` on each real snapshot to derive its memory budget. Then,
in the GPU environment, run the two model paths sequentially:

```bash
python chapters/01_reconstruct_qwen3/code/run_checkpoint.py --snapshot models/qwen3-8b --tokens results/qwen_tokens.json --model-revision "$QWEN8_REV" --backend transformers --device cuda:0 --dtype bf16 --out results/p1-8b-reference
python chapters/01_reconstruct_qwen3/code/run_checkpoint.py --snapshot models/qwen3-8b --tokens results/qwen_tokens.json --model-revision "$QWEN8_REV" --backend custom --device cuda:0 --dtype bf16 --out results/p1-8b-custom
```

Select a BF16 tolerance using measured reference/backend differences and record
it before accepting the comparison. Pass those numeric values to `compare_logits.py`;
do not reuse tiny FP32 thresholds blindly or inflate them merely to pass.

Repeat with `models/qwen3-32b`, `$QWEN32_REV`, and fresh output directories after
capacity checks. The custom attention is an intentionally simple dense reference:
score memory grows quadratically during prefill and KV heads are repeated. Start
with the short text fixture; the efficient long-context adapter is P2/P4 work.

No real 8B/32B run is implied by a passing tiny-checkpoint test. Preserve the real
comparison artifacts when you execute them and continue with the P1 full tutorial's
chunked/mixed-batch checks and warmed memory/timing experiments.

## References

The model API and local implementation comparison follow the
[Transformers Qwen3 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3).
The loader uses the tensor/header interfaces documented in the
[safetensors API](https://huggingface.co/docs/safetensors/main/en/api/torch).
These reading targets were checked on 2026-09-11; pin code/package revisions for
reproducible execution.
