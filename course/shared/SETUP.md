# DGX Spark laboratory setup

**DGX Spark is the default measurement platform for this course.** Derive the
expected work, check correctness on small fixtures, then measure the GB10 GPU.
CPU fixtures are optional mathematical/state checks; CPU timing is not an inference
performance baseline. Chapters 2 and 5 also contain untimed simulations and cost
models, which must be distinguished from measured GPU results.

## Use a Spark-compatible CUDA environment

All Python dependencies for the supplied scripts, plots, checkpoint tools, and
tests are declared in [pyproject.toml](../pyproject.toml). From `course/`, create
a dedicated Spark environment with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
export UV_PROJECT_ENVIRONMENT=.venv-spark
uv sync --python 3.12
source .venv-spark/bin/activate
```

On Linux, the project selects PyTorch's CUDA 13.0 index, including ARM64 wheels
for Spark, using [uv's PyTorch source configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
`uv sync` creates `uv.lock`; retain it with your experiment revision and use
`uv sync --locked` to reproduce that dependency resolution. The Python baseline
is 3.11–3.14; Python 3.12 is the default setup above. Run the CUDA preflight below
to verify the selected build against your installed driver and GB10.

You can also keep an existing working CUDA-enabled PyTorch environment provisioned using
the [NVIDIA Spark software guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
and [NVIDIA's Spark SGLang recipe](https://build.nvidia.com/spark/sglang/instructions).
Record the container digest or environment lock, driver, CUDA runtime, and actual
PyTorch build. Follow the recipe's current compatibility requirements before
pinning the environment. A generic CPU wheel does not enable GB10 execution.

To install the declared dependencies into that already activated environment,
use `uv pip install --python "$(command -v python)" -r pyproject.toml --no-sources`.
This uses its installed compatible PyTorch build instead of requiring the
project's CUDA index. Do not add `--upgrade` when retaining a provisioned stack.

Activate your chosen environment before using any `python` command below. An older
course `.venv` containing `torch ... +cpu` is a correctness-only environment;
do not use it to measure these labs. This update does not replace your installed
PyTorch or change GPU drivers.

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.ones(16,device="cuda"); print((x+x).sum().item())'
```

No separate pytest dependency is needed: the suite uses Python's `unittest`.
For the P4 CuTe implementation, add the optional kernel toolchain with
`uv sync --extra kernels` in the dedicated environment. Match the installed DSL
release, examples, CUDA toolkit, and driver using the
[CuTe quick start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html);
the DSL's CUDA 13 requirements can differ from PyTorch's CUDA 13.0 requirements.
SGLang, FlashAttention, FlashInfer, and LLM Compressor are external comparison
stacks: install the selected release in its documented environment when that
project calls for it. GPU drivers, Nsight tools, and multi-host networking are
system prerequisites and are not installed by `pyproject.toml`.

The experiment preflight checks CUDA availability and launches an actual GPU
operation. It records the real GPU name, compute capability, runtime/build,
visible device count, and reported memory. It does not relabel an H100 as Spark.
The measured exercises accept `--device cuda:0` and **reject CPU fallback**.

## First measured experiment

Start with a model-derived BF16 GEMM and a bandwidth calibration on Spark:

```bash
python chapters/03_performance_model/code/experiment.py --device cuda:0 --out results/spark-first-roofline --repeats 3
```

This measures Qwen3-8B Q-projection dimensions, K=N=4096, with no checkpoint
download. Calibration precedes six held-out M values. The copy experiment uses
128 MiB buffers; inspect residency and effective traffic before treating the rate
as a DRAM limit. Open `manifest.json`, `prediction.json`, `results.csv`, and
`prediction.svg`. A prediction error is a result to explain, not a reason to
retroactively alter the original prediction.

Spark uses unified LPDDR memory. Device-visible capacity is not all free model
capacity: CPU applications, the OS, copies, KV, and workspaces share resources.
Record concurrent GPU/CPU activity and memory pressure. Do not change clocks or
terminate unrelated processes during course runs. See the
[hardware specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

## Timing boundaries

Single-GPU experiments use CUDA events after ten warmup calls. The end event is
synchronized before reading elapsed time. Tensor fixtures, host transfers,
correctness assertions, scalar reads, and figures stay outside timing. Multi-kernel
PyTorch references can include GPU idle gaps caused by host dispatch within the
event interval; their results are not isolated fused-kernel latency.

The GEMM exercise uses BF16. Model/attention/quantization mathematical references
use explicitly documented FP32 on CUDA with TF32 disabled. An FP32 numerical
reference must not be compared with BF16 as an unexplained speed difference.
Model and serving experiments retain their separate boundaries from
[PROTOCOL.md](PROTOCOL.md); per-token diagnostic synchronization is not a serving
measurement policy.

Each experiment saves source hashes, assumptions/predictions, raw CSV, summary,
and SVG/PNG figures. Choose a new output directory per run. Plot points are
medians and whiskers are observed ranges, not confidence intervals.

## Two-GPU experiments: P7 and P8

**One Spark has one GB10 GPU.** Two processes on it do not provide TP=2 hardware.
Use two connected Sparks, one rank per host, or the prepared two-GPU rental
already allocated in the syllabus. Follow NVIDIA's linked Spark networking/NCCL
instructions from its playbooks, record actual transport/topology, and copy the
same course revision and environment to both hosts.

For two Sparks, replace `<spark-0-ip>` with the reachable address of the first
host. Use unique hostnames, a reachable rendezvous port, and the network interface
chosen by the validated NCCL setup. Launch both commands on their respective hosts.

Spark 0:
```bash
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=<spark-0-ip> --master_port=29500 chapters/07_tensor_parallelism/code/experiment.py --backend nccl --out results/p07-spark-pair --repeats 3
```

Spark 1:
```bash
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=<spark-0-ip> --master_port=29500 chapters/07_tensor_parallelism/code/experiment.py --backend nccl --out results/p07-spark-pair --repeats 3
```

Replace the script with P8's `code/experiment.py` and choose a fresh output path
for handoff measurement. On a genuine two-GPU rental host use
`--standalone --nproc_per_node=2` instead of the two-host launcher options.
Only rank 0 writes results; both ranks record their actual GPU/host identity.
Distributed timing uses synchronized wall durations at the collective/handoff
boundary, not subtraction of clocks across hosts.

Do not substitute two local Gloo processes for the measured experiment. With one
Spark, complete the untimed ownership/sharding checks and keep the distributed
measurement explicitly pending until the second GPU session.

## Optional correctness fixtures and prerequisites

The small `code/lab.py` checks may run on CPU, including Gloo protocol checks
where explicitly requested. They establish math/state correctness only. The
test suite runs untimed fixtures and simulations on CPU; CUDA and two-GPU tests
are explicitly skipped unless the corresponding hardware is present. An `OK`
with skips does not validate the skipped measurements.

```bash
python -m unittest discover -s tests -v
```

Know matrix multiplication, tensor strides, Python/PyTorch, probability, and basic
GPU execution. Before P4, understand warps, shared memory, synchronization, and
CuTe layouts. Use [CS336](https://cs336.stanford.edu/) for architecture/resource
preparation, and the pinned [CuTe setup guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
for the architecture-specific kernel lab. A Hopper kernel does not automatically
run on GB10.

## Model identity and full projects

Use the [P1 checkpoint workflow](../chapters/01_reconstruct_qwen3/checkpoint_workflow.md)
to audit local safetensors and compare selected full-vocabulary logits. Install
the dependencies from `pyproject.toml`, which include Transformers, safetensors,
Accelerate, and Hugging Face Hub (including the `hf` command).
Resolve and record immutable model/tokenizer revisions before downloading weights:

```bash
python -c 'from huggingface_hub import HfApi; a=HfApi(); print({m:a.model_info("Qwen/"+m).sha for m in ["Qwen3-8B", "Qwen3-32B"]})'
export QWEN8_REV='<8B commit hash>'
export QWEN32_REV='<32B commit hash>'
hf download Qwen/Qwen3-8B config.json --revision "$QWEN8_REV" --local-dir models/qwen3-8b
hf download Qwen/Qwen3-32B config.json --revision "$QWEN32_REV" --local-dir models/qwen3-32b
```

After capacity planning, omit `config.json` to download each complete snapshot.
Use [hf CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
for cache management. Render non-thinking prompts once, save exact IDs, and run
32B reference/custom models sequentially. Do not commit weights.

Every chapter retains its full tutorial, measurable goals, background, references,
and standalone entry experiment. P2/P5 model-only runs are preparation for their
Spark serving/speculation measurements, not substitutes for them. Students extend
the references into the evolving engine as required by the original syllabus.
Use [SELF_STUDY.md](SELF_STUDY.md), [PROTOCOL.md](PROTOCOL.md), and
[REPORT.md](REPORT.md) for navigation, controlled experiments, and assessment.

## Engine interfaces carried across projects

| Owner | Interface contract |
|---|---|
| P1 model | Explicit token positions and processed cache length; return logits and updated KV |
| P2 runtime | Requests own page tables, processed length, pending token, and emission timestamps |
| P3 modeling | Component rows record phase, shape, FLOPs, bytes, timing boundary, and manifest ID |
| P4 attention | Explicit Q/KV layout and ragged lengths; unsupported shapes use a declared reference path |
| P5 speculation | Commit the accepted prefix; rejected positions become unreachable; correction/bonus stays pending until processed |
| P6 precision | Serialized format, actual execution backend, and KV dtype are separate fields |
| P7 distribution | State records rank ownership and sharding axes |
| P8 handoff | Destination acknowledges installed state before source ownership is released |
