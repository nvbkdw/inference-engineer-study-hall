# RMSNorm optimization review

The first optimization for [rms_norm.py](rms_norm.py) is to preserve the known
width and row stride when passing tensors to CuTe. This enables vectorized input
loads and output stores. A matched synthetic benchmark on NVIDIA GB10 measured
**9.25 → 7.20 µs**, a **1.29× speedup**, for 16,384 rows of width 128.

This note records the review and its experiments, including a follow-up with
compact argument packing in the public function. Wide-row layout changes,
reduction changes, and launch tuning need separate comparisons before adoption.

## Current memory-access pattern

Let `R` be the number of flattened rows and `D` the row width, both in elements.
Input `x` and output `y` have shape `[R, D]`; weight `w` has shape `[D]`. Storage
is BF16, or 2 bytes per element. Reduction and scaling use FP32:

```text
sum_r   = sum(x[r, c]² for c in 0..D-1)
scale_r = rsqrt(sum_r / D + eps)
y[r, c] = BF16(x[r, c] * scale_r * w[c])
```

Each thread block owns four rows. Each 32-thread warp owns one row. Define
`K = D / 32`; the current kernel requires `D` divisible by 32. Lane `l` accesses:

```text
column(l, j) = l * K + j,    j = 0..K-1
```

For `D = 128`, each lane owns four consecutive BF16 elements:

| Lane | Columns | Bytes per lane |
| --- | --- | ---: |
| 0 | 0–3 | 8 |
| 1 | 4–7 | 8 |
| 2 | 8–11 | 8 |
| 31 | 124–127 | 8 |

Two properties matter. **Vectorization** combines a thread's adjacent elements
into a wider memory instruction. **Coalescing** combines addresses across a
warp into memory transactions. Contiguous elements within one thread do not
guarantee efficient transactions across the warp.

With scalar 2-byte loads at `D = 128`, adjacent lanes access addresses 8 bytes
apart. For an aligned row, one warp instruction requests 64 useful bytes spread
across eight 32-byte sectors. A single 8-byte load per lane requests the entire
256-byte row using those same eight sectors. This reduces memory instructions
and repeated sector requests; it does not imply an equal reduction in DRAM
traffic because caches can serve repeated requests. See NVIDIA's
[description of coalesced accesses](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

## 1. Preserve width, stride, and alignment information

In the original benchmark, argument packing used the shared `_tensor` wrapper
in [utils.py](utils.py):

```python
from_dlpack(x.detach(), assumed_align=64).mark_layout_dynamic(
    leading_dim=x.ndim - 1
)
```

For a two-dimensional tensor, this keeps the last stride equal to one but makes
the row stride dynamic. Although `width` is a separate compile-time constant,
the compiler does not automatically recover the tensor's known row stride from
it. An aligned allocation therefore does not prove that every row starts at a
vector-aligned address.

Machine-code inspection at `D = 128` confirmed the consequence:

| Per-thread operation | Original wrapper | Compact layout wrapper |
| --- | --- | --- |
| Input load | Four `LDG.E.U16` | One `LDG.E.64` |
| Weight load | One `LDG.E.64` | One `LDG.E.64` |
| Output store | Four `STG.E.U16` | One `STG.E.64` |

The following argument-packing fragment was tested with contiguous, 64-byte
aligned tensors. `x2d` and `y2d` have shape `[R, D]`; the existing device, dtype,
shape, and epsilon checks still apply:

```python
args = (
    from_dlpack(x2d.detach(), assumed_align=64)
        .mark_compact_shape_dynamic(mode=0),
    from_dlpack(weight.detach(), assumed_align=64),
    from_dlpack(y2d.detach(), assumed_align=64)
        .mark_compact_shape_dynamic(mode=0),
)
```

Only the row count remains dynamic. Width and row stride stay known, and the
weight layout is static. This allows CuTe to vectorize the existing `.load()`
and store expressions without an explicit copy atom. It also simplifies address
calculations, so the measured gain cannot be attributed to vectorization alone.

An integrated fast path needs explicit contiguity and pointer-alignment checks.
`assumed_align=64` describes an existing property; it does not align a tensor.
Offset views can violate it. Unsupported tensors need a fallback that represents
their actual strides and alignment, or an explicit rejection. The current
wrapper's unconditional alignment assumption is not a safe fallback for every
offset view.

Keep the fast path separate from the shared wrapper used by other kernels.
Its compilation-cache key must distinguish the packing scheme and static width
from the existing dynamic-layout signature. They have different argument
representations and cannot share a compiled entry.

### Measured result and timing boundary

| Property | Recorded value |
| --- | --- |
| Workload | Synthetic `[16384, 128]`; no model checkpoint |
| GPU | NVIDIA GB10, compute capability 12.1 |
| Software | CuTe DSL 4.2.1; PyTorch 2.14.0+cu130; cuda-python 13.4.1 |
| Precision | BF16 input/output; FP32 arithmetic |
| Epsilon / seed | `1e-6` / `2026` |
| Warmup | 10 eager calls and 10 graph replays per variant |
| Samples | 15 paired repeats, alternating variant order |
| Calls per sample | 300 kernel calls captured in a CUDA graph |
| Clock | CUDA events around graph replay; end event synchronized |
| Cache policy | Reused inputs, weights, and preallocated outputs; no eviction |
| Original median | 9.2517 µs per call |
| Compact-layout median | 7.1987 µs per call |
| Speedup | 1.2852×; paired-repeat bootstrap 95% interval: 1.1660–1.2881× |

The interval uses 10,000 resamples with seed 2026. Timing includes GPU kernel
execution and graph scheduling. It excludes compilation, graph capture,
allocation, argument packing, Python dispatch, and correctness checks. This is
a cache-resident kernel comparison, not an end-to-end model or DRAM-bandwidth
measurement. Desktop GPU processes remained active and clocks were unchanged.

Both variants passed an independent FP64 oracle at row counts 1, 5, and 16,384,
using `rtol=4e-3` and `atol=1e-5`. The small cases also exercise partial blocks.

### Why the public benchmark can show no speedup

A follow-up tested the updated public function at shape `(1024, 16, 128)`,
which flattens to the same `[16384, 128]` workload. Its wrapper used
`assumed_align=128` and `mark_compact_shape_dynamic(mode=0)` for all tensors.
The comparison used the same current kernel with the original dynamic packing.
Each timing boundary used 15 paired repeats of 200 calls on GB10:

| Timing boundary | Dynamic packing | Compact packing |
| --- | ---: | ---: |
| Public call, no CUDA graph | 22.66 µs | 23.96 µs |
| Preallocated output and prepared arguments, no graph | 12.14 µs | 8.20 µs |
| Preallocated output and prepared arguments, graph replay | 9.28 µs | 6.86 µs |

The compact kernel still generated `LDG.E.64` and `STG.E.64` instructions.
Graph timing showed a 1.35× improvement, while public-call timing showed a small
regression. All compared paths passed the FP64 oracle on the full fixture.

The public benchmark in that experiment called Python code repeatedly with
`use_cuda_graphs=False`. Each call validated inputs, allocated output, constructed
views and DLPack wrappers, looked up the compilation cache, and launched the kernel. CUDA events
measure elapsed GPU-stream time, which can include idle gaps while the CPU
prepares the next launch. A faster kernel can leave a longer idle gap when
submission limits throughput. These timings therefore need not improve with
kernel execution time. CUDA-graph replay removes repeated host submission work;
see [PyTorch's CUDA-graph explanation](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).

Also, `assumed_align=128` means a **128-byte-aligned base address**, not a
128-bit load instruction. At width 128, each lane owns only four BF16 elements:
8 bytes, or 64 bits. Increasing the base alignment from 64 to 128 bytes does not
increase that fragment size or the alignment of every lane's address.

Use kernel timing to assess instruction/layout changes and public-call timing
to assess the calling interface. Reusing output buffers and prepared arguments
addresses the latter. The saved
[follow-up artifacts](../../../../results/rmsnorm-timing-diagnosis-3ovapzl2/)
include raw samples, source snapshots, timing metadata, and disassembly. This
remains a synthetic reused-buffer experiment, with no model-level speedup claim.

### Standard kernel benchmark policy

The current [benchmark utilities](utils.py) now use CUDA-graph replay for both
`benchmark_cuda` and `benchmark_cuda_cold`. Allocate input, output, and scratch
buffers before invoking the helper, or provide a `workspace_generator` that
allocates them once and returns `cute_testing.JitArguments` for the callable.
The callable must reuse those buffers and enqueue work on the current stream.
The factory runs on the capture stream, outside capture and timing, so it can
also compile a kernel and prepare stream-specific arguments.

`benchmark_cuda` warms each workspace, captures `inner` calls once, and reuses
the graph for every repeat. CUDA events surround each replay and observations
remain milliseconds per call. The cold-cache helper captures a single call
and uses Triton's eviction buffer before timing each replay. Neither helper
falls back to timing Python kernel dispatch.

RMSNorm's benchmark follows this contract for both backends. CuTe reuses its
compiled kernel, argument pack, and BF16 output. The Torch baseline preallocates
FP32 intermediates and BF16 output; conversions remain timed GPU operations.
The manifest records `cuda_graphs=true`, `preallocated_buffers=true`, and the
new timing boundary. Historical public-call observations above retain their
original meaning and should not be compared as if they used the new boundary.

## 2. Use short vector chunks for wide rows

The course's [model dimensions](../../../../shared/performance.py) use hidden
widths 4096 for Qwen3-8B and 5120 for Qwen3-32B. Under the current layout,
adjacent lanes start `2K` bytes apart: 256 bytes and 320 bytes, respectively.
Even a 16-byte vector load then touches a separate 32-byte sector for each lane.

A candidate layout assigns `V = 8` consecutive BF16 elements to each lane and
then advances the whole warp to the next chunk:

```text
column(l, t, v) = t * (32 * V) + l * V + v

l = 0..31
v = 0..7
t = 0..D/(32*V)-1
```

At `D = 4096`, lane 0 accesses columns 0–7, then 256–263, and so on. Lane 1
accesses columns 8–15, then 264–271. Each vector instruction covers 512
consecutive bytes across the warp: 16 fully used sectors instead of 32
half-used sectors. These are address-based predictions of sector requests, not
measured hardware counters.

Input, weight, and output must use the same mapping. This version requires
`D % 256 == 0` and suitable row alignment. Width 128 needs a smaller vector;
other widths need another supported layout or explicit masking.

A temporary chunked variant compiled and passed the FP32 reference check at
width 4096. Its machine code contains 128-bit loads and stores. However, it also
changes the local reduction, so it is not an isolated layout comparison. No
speedup over the original wide kernel has been established: the original
width-4096 specialization crashed during compilation in the installed stack.

## 3. Reduce dependency length and register pressure

The current local sum uses one accumulator:

```python
for i in cutlass.range_constexpr(thrX.shape[0]):
    squared_sum += v[i] * v[i]
```

This creates a chain of `K` dependent updates before the five warp-shuffle
stages. At widths 4096 and 5120, `K` is 128 and 160. Several independent
accumulators could shorten each chain, followed by a final local reduction.
A vector reduction is another candidate, but its generated instruction order
must be inspected rather than assumed to form a balanced tree.

The temporary width-4096 chunked variant used **215 registers per thread**,
with no local memory reported by the binary's resource metadata. Keeping many
FP32 values live can limit the number of resident warps even without spilling.
Multiple accumulators can increase this pressure, so they are not automatically
a win.

Another candidate assigns multiple warps to each wide row. This reduces values
per thread but adds a cross-warp reduction and synchronization. Tune warps per
row and rows per block together. For small row counts, the current four-row
block also leaves entire warps idle. One-row blocks can distribute several rows
across more multiprocessors, but a single row still provides only one block
unless the algorithm is split further.

The expected benefit depends on whether dependency latency, register pressure,
memory requests, or launch overhead dominates. Changed reduction order also
requires renewed numerical checks against the FP64 oracle.

## 4. Lower-priority changes

**Shared-memory weights.** Four warps reuse the same vector, so cache reuse is
plausible. Weight sizes are only 256 bytes, 8 KiB, and 10 KiB at widths 128,
4096, and 5120. Explicit staging adds shared-memory traffic and synchronization.
The existing TODO should follow profiling evidence of a weight-load bottleneck.

**Host overhead.** The public function allocates output and constructs DLPack
arguments on each call. Reusing output buffers and prepared arguments, or using
CUDA graphs in the runtime, may matter for small workloads. These change the
public-call timing boundary and need their own comparison. The kernel-only
speedup above does not quantify them.

**Compile-time prints.** Python `print()` inside a CuTe JIT function runs during
compilation. Removing it improves diagnostic output but does not speed up the
generated GPU kernel. This differs from runtime `cute.printf()`; see the
[CuTe 4.2.1 debugging documentation](https://docs.nvidia.com/cutlass/4.2.1/media/docs/pythonDSL/cute_dsl_general/debugging.html).

## Validation limits and evidence

The original experimental RMSNorm test run passed three test methods, then
terminated with a segmentation fault. An isolated compilation of the original
width-4096 kernel reproduced a crash inside the compiler pass pipeline. This
does not establish a GPU memory-access fault or its root cause. The full test
suite did not pass, and original wide-row performance remains unmeasured.

For `R` rows, the logical minimum traffic is `4RD + 2D` bytes: one BF16 input
read, one BF16 output write, and one weight-vector read. Actual traffic depends
on caches and memory transactions. Vectorization reduces instructions and can
improve sector utilization without changing this logical minimum.

The highest-priority implementation is a guarded compact-layout fast path.
Wide-row chunking, local reduction, and launch geometry should then be compared
separately. Any course integration should follow
[the experiment protocol](../../../../shared/PROTOCOL.md), including correctness
checks, predictions saved before timing, raw observations, and a declared cache
policy. No real-model inference measurements were made in this review.

The review's local artifacts are in
[`results/rmsnorm-review-8uq_6bzz/`](../../../../results/rmsnorm-review-8uq_6bzz/):

- [`prediction.json`](../../../../results/rmsnorm-review-8uq_6bzz/prediction.json): hypotheses saved before measurement.
- [`manifest.json`](../../../../results/rmsnorm-review-8uq_6bzz/manifest.json): environment, timing scope, and measured source hashes.
- [`compare_128.py`](../../../../results/rmsnorm-review-8uq_6bzz/compare_128.py): matched comparison and correctness checks.
- [`results.csv`](../../../../results/rmsnorm-review-8uq_6bzz/results.csv) and [`summary.json`](../../../../results/rmsnorm-review-8uq_6bzz/summary.json): raw observations and statistics.
- [`matched-dynamic-128-0.sass`](../../../../results/rmsnorm-review-8uq_6bzz/matched-dynamic-128-0.sass) and [`matched-compact-128-0.sass`](../../../../results/rmsnorm-review-8uq_6bzz/matched-compact-128-0.sass): generated machine instructions.
- [`blocked_variant.py`](../../../../results/rmsnorm-review-8uq_6bzz/blocked_variant.py) and [`compact_blocked-4096-0.sass`](../../../../results/rmsnorm-review-8uq_6bzz/compact_blocked-4096-0.sass): temporary wide-row variant and resource evidence.

These results are local, ignored experiment artifacts and may be absent in a
fresh clone. The manifest identifies the measured source snapshot; later
source edits are not automatically covered by its measurements.
