# Validation matrix

Validated with **CuTeDSL 4.8.0**, **tensor-layouts 0.3.2**, Python 3.12, and the committed `uv.lock` / `frontend/package-lock.json`. Cross-target compilation runs on Linux aarch64. Local GPU execution uses an NVIDIA GB10; Hopper and tcgen05 kernels are not executed on that GPU.

On 2026-09-22, all **44 Python tests** (including the optional GPU checks) and **11 browser tests** passed. Ruff and Svelte/TypeScript checks passed; the production frontend and Python distributions built successfully. The wheel was installed and checked in an isolated environment without CUDA or CuTeDSL.

## Compiler targets and operation families

| Target | Cases checked against actual CuTe partitions | Storage checks |
| --- | --- | --- |
| `sm_80` | Dense FP16/BF16, K=8/16; tiled replication; scalar copies and `ldmatrix` transpose variants | Register A/B/C thread/value maps |
| `sm_90a` | Dense FP16/BF16, M=64, every N from 8 to 256 by 8; shared/register A; tiled replication; transposed major modes | Shared A/B and register A/C; warpgroup participation |
| `sm_100a` | Dense FP16/BF16, M=64/128, every N from 8 to 256 by 8; shared/tensor A; transposed major modes; single CTA | Shared B; tensor A and C, including FP16 subwords and accumulator column units |

FP16→FP16, FP16→FP32, and BF16→FP32 are included in each family. Tests sample boundary and interior thread/value coordinates across every shape; they do not exhaustively enumerate every element of every configuration. Compiler-produced layout descriptors support static tilings; representative multi-atom tilings are tested. No claim is made for untested target variants, every Blackwell device, or deferred sparse/block-scaled/multi-CTA instructions.

Production snapshots CuTe TV descriptors and evaluates them through the CPU adapter. `tests/compiler_oracle.py` independently builds identity-coordinate tensors and calls CuTe `partition_A/B/C` and `partition_S/D`. The returned coordinates serve as the oracle. This catches layout ordering, nested modes, replication, and cross-thread copy redistribution without reusing the production evaluator as the expected answer.

Tensor-memory expected addresses additionally use the independent NVIDIA [fragment layout definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm100_frag.hpp): M=128 uses all data paths; M=64 uses half subpartitions. FP16 A packs two elements per 32-bit column; FP16 C occupies full accumulator columns. Runtime allocation bases remain unknown. NVIDIA's [PTX matrix instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) defines the instruction and storage semantics.

## Neutrality and diagnostics

- Dense MMA probes are checked for unchanged insertion-block operation counts on all three targets.
- Layout, tensor, copy, MMA, and symbolic-layout probes are checked for unchanged kernel IR.
- PTX instruction sequences are compared with probes enabled/disabled after normalizing compiler specialization symbol names.
- A real global→shared→global GPU copy is checked against host input with probes enabled and disabled. A real 16×8×16 FP16 warp GEMM is checked against independent host multiplication.
- Source spans, repeated labels, nested capture sessions, declared ancestry, unsupported operations, partial failures, empty captures, and reuse of compiled callables are checked.
- Repeated `cute.compile` calls recapture the same driver correctly. In 4.8.0 some ordinary JIT cache hits still retrace; the diagnostic test specifically exercises a compiled callable that bypasses probes.

## Backing tensor coordinates

`tests/compiler_ancestry.py` compiles real local tiles and slices while asserting unchanged insertion-block IR for every probe. Cases include static and CTA-indexed tiles, projected tilers, remainder modes, nested tile chains, slices, constant affine CTA arithmetic, and partial boundary tiles. Independent hand-calculated coordinates verify the serialized transforms and backing element/byte offsets. The original tensor is deliberately unprobed; tests verify automatic recovery and reuse of the same compiler tensor. Gathered tilers retain a backing snapshot with an unsupported mapping diagnostic.

Portable tests cover highlights in both directions, exact tile/slice footprints, aliases, out-of-bounds clipping, unresolved bindings, textual declarations without coordinate inference, invalid/cyclic ancestry, and schema `1.0` compatibility. The browser test changes CTA coordinates, selects local and global cells, follows a slice through two backing tensors, navigates to a backing cell, and verifies that explanations update with the local tensor off-screen. These checks validate logical coordinate ancestry, not runtime allocation addresses or arbitrary producer tracing.

## CPU and interaction checks

Independent hand-calculated row/column-major, hierarchical, composed, zero/negative-stride, alias, swizzle-bit, and dtype-width cases exercise the CPU adapter. Bank checks cover contiguous accesses, same-bank distinct words, broadcasts, inactive lanes, unaligned accesses, duplicate lanes, vectors, and same-word write races. Imported captures are tested while CUDA and CuTe imports are blocked.

The browser suite covers loading, embedded source selection, coordinate explanations, arrow-key movement, linked copy cells, thread selection, nested slicing, swizzle comparison, explicit bank analysis, Hopper/Blackwell operand views, operand switching, and coordinate navigation. Type checking and a production build precede the tests. A distribution smoke check installs the wheel in a separate environment without capture dependencies and checks bundled assets and API playback.

Canvas checks cover sidebar-scoped groups, incoming and outgoing relationships, sibling operands, stable group membership during cell selection, zoom controls, draggable and resizable tensor panels, expanded mode, optional red links, thread/value highlights, and exclusion of unrelated tensors. Resizing tests drag the bottom-right control in both directions, verify that all cells remain available without internal scrolling, and check that arrangement preserves dimensions. A 128×96 tensor renders all 12,288 cells across multiple API batches. A million-by-million tensor shows its full shape, then navigates to and selects its final cell, including real pointer picking at extreme zoom and keyboard movement. Every mapping request remains capped at 4,096 cells. These tests cover full-shape geometry with bounded viewport details; they do not materialize a trillion cells. CPU tests also reach the last window of a billion-by-billion array and reject false coordinate equivalence between MMA matrices and observed fragments.

## Interactive compilation

Workbench API tests compile the bundled copy example for `sm_80` and MMA operand examples for `sm_90a` and `sm_100a` through real subprocesses. They validate source snapshots, downloadable captures, and mapping queries. A separately authored entry uses a custom argument factory, positional and keyword arguments, and a local project import; editing its shape produces a new capture while the earlier run remains queryable.

Lifecycle checks cover syntax errors, errors after a probe, empty observations, bounded output, concurrent submission rejection, cancellation, timeout, and server shutdown. Boundary tests reject foreign Host/Origin headers, cross-site requests, missing session tokens, malformed entry names, oversized source, and execution in read-only mode. Missing compiler support leaves portable playback available.

Browser tests type a kernel in CodeMirror, compile with Ctrl/⌘+Enter, select a captured coordinate, verify its arithmetic/source, download JSON, edit and recompile, inspect the preceding capture after a syntax error, restore the draft after reload, import a `.py` file, and cancel a running job. The wheel includes the editor assets, worker, and starter sources. These tests establish local compilation behavior; they do not claim sandbox isolation or execution correctness for arbitrary user code.

## Remaining external validation

GPU execution of Hopper WGMMA and Blackwell tcgen05 requires matching hardware and a user's complete synchronized GEMM. The Hopper/Blackwell MMA example compiles operand construction only; the Ampere-style warp GEMM example executes in the optional GPU check. Beginner sessions on independently authored kernels have not been conducted; see [the exercise protocol](usability.md). Position-dependent pointer swizzles, symbolic extent bindings, tuple-output compositions, and opaque backing descriptors remain explicit limits rather than guessed mappings.
