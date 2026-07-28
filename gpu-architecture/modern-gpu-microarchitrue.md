



## Memory Hierarchy
Global memory, size, bandwidth
constant memory, size
L2 cache, size
shared memory
registers

## Streaming Multiprocessor
SM, count
block (CTA), warp SIMT, registers, 
block cluster
SM resources, occupancy
SMEM size
register size, 

## Shared memory
SMEM bank, bank width

## TMA - Memory copy hardware
cp.copy ptx
swizzle
mbarrier object in SMEM




## Tensor Core - MMA hardware
ptx, mma.sync, tcgen05
TMEM
ldmatrix


# Put It Together

Rows follow the sections above (memory hierarchy → SM → SMEM → TMA → tensor core).
Columns run oldest → newest. Note the last two are **both** "Blackwell" but are different
microarchitectures: SM100 is the datacenter line, SM12x is the GeForce line.

| Spec | Ada Lovelace — L40S | Hopper — H100 SXM5 | Blackwell DC — B200 SXM | Blackwell GeForce — GB10 (DGX Spark) |
| --- | --- | --- | --- | --- |
| **Die / process** | AD102, 1 die, 76.3B transistors, TSMC 4N | GH100, 1 die, 80B transistors, TSMC 4N | GB100, **2 dies** + NV-HBI 10 TB/s link, 208B transistors, TSMC 4NP | GB10 superchip: Blackwell GPU + 20-core Arm CPU on one package over NVLink-C2C |
| **Compute capability** | 8.9 (`sm_89`) | 9.0 (`sm_90` / `sm_90a`) | 10.0 (`sm_100` / `sm_100a`) | **12.1** (`sm_121` / `sm_121a`); RTX 50-series is 12.0 (`sm_120`) |
| **TDP** | 350 W | 700 W | 1000 W | ~240 W (whole DGX Spark box, CPU included) |
| ***Memory hierarchy*** | | | | |
| gMEM size | 48 GB **GDDR6** w/ ECC (384-bit) | 80 GB HBM3 (5 stacks, 5120-bit) | 192 GB HBM3e (8 stacks, 8192-bit) | 128 GB **LPDDR5x**, physically **unified & coherent** with the CPU |
| gMEM bandwidth | 864 GB/s | 3.35 TB/s (3.9×) | 8 TB/s (9.3×) | **273 GB/s** (~1/30 of B200) |
| L2 cache | 96 MB | 50 MB (unified) | ~126 MB (63 MB per die — **not** unified across dies) | not published |
| Constant memory | 64 KB bank, per-SM constant cache | same | same | same |
| NVLink (scale-up) | **None** — PCIe Gen4 ×16, 64 GB/s bidir | NVLink 4, 900 GB/s/GPU | NVLink 5, 1.8 TB/s/GPU | **No GPU↔GPU NVLink**; NVLink-C2C to the on-package CPU only. 2 boxes pair over 200 GbE (ConnectX-7 RDMA) |
| ***Streaming Multiprocessor*** | | | | |
| SM count | 142 (144 on full die) | 132 (144 on full die) | **148** (74 per die) | **48** |
| FP32 lanes | 128/SM → 18,176 | 128/SM → 16,896 | 128/SM → 18,944 | 128/SM → 6,144 |
| Warp schedulers | 4 partitions/SM, 32-thread warps | same | same | same |
| Max warps / threads per SM | **48 warps / 1536 threads** | 64 warps / 2048 threads | 64 warps / 2048 threads | **48 warps / 1536 threads** |
| Max CTAs per SM | 24 | 32 | 32 | 24 |
| Register file | 256 KB/SM (64K × 32-bit), max 255 regs/thread | same | same | same |
| L1 + SMEM (unified) | **128 KB/SM** | 256 KB/SM | 256 KB/SM | **128 KB/SM** |
| Thread block cluster | **No** | Yes — 8 CTAs portable (16 max), DSMEM | Yes, same, **+ Cluster Launch Control** (`clusterlaunchcontrol.try_cancel`) for persistent-kernel scheduling | **No multi-CTA clusters** (1×1×1 only), no DSMEM, no CLC |
| ***Shared memory*** | | | | |
| sMEM per SM | **100 KB** (carved from the 128 KB unified) | 228 KB | 228 KB | **100 KB** (carved from the 128 KB unified) |
| Max dynamic sMEM per CTA | 99 KB | 227 KB (opt-in via `cudaFuncAttributeMaxDynamicSharedMemorySize`) | same | 99 KB |
| sMEM banks | 32 banks × 4 B wide | same | same | same |
| DSMEM | No | Cluster-wide sMEM addressing across ≤8 CTAs | same | No |
| ***TMA*** | | | | |
| TMA | **No** — only `cp.async` (LDGSTS, Ampere-style, ≤16 B/thread) | Yes (1st gen): `cp.async.bulk[.tensor]`, tiled + im2col, tensor map descriptor, 32/64/128 B swizzle, `mbarrier` completion | Yes + `im2col_w`/`w128`, **gather4/scatter4**, and `tcgen05.cp` SMEM→TMEM | **Yes** — `cp.async.bulk[.tensor]` with tensor maps and swizzle (the big win over Ada); no `tcgen05.cp` |
| Async barrier | `mbarrier` arrive/wait, **no transaction count** | `mbarrier` in sMEM, transaction-count based | same | `mbarrier` with transaction count (needed by TMA) |
| ***Tensor Core*** | | | | |
| Generation | 4th gen (Ada), 4 tensor cores/SM | 4th gen (Hopper), 4 tensor cores/SM (one per partition) | 5th gen, **one SM-wide** tensor core (`tcgen05`) | 5th gen (GeForce), 4 tensor cores/SM — **no TMEM** |
| MMA instruction | `mma.sync` — **synchronous, warp-level** (32 threads), operands staged through `ldmatrix` | `wgmma.mma_async` — issued by a whole **warpgroup** (128 threads) | `tcgen05.mma` — issued by a **single thread**; `cta_group::1` or **`cta_group::2`** (2 SMs cooperate on one MMA) | `mma.sync` — warp-level, **but with block-scaled kinds** (`kind::mxf8f6f4`, `kind::mxf4nvf4`). No `wgmma`, no `tcgen05` |
| A / B operand source | Both RMEM (SMEM → RMEM via `ldmatrix`) | A: RMEM or SMEM, B: SMEM | A: SMEM or **TMEM**, B: SMEM | Both RMEM (TMA → SMEM → RMEM via `ldmatrix`) |
| Accumulator location | RMEM (registers) | RMEM (registers) → register pressure caps tile size | **TMEM** — 256 KB/SM (128 lanes × 512 cols × 4 B), allocated in 32-col units | RMEM (registers) |
| Precisions | TF32, BF16, FP16, FP8 (E4M3/E5M2), INT8/INT4 — **no FP64 tensor core** | FP64, TF32, BF16, FP16, FP8 (E4M3/E5M2), INT8 | + FP6, **FP4** (NVFP4 = E2M1 + E4M3 per-16 scale; MXFP4/6/8 block-scaled) | BF16, FP16, FP8, FP6, **FP4** (NVFP4/MXFP4 block-scaled), INT8 — **no FP64 tensor core** |
| Dense BF16 | 181 TFLOPS | 989 TFLOPS | 2.25 PFLOPS | ≈125 TFLOPS † |
| Dense FP8 | 366 TFLOPS | 1.98 PFLOPS | 4.5 PFLOPS | ≈250 TFLOPS † |
| Dense FP4 | — | — | 9 PFLOPS | ≈500 TFLOPS (NVIDIA markets "1 PFLOP", which is the sparse figure) |
| FP64 tensor | — (1.4 TFLOPS vector, 1/64 rate) | 67 TFLOPS | 40 TFLOPS (de-emphasized) | — (1/64 vector rate) |
| Structured sparsity | 2:4, 2× | 2:4, 2× | 2:4, 2× | 2:4, 2× |

Notes:
- All TFLOPS are **dense**; marketing decks usually quote the 2× sparse number.
- † GB10's BF16/FP8 numbers are **derived** by scaling down from the published FP4 figure using
  the GeForce Blackwell 1:2:4 BF16:FP8:FP4 ratio — treat as approximate, and measure on the box.
- NVIDIA calls both Ada and Hopper tensor cores "4th gen", but they are **not** the same
  hardware: Hopper adds async warpgroup MMA with SMEM-resident operands, Ada is stuck on
  synchronous warp-level `mma.sync`. Kernels do not port across without a rewrite.
- The same trap repeats with Blackwell. **SM12x is not a subset or superset of SM100.** It shares
  the FP4 numerics and TMA, but has *no* TMEM, *no* `tcgen05`, *no* clusters, and Ada-class
  occupancy/sMEM limits. `sm_100a` cubins do not run on `sm_121a`; CUTLASS ships separate SM100
  and SM120 kernel families for exactly this reason.
- **DGX Spark is a dev box, not a serving box.** 273 GB/s means decode throughput is roughly
  1/30 of a B200 — the value is the 128 GB of unified, CPU-coherent memory letting you load and
  debug a large model locally before deploying on GB200. Bandwidth-bound intuition built here
  will not transfer; compute-bound (prefill, FP4 GEMM shape) intuition mostly will.
- **RTX 5090** = GB202, `sm_120`, 170 SMs, 32 GB GDDR7 @ 1.79 TB/s, ~1.68 PFLOPS dense FP4 —
  same SM12x microarchitecture as GB10, roughly 3.5× the SMs and 6.5× the bandwidth.
- **L40S** has more SMs than H100 but ~1/4 the bandwidth, half the sMEM per SM, no TMA, no
  clusters, and no NVLink. It is compute-dense and **memory/interconnect-starved** — fine for
  small-model or batch-heavy serving, painful for tensor-parallel LLM inference.
- **L40** is the same AD102 with 142 SMs at 300 W and lower clocks (~90 TFLOPS dense BF16).
- **H200** is the same GH100 die (132 SMs, everything on-chip identical to H100 SXM5); only memory changes: 141 GB HBM3e @ 4.8 TB/s.
- **B300 / GB300 (Blackwell Ultra)** is the same SM100 microarchitecture: 288 GB HBM3e @ 8 TB/s and ~1.5× the FP4 dense throughput of B200, with FP64 cut further.
- **GB200** = 2× B200 + Grace CPU on one module; per-GPU numbers above still hold (clocked to ~1200 W).
- The two-die B200 presents as **one CUDA device**, but L2 is per-die — cross-die traffic crosses NV-HBI, so locality still matters.