# Lab: DGX Spark (GB10) — `spark-2970`

Companion to [README.md](README.md). The curriculum there is hardware-neutral; this file is
what *your* box can and cannot teach, measured rather than assumed.

## DGX Spark Notes
1. Your GPU has no PCIe data path

000f:01:00.0  LnkCap: 2.5GT/s x16   LnkSta: 2.5GT/s x1 (downgraded)

That's PCIe Gen1 ×1, ~250 MB/s — and the capability is only 2.5 GT/s too, so it's by design, not a misconfiguration. The GB10 GPU isn't data-attached over PCIe at all; it's on-package over NVLink-C2C, and GPU memory is the 119 GiB of LPDDR5X, coherently shared. That's why nvidia-smi reports memory as "Not Supported / 0 MiB".

So a real chunk of the plan I wrote yesterday is dead on this box: pinned vs pageable memory, H2D bandwidth, copy/compute overlap, and GPUDirect Storage-to-VRAM all optimize a copy that doesn't exist here. In exchange you get the GH200/GB200 coherent programming model locally, which almost nobody can practice — so I replaced the GDS lab with a coherent-memory lab (managed vs system-allocated vs explicit, migration behavior, and a 150 GB-over-119 GB oversubscription test that is literally the SSD-as-VRAM thesis arriving through the driver's front door).

2. Your storage:memory gap is 20:1, not 240:1

┌───────────────────────┬─────────────┬──────────────────────────┐
│                       │ H100 server │          Spark           │
├───────────────────────┼─────────────┼──────────────────────────┤
│ GPU-visible bandwidth │ 3350 GB/s   │ 273 GB/s                 │
├───────────────────────┼─────────────┼──────────────────────────┤
│ NVMe read             │ ~14 GB/s    │ ~14 GB/s (Gen5 ×4, real) │
├───────────────────────┼─────────────┼──────────────────────────┤
│ Gap                   │ 240:1       │ 20:1                     │
└───────────────────────┴─────────────┴──────────────────────────┘

Same drive class, 12× less memory bandwidth to lose. The SSD-as-VRAM capstone is structurally easier on your box than on the consumer desktop the WiCi post targets. Running the arithmetic for a 750B/40B-active MoE at 4-bit: fully resident would be ~13.7 tok/s, 32% resident at uniform access is ~1 tok/s, 90% hit rate gets ~7 tok/s. That's a live 1→14 tok/s ladder to climb, not a foregone conclusion — a genuinely good capstone target.

Other things worth knowing

- Two L3 domains, asymmetric: cpu0–9 share 8 MB, cpu10–19 share 16 MB. Each cluster is 5× A725 + 5× X925. Cluster 1 is strictly better (2× L3, higher clocks). Benchmark on cpu15–19.
- ARM's weak memory model is an asset — your Phase-1 lock-free ring buffer will actually be tested, where x86 TSO would hide the missing barriers.
- max_hw_sectors_kb=128 caps I/O size, so hitting the BDP target (~1.1 MB in flight) needs QD ≥ 9 minimum, realistically 32–64. Concrete Phase-2 arithmetic.
- VFIO target problem solved: your only NVMe is the boot drive with / on it — never unbind it. But you have two idle ConnectX-7 cards and an unused Realtek NIC, and SMMUv3 is active with iommu.passthrough=0. Use a NIC function.
- Drop the resctrl cache-partitioning exit test — x86-only, and ARM MPAM isn't exposed here.
- Noise: k8s/flannel + docker + ~14 veths are running. Quiet the box or your p99s are fiction.

Highest-leverage upgrade

A QSFP112 DAC loopback cable between your two physically separate ConnectX-7 cards (PCI domains 0000 and 0002). All four ports are currently DOWN with no cable. One cable turns Phase 6 from "Soft-RoCE API practice" into real 200 GbE line-rate numbers between two processes on one machine. I've flagged it as unverified — mlx5 card-to-card loopback on one host normally works, but confirm before buying, and put each port in its own netns so traffic actually leaves the NIC.

Everything the Spark can't cover (PCIe H2D, GDS, NVLink/NCCL, MIG, NUMA) is ~3 focused weekends of cloud H100 time, roughly $50–150 total. No hardware purchase needed.

Section 6 of the lab doc has a copy-pasteable first week: the apt install list (fio, liburing-dev, rustc and libibverbs-dev are all missing), the box-quieting steps, three baseline measurements, and Soft-RoCE bring-up so verbs are available today.


---

## 1. Measured inventory

### CPU — 20 cores, heterogeneous, **two L3 domains**

| Cluster | CPUs | Core | L1d | L2 | L3 (shared) | Max clock |
|---|---|---|---|---|---|---|
| 0 | `0–4` | Cortex-A725 (`0xd87`) | 64 K | 512 K | **8 MB** (cpu0–9) | 2808 MHz |
| 0 | `5–9` | **Cortex-X925** (`0xd85`) | 64 K | 2 MB | **8 MB** (cpu0–9) | 3900 MHz |
| 1 | `10–14` | Cortex-A725 (`0xd87`) | 64 K | 512 K | **16 MB** (cpu10–19) | 2860 MHz |
| 1 | `15–19` | **Cortex-X925** (`0xd85`) | 64 K | 2 MB | **16 MB** (cpu10–19) | 3978–4004 MHz |

- aarch64, SVE2 @ **128-bit** vector length, `bf16` + `i8mm` present, `lse` atomics.
- Governor: `performance`. No SMT.
- **The L3 asymmetry (8 MB vs 16 MB) is real and undocumented in the marketing.** Cluster 1 is
  the better cluster on two counts: 2× the L3 and slightly higher clocks.

### Memory — 119 GiB LPDDR5X, unified and CPU/GPU-coherent

```
1 NUMA node · 4 KiB base pages · THP = madvise · 0 hugepages preallocated
Hugepagesize 2048 kB · memlock ulimit ≈ 15 GB · 15 GiB swap
```
- ~273 GB/s theoretical (per your [GPU table](../gpu-architecture/modern-gpu-microarchitrue.md)).
  Measure the CPU-side achievable share yourself — it will be well under that.

### GPU — GB10, `sm_121`, driver 580.126.09, CUDA 13.0

```
000f:01:00.0  LnkCap: 2.5GT/s x16   LnkSta: 2.5GT/s x1 (downgraded)
nvidia-smi memory: "Not Supported" / 0 MiB
```

> ### ⚠ The single most important fact about this machine
>
> **The GPU's PCIe link is Gen1 ×1 — ~250 MB/s.** That is not a fault and not a misconfiguration.
> On GB10 the GPU is *not data-attached over PCIe at all*; it sits on-package with the CPU over
> **NVLink-C2C**, and the PCIe function exists only for enumeration and display. GPU memory
> **is** system LPDDR5X, coherently shared — which is why `nvidia-smi` reports no memory total.
>
> Consequences, stated bluntly:
> - There is **no host↔device PCIe path** to measure, tune, or optimize.
> - Pinned vs pageable host memory, `cudaMemcpy` bandwidth, bounce buffers, H2D/D2H overlap —
>   **all of these lessons are absent here**, because the copy they optimize does not exist.
> - **GPUDirect Storage as classically defined is meaningless on this box**, because the thing
>   GDS eliminates (the CPU bounce buffer on the way to a *discrete* VRAM across PCIe) is
>   already gone. NVMe DMAs straight into LPDDR5X and the GPU reads it in place.
> - Anything you measure as a "GPU memory bandwidth" number is a **273 GB/s LPDDR5X number**,
>   not a 3.35 TB/s HBM number. It is ~12× off from a serving GPU.

### Storage — one Samsung 4 TB, **PCIe Gen5 ×4**

```
/dev/nvme0n1  SAMSUNG MZALC4T0HBL1-00B07   4.10 TB, 655 GB used (~3.4 TB free)
0004:01:00.0  LnkCap/LnkSta: 32GT/s x4      → 16 GB/s raw link
scheduler=[none]  nr_requests=1023  read_ahead_kb=128
logical_block_size=512   max_hw_sectors_kb=128   ← max I/O size is 128 KB
```
- **This is the boot drive with `/` mounted on it.** Do not bind it to `vfio-pci`.
- `max_hw_sectors_kb=128` sets your BDP arithmetic: to hold ~1.1 MB in flight you need
  **QD ≥ 9 minimum**, realistically 32–64. You cannot get there with one 1 MB read.

### Network — 2× ConnectX-7 dual-port (4 RoCE ports), **all links DOWN**

```
0000:01:00.0/.1  ConnectX-7  Gen5 x4   → rocep1s0f0,  rocep1s0f1
0002:01:00.0/.1  ConnectX-7  Gen5 x4   → roceP2p1s0f0, roceP2p1s0f1
mlx5_ib + ib_uverbs loaded · ib_write_bw / ibv_devinfo present
All 4 ports: state DOWN, physical_state DISABLED (no cable)
Live connectivity is WiFi (wlP9s9) + tailscale. Realtek RTL8127 (enP7s7) is DOWN and unused.
```

### Platform

```
Ubuntu 24.04.3 · kernel 6.14.0-1015-nvidia · aarch64
IOMMU: ARM SMMUv3 ×3, 21 groups, iommu.passthrough=0  (translation ACTIVE — VFIO works)
cmdline: pci=pcie_bus_safe   (caps PCIe MaxPayloadSize conservatively)
Each device sits in its own PCI domain behind its own NVIDIA root port.
```

**Tools present:** `perf` `bpftrace` `numactl` `nsys` `ncu` `nvcc` (13.0.88) `cuda-gdb`
`compute-sanitizer` `ib_write_bw` `ibv_devinfo` `cmake` `gcc` `python3`
**Missing, install first:** `fio` `rustc`/`cargo` `liburing-dev` `libibverbs-dev` `dcgmi`
**Not loaded:** `nvidia-fs` (GDS kernel module); `/usr/local/cuda/gds/cufile.json` exists

**Noise to control:** k8s (flannel + `cni0` + ~14 veths), docker, 4 logged-in users, uptime 6 days.
Benchmarking on this box without quieting it will produce garbage p99s.

---

## 2. What this box uniquely teaches

Don't treat the Spark as a downgraded H100 server. It's a different — and in two ways more
*forward-looking* — machine.

| # | What it teaches | Why it matters |
|---|---|---|
| 1 | **Coherent unified memory as the programming model** | This is GH200/GB200's model. `cudaMallocManaged`, system-allocated memory addressable by the GPU, ATS, no explicit copies. Everything you learn here transfers *up* to GB200 NVL72 — and it's the model the whole industry is moving toward. Most people can't practice it. |
| 2 | **A weak memory model, for real** | ARM is weakly ordered. On x86, TSO silently masks most missing-barrier bugs. Here, `Relaxed` vs `Acquire`/`Release` produces observable differences. Your Phase-1 lock-free ring buffer will actually be *tested* by this hardware. |
| 3 | **Heterogeneous cores + two L3 domains** | Real placement decisions: X925 vs A725, cluster 0 vs cluster 1. Poor-man's NUMA — cross-cluster coherence traffic is measurable with `perf c2c` even though `numactl` reports one node. |
| 4 | **A storage:memory gap that is 12× narrower than a serving box** | See §3. This is the reason Track A (SSD-as-VRAM) is *more* tractable here than on a 5090 desktop. |
| 5 | **Full PCIe Gen5 ×4 endpoints you're allowed to break** | Two idle ConnectX-7 cards + an unused Realtek NIC = legitimate VFIO / user-space-driver targets that won't brick the box. |

### The ratio that defines this machine

| | H100 server | **DGX Spark** |
|---|---|---|
| GPU-visible memory bandwidth | 3350 GB/s (HBM3) | **273 GB/s (LPDDR5X)** |
| NVMe sequential read | ~14 GB/s | ~14 GB/s (Gen5 ×4) |
| **Storage : memory gap** | **≈ 240 : 1** | **≈ 20 : 1** |
| Host↔GPU transfer | PCIe Gen5, ~55 GB/s, explicit copy | **none — coherent, zero-copy** |

**Read that table twice.** On a discrete-GPU server, streaming weights off NVMe means giving up
240× bandwidth *and* paying a PCIe hop. Here it means giving up 20× and paying nothing. The
SSD-as-VRAM thesis is structurally easier on this box than on the consumer desktop the WiCi post
targets — and the numbers land in an interesting regime rather than a hopeless one:

> A 750B-class MoE at 4-bit (~377 GB, ~40B active ≈ 20 GB/token):
> - **Fully resident** (impossible — only 119 GB fits): 273/20 ≈ **13.7 tok/s ceiling**
> - **~32% resident, uniform access**: ~13.6 GB/token from NVMe → **~1 tok/s**
> - **90% hit rate via prediction/locality**: ~2 GB/token from NVMe → **~7 tok/s**
>
> So on your box the whole game plays out between **1 and 14 tok/s**, and *both* tiers bind.
> That is a genuinely good capstone target: measurable, improvable, and not a foregone conclusion.

---

## 3. What this box structurally cannot teach

Be honest about these now rather than discovering them in month four.

| Gap | Why | Fill it with |
|---|---|---|
| **Host↔GPU PCIe** — pinned memory, H2D bandwidth, copy/compute overlap | No PCIe data path exists (Gen1 ×1) | Cloud H100/L40S hours (~$3–5/hr) |
| **GPUDirect Storage (P2P to VRAM)** | Nothing to bypass; `nvidia-fs` not even loaded | Cloud H100 + local NVMe instance |
| **Discrete-GPU bandwidth intuition** | 273 GB/s vs 3350 GB/s is a 12× different world | Cloud. Your own GPU doc already warns: *"bandwidth-bound intuition built here will not transfer."* That warning now applies to the systems curriculum too. |
| **NUMA** | Single node | Partial substitute: the two L3 clusters. Full: any 2-socket server, or cloud. |
| **NVLink / NCCL / multi-GPU collectives** | One GPU, no GPU–GPU NVLink | Cloud 2×/8× H100 for a weekend |
| **MIG** | Not supported on GB10 | Cloud A100/H100 |
| **Intel CAT/MBA cache partitioning** (`resctrl`) | x86-only. ARM MPAM is not exposed here. | Skip, or any modern Xeon/EPYC. Drop this from Phase 1's exit test. |
| **VFIO on NVMe** | Only drive is the boot drive | Use a **ConnectX-7 function** or the **Realtek NIC** as the VFIO target instead |
| **Live RDMA numbers** | All 4 ports DOWN, no cable | See §4 — this one is cheap to fix |

---

## 4. Three cheap upgrades, ranked by leverage

**1. A QSFP112 DAC loopback cable (~$50–150) — highest leverage by far.**
You have two *physically separate* ConnectX-7 cards (PCI domains `0000` and `0002`). Cable one
port of card A to one port of card B and you have real 200 GbE RoCE hardware between two
processes on one machine — genuine line-rate numbers, real MR/QP/CQ behavior, real congestion.
That converts Phase 6 from "API-only via Soft-RoCE" into a full lab.
*Unverified — mlx5 loopback between two cards on one host normally works, but confirm before
buying. Put each port in its own network namespace so traffic actually leaves the NIC.*

**2. Cloud H100 hours (~$50–150 total for the whole curriculum).**
You need maybe 3 focused sessions: (a) PCIe H2D + pinned memory + GDS, (b) NCCL/NVLink topology,
(c) MIG/MPS isolation measurements. Budget one weekend each. Nothing here needs weeks.

**3. Check for a free M.2 slot.**
If your unit has a second one, a cheap 512 GB drive is your VFIO / user-space-NVMe-driver target
and unblocks the best lab in Phase 4. Verify physically — don't assume.

**Do not buy:** a discrete GPU for this box (no slot), or more RAM (soldered LPDDR5X).

---

## 5. Phase-by-phase feasibility

| Phase | On the Spark | Adjustment |
|---|---|---|
| **0** Quantitative baseline | ✅ Full | Recompute every Part II number for *this* box — the ratios are different and that's the lesson. |
| **1** Cores, caches, DRAM, NUMA | ✅ Strong, **better than x86 in two ways** | Weak memory model makes the lock-free lab real. Replace the NUMA lab with a **cross-L3-cluster** lab (cpu5 ↔ cpu15). **Drop the `resctrl` exit test** — no MPAM. Add: X925 vs A725 placement, and measure the 8 MB vs 16 MB L3 asymmetry. |
| **2** Kernel + io_uring | ✅ Full | Kernel 6.14 has everything. `apt install fio liburing-dev` first. The realized-bandwidth ladder on the Gen5 drive is the centerpiece — do it on a scratch file under `/`, `O_DIRECT`, and mind `max_hw_sectors_kb=128`. |
| **3** PCIe, DMA, IOMMU | ⚠️ Partial → **retarget** | GPU PCIe is dead, but NVMe and both CX-7s are real Gen5 ×4. SMMUv3 is active (`iommu.passthrough=0`) so DMA remapping cost is measurable. **VFIO target = an idle ConnectX-7 function or the Realtek NIC, never nvme0.** Bonus lab: measure the effect of `pci=pcie_bus_safe` on MPS. |
| **4** NVMe + bypass + GDS | ✅ NVMe full / ❌ GDS n/a | Do the whole storage stack here. **Reframe the GDS lab**: instead of "does P2P work," ask *"what does GDS even mean when GPU memory is system memory?"* — write that answer up; it's a better artifact than a benchmark. Then verify the real path on cloud H100. |
| **5** GPU as system resource | ✅ Mostly, ⚠️ different | Contexts, streams, CUDA graphs, launch overhead, VMM APIs (`cuMemCreate`/`cuMemMap`), CUDA IPC dedup — all work and all matter. **No MIG.** Verify MPS on GB10 before planning around it. **Add a coherent-memory lab** (below) — it's the box's best asset. |
| **6** RDMA + collectives | ⚠️ API yes, numbers no | Verbs/QP/MR/CQ learning works today via Soft-RoCE (`modprobe rdma_rxe`). Real numbers need the §4 cable. NCCL is single-rank only. |
| **7A** MoE streaming capstone | ✅ **Ideal — do this one** | 3.4 TB free, Gen5 drive, 119 GB unified cache tier, 20:1 gap, and no PCIe hop to hide behind. The arithmetic in §2 gives you a real 1→7→14 tok/s ladder to climb. |
| **7B** VRAM sharing capstone | ⚠️ Weaker | CUDA IPC dedup works and is worth doing. But "VRAM pressure" here is *system memory* pressure, so the eviction tier is swap/NVMe, not host DRAM. Different problem, still interesting — just say so in the writeup. |

### The lab that replaces GDS: coherent memory

Slot this into Phase 5. It has no equivalent on a discrete-GPU box and it is the model GB200 uses.

1. Compare, for the same 40 GB working set: `cudaMalloc` + explicit copy vs `cudaMallocManaged`
   vs plain `malloc` accessed directly from a kernel (system-allocated addressing).
   Measure bandwidth **and** first-touch/migration behavior.
2. Establish whether pages migrate at all, or whether access is genuinely in-place. Use
   `cudaMemAdvise` / `cudaMemPrefetchAsync` and see if they change anything.
3. Oversubscribe: allocate 150 GB managed against 119 GB physical and characterize the fault +
   swap behavior. **This is the SSD-as-VRAM thesis arriving through the front door** — the driver
   already implements a version of it, and finding out how well is a real result.
4. Write up: on a coherent platform, which of the four verbs (Fewer / Closer / Earlier / Shared)
   still have work to do, and which the hardware did for you?

---

## 6. First week — concrete

```bash
# 0. Install what's missing
sudo apt update && sudo apt install -y fio liburing-dev libibverbs-dev rdma-core \
     ibverbs-utils linux-tools-$(uname -r) hwloc numactl-dev build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # if going Rust

# 1. Quiet the box for benchmarking — pick a clean core set on the better cluster
#    cpu15-19 = X925 on the 16 MB L3. Reserve them via cgroup v2 cpuset, or isolcpus at boot.
sudo systemctl stop k3s kubelet docker 2>/dev/null   # whichever is actually running
sudo cpupower frequency-info                          # confirm governor stays 'performance'

# 2. Baseline the three tiers you own — before reading any more theory
#    memory:
lscpu -C && sudo perf stat -e cache-misses,cache-references -- <your pointer-chase>
#    storage (respect max_hw_sectors_kb=128):
fio --name=seq --filename=/scratch/testfile --size=64G --bs=128k --rw=read \
    --ioengine=io_uring --direct=1 --iodepth=64 --numjobs=4 --group_reporting
#    the "GPU PCIe is fake" experiment — prove it to yourself:
sudo lspci -s 000f:01:00.0 -vv | grep LnkSta      # Gen1 x1 = 250 MB/s
#    ...then run any H2D bandwidth test and observe it wildly exceeds that.

# 3. Bring up Soft-RoCE so the verbs API is available today
sudo modprobe rdma_rxe && sudo rdma link add rxe0 type rxe netdev wlP9s9 && ibv_devinfo -d rxe0
```

**Record every number in `labs/<phase>/RESULTS.md` with your prediction beforehand.**
Raw dumps are gitignored; the delta between prediction and measurement is the deliverable.

---

## 7. Re-run the inventory

```bash
lscpu; lscpu -C; numactl -H
for c in $(seq 0 19); do printf "cpu%-2s " $c; \
  cat /sys/devices/system/cpu/cpu$c/regs/identification/midr_el1; done
nvidia-smi; nvidia-smi topo -m; lspci -tv
for d in 000f:01:00.0 0004:01:00.0 0000:01:00.0; do sudo lspci -s $d -vv | grep -E "LnkCap|LnkSta"; done
lsblk; sudo nvme list; cat /sys/block/nvme0n1/queue/{max_hw_sectors_kb,nr_requests,scheduler}
ibv_devinfo | head -40; rdma link
ls /sys/class/iommu/; cat /proc/cmdline
```
