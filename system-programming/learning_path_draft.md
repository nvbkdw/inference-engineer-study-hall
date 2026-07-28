> **Superseded by [README.md](README.md)** — the expanded scope (adds quantitative foundations,
> CPU/cache/DRAM/NUMA, kernel control planes, GPU-as-system-resource, collectives, and a
> measurement track) plus phase-by-phase exit tests and a lab-hardware plan. Kept here as the
> original draft.

To build production-grade, low-latency infrastructure in Rust—such as GPU memory offloading engines, NVMe-over-Fabrics, or high-performance zero-copy storage engines—you need to bridge **low-level hardware architecture**, **kernel-bypass mechanisms**, and **Rust’s type and concurrency system**.

Here is a comprehensive breakdown of the fundamental knowledge required, followed by a step-by-step learning path.

---

## Part 1: Scope of Fundamental Knowledge

### 1. PCIe & Hardware Interconnects

* **Hardware Fundamentals:** PCIe topology (Root Complex, Switches, Endpoints), PCIe lanes/bandwidth (Gen 4/5/6), Base Address Registers (BARs), Transaction Layer Packets (TLPs), and Configuration Space.
* **Direct Memory Access (DMA):** Physical vs. Virtual DMA, DMA contiguity, IOMMU (Intel VT-d, AMD-Vi), address translation (IOVA to PA), and PCIe Peer-to-Peer (P2P) transactions.
* **Linux Kernel Interfaces:** VFIO (`/dev/vfio`), UIO, Linux DMA mapping APIs (`dma_map_single`, `dma_map_sg`), MSI/MSI-X interrupts.
* **Rust Concepts:**
* Memory-mapped I/O (MMIO) with `core::ptr::read_volatile` and `write_volatile`.
* Hardware memory barriers (`core::sync::atomic::fence`).
* Interfacing with Linux kernel primitives via unsafe wrappers like [`vfio-ioctls`](https://docs.rs/vfio-ioctls).



---

### 2. Virtual Memory, Page Management & Caching

* **Virtual Memory Architecture:** Multi-level Page Tables (PGD, PUD, PMD, PTE), Page Fault handling, Translation Lookaside Buffer (TLB) management, and TLB shootdowns.
* **Page Pinning & Allocation:** Page pinning (`pin_user_pages`), page locking (`mlock`), HugeTLB / Transparent Huge Pages (2MB / 1GB pages), and Non-Uniform Memory Access (NUMA) node placement.
* **Cache Alignment & CPU Hierarchy:** L1/L2/L3 cache line architecture (64-byte alignment), cache coherence protocols (MESI/MOESI), false sharing, and hardware prefetching.
* **Rust Concepts:**
* `mmap` / `munmap` abstractions using [`memmap2`](https://docs.rs/memmap2).
* Custom allocators (`std::alloc::Allocator` trait) enforcing cache-line alignment (`std::alloc::Layout::from_size_align`).
* Zero-copy parsing using crates like `zerocopy` or `bytemuck`.



---

### 3. NVMe & User-Space Storage

* **NVMe Specification:** Controller registers, Admin Queues vs. I/O Queues, Submission Queues (SQ) / Completion Queues (CQ), Doorbell registers, and LBAs.
* **Kernel Bypass & Async I/O:**
* Polled-mode I/O vs. interrupt-driven I/O.
* Kernel bypass with user-space drivers (e.g., SPDK architecture).
* Modern asynchronous Linux storage interfaces like [`io_uring`](https://docs.rs/io-uring) with registered buffers (`IORING_REGISTER_BUFFERS`).


* **Rust Concepts:**
* Lock-free ring buffer implementations for SQs and CQs.
* Managing lifetime safety over pinned DMA buffers using `Pin<Box<T>>` or `PhantomData`.



---

### 4. VRAM, GPU Memory & Heterogeneous Transfers

* **GPU Memory Hierarchy:** HBM / GDDR VRAM, Device Memory vs. Host Memory, Pinned (Page-Locked) Host Memory (`cudaHostAlloc`), Unified Virtual Addressing (UVA), and Unified Memory.
* **Advanced Transfers:**
* **CUDA IPC:** Sharing VRAM pointers across OS processes.
* **GPUDirect Storage (GDS):** Direct DMA from NVMe PCIe storage to GPU VRAM, bypassing host CPU RAM.
* **GPUDirect RDMA (GDR):** Direct DMA transfer between RDMA NICs and GPU VRAM over the PCIe switch.


* **Rust Concepts:**
* Binding to CUDA Driver APIs using [`cudarc`](https://docs.rs/cudarc) or `cust`.
* Managing raw CUDA driver context (`CUdeviceptr`, `cuMemAlloc`, `cuMemHostRegister`).



---

### 5. RDMA (Remote Direct Memory Access)

* **RDMA Architecture:** InfiniBand, RoCE (RDMA over Converged Ethernet), iWARP.
* **Verbs API Primitives:** Memory Regions (MRs), Local Key (`lkey`), Remote Key (`rkey`), Protection Domains (PD), Queue Pairs (QP), Completion Queues (CQ).
* **Communication Operations:**
* **One-Sided Operations:** `RDMA READ`, `RDMA WRITE`, Atomic operations (Read/Write directly to remote memory without remote CPU involvement).
* **Two-Sided Operations:** `SEND` / `RECEIVE`.
* **Transport Modes:** Reliable Connection (RC), Unreliable Datagram (UD), Unreliable Connection (UC).


* **Rust Concepts:**
* High-level async abstractions using [`async-rdma`](https://docs.rs/async-rdma) or low-level bindings (`rdma-sys`).
* Integrating poll-based CQs into Rust async runtimes (e.g., Tokio event loops).



---

## Summary Matrix of Key Technologies & Rust Tools

| Domain | Low-Level Mechanisms | Key Rust Crates / Interfaces |
| --- | --- | --- |
| **PCIe & HW** | MMIO, VFIO, IOMMU, BAR mapping, P2P | [`vfio-ioctls`](https://docs.rs/vfio-ioctls), `core::ptr::read_volatile` |
| **Memory / Cache** | Page Tables, HugePages, NUMA, Alignment | [`memmap2`](https://docs.rs/memmap2), `std::alloc::Layout`, `bytemuck` |
| **NVMe** | SQ/CQ Doorbells, Kernel Bypass, Ring Buffers | [`io-uring`](https://docs.rs/io-uring), `spdk-rs`, custom SQ/CQ rings |
| **VRAM** | Pinned Host RAM, CUDA IPC, GPUDirect | [`cudarc`](https://docs.rs/cudarc), CUDA Driver API wrappers |
| **RDMA** | `ibverbs`, Memory Regions (MR), One-Sided Ops | [`async-rdma`](https://docs.rs/async-rdma), `rdma-sys` |

---

## Part 2: Step-by-Step Learning Path

```
[Phase 1: Unsafe & Systems Rust] ──> [Phase 2: Virtual Memory & Page Tables]
                                              │
                                              ▼
[Phase 4: VRAM & GPU Transfers]  <── [Phase 3: PCIe & User-Space NVMe]
              │
              ▼
[Phase 5: RDMA & Remote Memory]  ──> [Phase 6: Capstone Integration]

```

### Phase 1: Systems Rust Foundations & Unsafe Mastery

* **Focus:** Understanding Rust’s raw memory model, pointer mechanics, and hardware fences.
* **Topics to Master:** `raw pointers` (`*const T`, `*mut T`), `UnsafeCell<T>`, memory alignment, `Layout`, `Pin`, `PhantomData`, atomic operations (`Ordering::SeqCst`, `Acquire`, `Release`), and hardware volatile reads/writes.
* **Mini-Project:** Build a cache-line-aligned (64-byte), lock-free Single-Producer Single-Consumer (SPSC) ring buffer in Rust without external dependencies.

### Phase 2: OS Virtual Memory, Page Management & Kernel Interfaces

* **Focus:** Interfacing directly with the OS virtual memory manager and zero-copy APIs.
* **Topics to Master:** POSIX `mmap`, `madvise`, `mlock`, Linux HugeTLB (2MB/1GB pages), asynchronous kernel bypass with `io_uring`.
* **Mini-Project:** Use [`memmap2`](https://docs.rs/memmap2) and [`io-uring`](https://docs.rs/io-uring) to build a multi-threaded, zero-copy file-backed page cache with custom page pinning and `IORING_REGISTER_BUFFERS`.

### Phase 3: PCIe Interconnects & User-Space Drivers

* **Focus:** Bypassing standard OS drivers to communicate with hardware directly over PCIe.
* **Topics to Master:** VFIO device binding, mapping PCIe BARs into user space, allocation of physical DMA-safe continuous memory buffers.
* **Mini-Project:** Write a user-space driver component in Rust using [`vfio-ioctls`](https://docs.rs/vfio-ioctls) that maps a target device's PCIe BAR space, configures an I/O ring buffer, and polls for hardware completion.

### Phase 4: VRAM Allocation, CUDA Drivers & GPU Memory Management

* **Focus:** Heterogeneous computing and GPU memory transfers.
* **Topics to Master:** Pinned host allocations (`cuMemHostRegister`), asynchronous GPU streams, CUDA IPC memory handles, and Unified Memory.
* **Mini-Project:** Using [`cudarc`](https://docs.rs/cudarc), write a Rust service that allocates page-locked (pinned) system RAM, streams data asynchronously to NVidia GPU VRAM over PCIe, and passes CUDA IPC handles to a second process.

### Phase 5: RDMA Networks & Distributed Memory Access

* **Focus:** Kernel-bypass network memory access using `libibverbs`.
* **Topics to Master:** Registering memory regions (`ibv_reg_mr`), Queue Pair state transitions (RESET $\to$ INIT $\to$ RTR $\to$ RTS), posting `RDMA READ` and `RDMA WRITE` work requests.
* **Mini-Project:** Using [`async-rdma`](https://docs.rs/async-rdma) (or Soft-RoCE `rdma_rxe` in a VM), build a remote key-value cache where clients execute one-sided `RDMA READ` operations directly on host memory without CPU intervention on the server.

### Phase 6: Capstone Integration – GPUDirect / NVMe-to-VRAM Streaming

* **Focus:** End-to-end zero-copy pipeline across storage, networking, PCIe, and GPU VRAM.
* **Ultimate Project Goal:** Build a prototype for local LLM parameter/tensor streaming or MoE offloading engine in Rust:
* Read weights from NVMe SSDs directly into CUDA VRAM (via GPUDirect Storage or custom PCIe P2P buffers), bypassing CPU staging buffers.
* Optionally accept remote tensor blocks over RDMA directly into GPU VRAM (GPUDirect RDMA).