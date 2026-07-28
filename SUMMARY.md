# Table of Contents

[LLM Inference System Engineering]

- [GPU Architecture](gpu-architecture/modern-gpu-microarchitrue.md) — memory hierarchy, SM, TMA, tensor cores, per-generation spec table
- [Kernel Programming](kernel-programming/GEMM.md) — GEMM optimization on modern GPU hardware
- [Systems Programming](system-programming/README.md) — scope + learning path for the host, buses, and storage/network tiers around the GPU
  - [Lab: DGX Spark (GB10)](system-programming/lab-dgx-spark.md) — measured hardware inventory and per-phase feasibility
  - [Original draft](system-programming/learning_path_draft.md)
- [SGLang Architecture Walkthrough](learning-sglang/architecture/README.md) — 12-chapter code walkthrough of the serving engine
