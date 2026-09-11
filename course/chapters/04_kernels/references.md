# Reading guide

| Priority | Primary source | Assigned use |
|---|---|---|
| Required | [FlashAttention](https://arxiv.org/abs/2205.14135) | Derive stable tiled normalization and compare memory traffic |
| Required | [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Work division; explain occupancy and communication between workers |
| Implementation | [CuTe DSL programming model](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) | Layouts, JIT arguments, reductions, and hardware-specific programming guides |
| Setup | [CuTe quick start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html) | Select the compatible release and a working architecture-specific example |
| Measurement | [Nsight Compute guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Choose counters for the concrete data-movement hypothesis |

Opened during preparation on 2026-09-11. Read one official GEMM implementation
closely; record its exact source commit. Reading completion means you can draw
the movement of one operand tile and explain its synchronization.
