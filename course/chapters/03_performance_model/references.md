# Reading guide

| Priority | Primary source | Assigned use |
|---|---|---|
| Required | [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | Reconstruct arithmetic intensity and crossover with explicit units |
| Required | [Transformer inference](https://jax-ml.github.io/scaling-book/inference/) | Adapt the weight/KV model to the two Qwen configurations |
| Tool | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | CPU/GPU timeline and NVTX ranges; identify the critical path |
| Tool | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Interpret counters and profiling/replay overhead |
| API | [PyTorch CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html) | Event timing and synchronization semantics |

Opened during preparation on 2026-09-11. Match profiler options to the installed
release. Reading exercise: identify one assumption in each model that your trace
could falsify before fitting a new parameter.
