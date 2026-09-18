# Reading guide

| Priority | Primary source | Assigned use |
|---|---|---|
| Required | [Megatron-LM](https://arxiv.org/abs/1909.08053) | Forward-pass tensor-parallel construction; isolate it from training machinery |
| API | [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html) | Process groups, collective completion, NCCL backend, and launch semantics |
| Tutorial | [Writing distributed applications with PyTorch](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) | Reproduce basic communication and rank-local state |
| Measurement | [NCCL tests](https://github.com/NVIDIA/nccl-tests) | Measure actual message sizes on the chosen topology |

Opened during preparation on 2026-09-11. Reading exercise: draw the two SwiGLU
shards and identify the only tensor that must be reduced in the simple forward.
