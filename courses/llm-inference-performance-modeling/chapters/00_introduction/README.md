# Chapter 0: From text to a serving system

This chapter introduces the architecture and vocabulary used throughout the
course. It is a conceptual reading chapter with mathematical intuition, worked
examples, and diagrams. It contains no code, projects, or required submissions.
No software installation, model download, or GPU is needed.

Read it at the beginning of Week 1, before the Chapter 1 lab. Allow approximately
3–4 hours within the existing Week 1 reading allocation; the semester remains
16 weeks with eight projects. Familiarity with vectors, matrix multiplication,
and elementary probability is enough. The equations describe what the system
does and why its costs change, without requiring an implementation.

Follow one request through these readings in order:

| Reading | Central idea |
|---|---|
| [1. Transformer architecture](architecture.md) | Tokens become vectors; attention combines context; an MLP transforms features; logits become a next-token distribution |
| [2. From generation to serving](inference.md) | Prefill creates reusable state; decode extends it; a server schedules many such requests |
| [3. Distributed inference and performance modeling](systems.md) | Placement creates communication; predictions must connect kernels, model execution, servers, and the whole system |
| [Annotated references](references.md) | Original papers and official documentation for further reading |

The course uses the dense Qwen3-8B and Qwen3-32B architectures as its running
examples. Chapter 1 introduces one small Qwen3 tiny configuration for correctness
and timing before moving to the real checkpoints. This introduction adds no
model variants.

After reading, the relationships between token count, tensor shapes, persistent
cache state, scheduling, and user-visible latency should form a single picture.
The later chapters turn each part of that picture into measured evidence.

[Course home](../../README.md) · [Begin the reading](architecture.md) · [Next chapter: Reconstruct Qwen3](../01_reconstruct_qwen3/background.md)
