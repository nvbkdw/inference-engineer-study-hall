# Reading guide

| Priority | Primary source | Assigned use |
|---|---|---|
| Required | [DistServe](https://arxiv.org/abs/2401.09670) | Connect interference, stage allocation, and SLO-qualified throughput |
| Design | [Dynamo disaggregated serving, v0.9.0](https://docs.dynamo.nvidia.com/dynamo/v-0-9-0/design-docs/disaggregated-serving) | Trace handoff responsibilities; use matching code for any deployment |
| Transport | [PyTorch distributed tutorial](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) | Blocking/nonblocking communication and explicit process-local ownership |

Opened during preparation on 2026-09-11. Reading exercise: draw every queue and
copy along the handoff path, then mark the exact event after which source state
can be released. Distinguish a protocol diagram from evidence of measured overlap.
