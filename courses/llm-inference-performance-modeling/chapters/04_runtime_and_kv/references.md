# Reading guide

| Priority | Primary source | Focus and reading question |
|---|---|---|
| Required | [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | Iteration scheduling; what work can enter/leave between iterations? The presentation page includes talk material. |
| Required | [PagedAttention paper](https://arxiv.org/abs/2309.06180) | Page tables and sharing; distinguish internal fragmentation from reservation. |
| Code | [Mini-SGLang](https://github.com/sgl-project/mini-sglang) | Trace request ownership through scheduler and cache code at a pinned commit. |

Opened during preparation on 2026-09-11. Implement the ownership reference first;
then draw a correspondence between your transitions and the selected code path.
