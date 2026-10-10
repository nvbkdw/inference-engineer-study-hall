# Beginner inspection exercise

Human usability sessions have not yet been conducted. Automated interaction tests do not substitute for this exercise.

Ask a CuTeDSL learner to bring a small compiling kernel and add probes beside a tensor, a copy, and an MMA operation. Let them use the README and viewer without coaching. Record where they hesitate, the terms they misunderstand, and any point where the capture cannot explain their kernel.

1. Select a tensor element and explain its logical coordinate and relative storage offset.
2. Select another nested mode or fixed slice. Explain what changed in the coordinate calculation.
3. Find the selected copy element's destination and its source/destination thread/value owners.
4. Find an MMA operand's role and storage. Distinguish warp/warpgroup participation from ownership.
5. Compare shared-memory bank placement and an explicit scalar access group. Explain why a shared bank color alone is insufficient to claim a conflict.
6. Identify one unresolved field and explain what additional information would be needed.

Use the learner's explanations to revise labels and examples. Keep runtime values, physical register allocation, and unsupported conflict models out of the expected answers. Do not record the learner's source or publish their capture without their permission.
