## High level outline 
Qwen3 optimization

profile -> hypothesis -> predict speed up -> optimization -> measure speed up -> explain (does it apply to differnt input?)

1) profile the whole model Qwen3-8B
2) Pick kernel operations (conver all model into kernels):
    * RMSNorm
    * SwiGLU
    * Q/K norm + RoPE
    * Self-attention
3) Fine tune kernel dims on both 8B and 32B
4) Measure performance speedup, with ablation study across four corners (prefill bs1 and bs4, decode bs1 and bs4 )
5) Put all together, summarize, explain each speedup
