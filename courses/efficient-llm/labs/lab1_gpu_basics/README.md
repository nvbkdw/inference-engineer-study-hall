# Lab1: Efficient AI Fundamentals

A hands-on lab for understanding efficiency fundamentals in AI through a detailed look at transformer decoder layers (latency, MAC/FLOPs, I/O, profiling, and Flash Attention).

- **lab1(colab).ipynb:** For you to complete on Google Colab (images are embedded in the notebook). Upload the whole `lab1_gpu_basics` folder to your Google Drive, open this notebook from there, and run the Setup cell first.
- **lab1.ipynb:** Same content, for running locally on your own NVIDIA GPU (images are loaded from `assets/`, so they don't render on Colab).

**Please submit only one notebook.**

## Learning Objectives

By completing this lab, students will be able to:

1. **Measure latency and compute MAC/FLOPs/I/O** for common linear algebra ops (gemm/gemv)
2. **Use roofline models** to reason about compute- vs memory-bound regimes
3. **Profile GPU kernels** with PyTorch Profiler and interpret kernel timelines
4. **Reduce launch overhead** via kernel fusion, `torch.compile`, and CUDA Graphs
5. **Analyze Flash Attention** to explain its speed and memory advantages

## Prerequisites

- PyTorch 2.0+
- Familiar with basic deep learning concepts such as neural networks
- Basic understanding of transformers and attention mechanisms

## Installation

```bash
# Install miniconda (only needed when running locally)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Install cudatoolkit
conda install nvidia::cuda-nvcc==12.4.99

# Install dependencies (these versions are required: newer `transformers`
# releases changed an API that `utils/config.py` relies on)
pip install "transformers==4.57.6" "tokenizers==0.22.2" \
            "huggingface_hub==0.36.2" "accelerate==1.11.0"

# Flash Attention (only needed for Part 5)
pip install --no-build-isolation flash-attn
```

**On Colab** you do not need any of the above except the pinned versions: the notebook's
Setup section installs them for you. Run that cell, restart the runtime when prompted, and
continue. Colab's preinstalled `transformers` is too new and will fail on import otherwise.

## Lab Structure

The lab is organized in `lab1.ipynb` with 5 parts. **Total Points: 80 points + 20 bonus points**

### Part 1: Fundamental Metrics (20 points)
- Implement a naive (tri-loop) GEMM; measure latency
- Compute MAC/FLOPs and I/O for gemm/gemv

| Question | Description | Points |
|----------|-------------|--------|
| 1.1 | Matrix Multiplication Implementation | 5 |
| 1.2 | Latency Measurement Analysis | 5 |
| 1.3 | MAC/FLOPs Calculation for GEMM/GEMV | 5 |
| 1.4 | I/O Calculation for GEMM/GEMV | 5 |

### Part 2: Roofline Model (15 points)
- Plot arithmetic intensity and rooflines for different sizes
- Compare gemm vs gemv positioning

| Question | Description | Points |
|----------|-------------|--------|
| 2.1.1 | GEMM Roofline Plot | 5 |
| 2.1.2 | Compute-Bound Threshold Calculation | 5 |
| 2.2 | GEMV Roofline Analysis | 5 |

### Part 3: Case Study — Gemma-3 Decoder Layer (30 points)
- MAC/FLOPs and I/O for attention/MLP paths
- Prefill vs decode analysis (KV cache) and latency measurements

| Question | Description | Points |
|----------|-------------|--------|
| 3.1 | Attention MAC/FLOPs Calculation | 10 |
| 3.2 | Attention I/O Calculation | 5 |
| 3.4.1 | Prefill vs Decode Performance Analysis | 5 |
| 3.4.2 | Batch Size Impact on Roofline | 5 |
| 3.4.3 | Prefill Length Impact on Roofline | 5 |

### Part 4: Profiling & Kernel Fusion (15 points)
- PyTorch Profiler usage and Chrome trace export
- Kernel fusion, `torch.compile`, and CUDA Graphs effects on launches

| Question | Description | Points |
|----------|-------------|--------|
| 4.3.1 | GeLU Implementation | 3 |
| 4.3.2 | GeLU Latency Benchmark | 2 |
| 4.3.3 | GeLU Benchmark Findings | 2 |
| 4.3.4 | GeLU Profiler Analysis | 5 |
| 4.4.1 | Compiled MLP Profiling | 3 |

### Part 5: Flash Attention (10 bonus points)
- Benchmark SDPA vs Flash Attention (latency and memory)
- Kernel-count reduction and roofline implications

**All questions are moved to bonus**

<!-- | Question | Description | Points |
|----------|-------------|--------|
| 5.1 | Standard Attention Memory Complexity | 5 |
| | • Sub-question 1: Memory complexity analysis | (1.5) |
| | • Sub-question 2: Problematic sequence length | (1.5) |
| | • Sub-question 3: Major overhead identification | (2) |
| 5.2 | Flash Attention Performance Analysis | 5 |
| | • Sub-question 1: Memory-bound explanation | (1.5) |
| | • Sub-question 2: Flash Attention mechanism | (2) |
| | • Sub-question 3: Speedup vs sequence length | (1.5) | -->

### Bonus Questions (20 points)

| Question | Description | Points |
|----------|-------------|--------|
| bonus_3.1 | Attention Module MAC Calculation | 2.5 |
| bonus_3.2 | Attention Module I/O Calculation (Prefill) | 2.5 |
| bonus_4.2 | Decoder Layer Profiling (Decode) | 2.5 |
| bonus_4.4.2 | CUDA Graphs Profiling | 2.5 |
| bonus_5.1 | Flash attention v.s SDPA | 5 |
| bonus_5.2 | Flash attention analysis | 5 |

## File Structure

```
lab1/
├── README.md               # This file
├── lab1.ipynb              # Main lab notebook
├── assets/                 # Figures used in the notebook
└── utils/
    ├── __init__.py
    ├── benchmark.py        # Benchmarking utilities
    ├── config.py           # Model and GPU configurations
    ├── model.py            # Helper functions for inference
    └── visualization.py    # Plotting functions
```

## Helper Functions Reference

For detailed documentation of all helper functions (configuration, benchmarking, visualization, and model utilities), see **[utils/README.md](utils/README.md)**.

## Acknowledgment

We thank Zhijian Liu's Z Lab for providing this lab.

