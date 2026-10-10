"""
Utility functions for the Efficient AI Lab.
"""

from .config import GEMMA3_270M_CONFIG, GEMMA3_PESUDO_CONFIG, GPU_SPECS, get_ridge_point
from .benchmark import (
    benchmark_latency,
    benchmark_memory,
    count_cuda_kernels,
    warmup,
    default_schedule,
)
from .visualization import (
    plot_roofline,
    set_style,
    plot_gemm_gemv_latency,
    plot_gemm_gemv_latency_split,
    plot_gemm_torch_matmul_latency,
    plot_flash_attn_improvement,
)
from .model import create_prefill_inputs, create_decode_inputs_without_cache, create_decode_inputs_with_cache, \
    get_position_embeddings, initialize_kv_cache, prefill_forward, decode_forward, multi_layer_prefill_forward, multi_layer_decode_forward

__all__ = [
    # Config
    "GEMMA3_270M_CONFIG",
    "GEMMA3_PESUDO_CONFIG",
    "GPU_SPECS", 
    "get_ridge_point",
    # Benchmark
    "benchmark_latency",
    "benchmark_memory",
    "count_cuda_kernels",
    "warmup",
    "default_schedule",
    # Visualization
    "plot_roofline",
    "set_style",
    "plot_gemm_gemv_latency",
    "plot_gemm_gemv_latency_split",
    "plot_gemm_torch_matmul_latency",
    "plot_flash_attn_improvement",
    # Model
    "create_prefill_inputs",
    "create_decode_inputs_without_cache",
    "create_decode_inputs_with_cache",
    "get_position_embeddings",
    "initialize_kv_cache",
    "prefill_forward",
    "decode_forward",
    "multi_layer_prefill_forward",
    "multi_layer_decode_forward",
]
