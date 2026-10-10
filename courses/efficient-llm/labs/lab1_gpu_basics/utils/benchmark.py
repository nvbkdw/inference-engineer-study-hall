"""
Benchmark utilities for the Efficient AI Lab.
"""

import torch
import time
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
from contextlib import contextmanager
from torch.profiler import schedule


default_schedule = schedule(
    wait=1,       # Skip the very first step (often has high overhead)
    warmup=3,     # Run 3 steps to warm up the cache and CUDA
    active=6,     # Profile and average these 6 steps
)


def warmup(fn: Callable, *args, n_warmup: int = 10, **kwargs):
    """Warmup function to stabilize CUDA timing."""
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()


def benchmark_latency(
    fn: Callable,
    *args,
    n_warmup: int = 10,
    n_repeat: int = 100,
    **kwargs
) -> float:
    """
    Benchmark a function and return mean latency in ms.
    
    Returns:
        latency_ms
    """
    warmup(fn, n_warmup=n_warmup, *args, **kwargs)
    
    times = []
    for _ in range(n_repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn(*args, **kwargs)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times = torch.tensor(times)
    return times.mean().item()


def benchmark_memory(
    fn: Callable,
    *args,
    n_warmup: int = 10,
    n_repeat: int = 100,
    **kwargs
) -> Tuple[float, float]:
    """
    Benchmark function with memory tracking.
    
    Returns:
        (peak_memory_gb, allocated_memory_gb)
    """
    warmup(fn, n_warmup=n_warmup, *args, **kwargs)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    for _ in range(n_repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn(*args, **kwargs)
        end.record()
        
        torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    allocated_memory = torch.cuda.memory_allocated() / 1e9
    
    return peak_memory, allocated_memory


def count_cuda_kernels(prof) -> Dict[str, int]:
    """
    Count CUDA kernels from a profiler result.
    
    Returns:
        dict with kernel category counts
    """
    kernel_counts = {
        "gemm": 0,
        "elementwise": 0,
        "softmax": 0,
        "layernorm": 0,
        "memory": 0,  # memcpy, memset
        "other": 0,
    }
    
    for event in prof.key_averages():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            name = event.key.lower()
            if "gemm" in name or "matmul" in name or "mm_" in name or "cublas" in name:
                kernel_counts["gemm"] += event.count
            elif "softmax" in name:
                kernel_counts["softmax"] += event.count
            elif "layernorm" in name or "layer_norm" in name or "rms_norm" in name:
                kernel_counts["layernorm"] += event.count
            elif "elementwise" in name or "add" in name or "mul" in name or "silu" in name:
                kernel_counts["elementwise"] += event.count
            elif "memcpy" in name or "memset" in name:
                kernel_counts["memory"] += event.count
            else:
                kernel_counts["other"] += event.count
    
    kernel_counts["total"] = sum(v for k, v in kernel_counts.items() if k != "total")
    return kernel_counts
