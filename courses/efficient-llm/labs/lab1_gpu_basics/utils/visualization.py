"""
Visualization utilities for the Efficient AI Lab.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_roofline(
    gpu_name: str,
    peak_flops: float,  # TFLOPS
    peak_bandwidth: float,  # TB/s
    operations: Optional[Dict[str, float]] = None,  # name -> arithmetic intensity
    x_range: Optional[Tuple[float, float]] = None,  # (x_min, x_max) in FLOPs/Byte
    y_range: Optional[Tuple[float, float]] = None,  # (y_min, y_max) in TFLOPS
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a roofline model for the given GPU.
    
    Args:
        gpu_name: Name of GPU for title
        peak_flops: Peak FP16 TFLOPS
        peak_bandwidth: Peak memory bandwidth in TB/s
        operations: Dict mapping operation names to their arithmetic intensity (FLOPs/Byte)
        x_range: Optional manual x-axis limits (x_min, x_max) in FLOPs/Byte
        y_range: Optional manual y-axis limits (y_min, y_max) in TFLOPS
        title: Optional custom title
        ax: Optional axes to plot on
        
    Returns:
        matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate ridge point
    ridge_point = peak_flops / peak_bandwidth  # FLOPs/Byte
    
    # Create roofline
    # x-axis: arithmetic intensity (FLOPs/Byte), log scale
    # y-axis: attainable performance (TFLOPS), log scale

    # Pick a neat view window automatically based on passed-in AIs.
    # This keeps very small AIs (e.g., < 1) visible without requiring callers to guess ranges.
    ABS_X_MIN, ABS_X_MAX = 1e-6, 1e8
    ABS_Y_MIN = 1e-6
    PAD_LOW, PAD_HIGH = 0.5, 2.0  # multiplicative padding in log-space

    if x_range is not None:
        x_min, x_max = x_range
    else:
        ai_vals: List[float] = []
        if operations:
            for ai in operations.values():
                if ai is None:
                    continue
                try:
                    ai_f = float(ai)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(ai_f) and ai_f > 0:
                    ai_vals.append(ai_f)

        # Always include ridge point so the knee is visible.
        candidates = ai_vals + [ridge_point] if np.isfinite(ridge_point) and ridge_point > 0 else ai_vals

        if candidates:
            cmin, cmax = float(np.min(candidates)), float(np.max(candidates))
            x_min = max(ABS_X_MIN, cmin * PAD_LOW)
            x_max = min(ABS_X_MAX, cmax * PAD_HIGH)
        else:
            # Backward-compatible defaults when nothing is provided.
            x_min, x_max = 0.1, 5000

        # Avoid degenerate ranges (e.g., single AI value).
        if not (np.isfinite(x_min) and np.isfinite(x_max)) or x_min <= 0 or x_max <= 0 or x_min >= x_max:
            x_min, x_max = 0.1, 5000

    x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    
    # Memory-bound region: performance = bandwidth * AI
    # Compute-bound region: performance = peak_flops
    y_memory = peak_bandwidth * x  # TB/s * FLOPs/Byte = TFLOPS
    y_compute = np.full_like(x, peak_flops)
    y = np.minimum(y_memory, y_compute)
    
    # Plot roofline
    ax.loglog(x, y, 'b-', linewidth=2.5, label='Roofline')
    
    # Fill regions
    ax.fill_between(x, 0.01, y, where=(x < ridge_point), alpha=0.2, color='red', 
                    label='Memory-bound region')
    ax.fill_between(x, 0.01, y, where=(x >= ridge_point), alpha=0.2, color='green',
                    label='Compute-bound region')
    
    # Mark ridge point
    ax.axvline(x=ridge_point, color='gray', linestyle='--', linewidth=1.5)
    
    # Plot operations if provided
    if operations:
        colors = plt.cm.tab10(np.linspace(0, 1, len(operations)))
        for (name, ai), color in zip(operations.items(), colors):
            # Attainable performance at this AI
            perf = min(peak_bandwidth * ai, peak_flops)
            ax.scatter([ai], [perf], s=50, c=[color], zorder=5, edgecolor='black', linewidth=1.5, label=name)
    
    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity', fontsize=12)
    ax.set_ylabel('Attainable Performance (TFLOPS)', fontsize=12)
    ax.set_title(title or f'Roofline Model - {gpu_name}', fontsize=14, fontweight='bold')
    
    ax.set_xlim(x_min, x_max)

    if y_range is not None:
        y_min, y_max = y_range
    else:
        # Ensure both memory-bound and compute-bound portions look good.
        y_min = max(ABS_Y_MIN, min(peak_bandwidth * x_min, peak_flops) * 0.5)
        y_max = peak_flops * 1.5
        if y_min >= y_max:
            y_min = max(ABS_Y_MIN, y_max / 1000.0)
    ax.set_ylim(y_min, y_max)
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, which='both', linestyle='-', alpha=0.3)
    
    # Add peak annotations
    ax.annotate(f'Peak: {peak_flops} TFLOPS', 
                xy=(x_min * (x_max / x_min) ** 0.6, peak_flops * 1.02),
                fontsize=12, va='bottom')
    
    plt.tight_layout()
    return ax


def plot_gemm_gemv_latency(
    sizes: List[int],
    gemm_latency_ms: List[float],
    gemv_latency_ms: List[float],
    title: Optional[str] = None,
    xlabel: str = 'Matrix dimension (N)',
    ylabel: str = 'Latency (ms)',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot latency of GEMM and GEMV across matrix sizes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, gemm_latency_ms, 'o-', label='GEMM', linewidth=2, markersize=8)
    ax.plot(sizes, gemv_latency_ms, 's-', label='GEMV', linewidth=2, markersize=8)

    ax.set_xticks(sizes)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax

def plot_gemm_torch_matmul_latency(
    sizes: List[int],
    gemm_tiled_latency_ms: List[float],
    torch_matmul_latency_ms: List[float],
    title: Optional[str] = None,
    xlabel: str = 'Matrix dimension (N)',
    ylabel: str = 'Latency (ms)',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot latency of gemm_tiled and torch.matmul across matrix sizes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, gemm_tiled_latency_ms, 'o-', label='gemm_tiled', linewidth=2, markersize=8)
    ax.plot(sizes, torch_matmul_latency_ms, 's-', label='torch.matmul', linewidth=2, markersize=8)

    ax.set_xticks(sizes)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_gemm_gemv_latency_split(
    sizes: List[int],
    gemm_latency_ms: List[float],
    gemv_latency_ms: List[float],
    title: Optional[str] = 'GEMM vs GEMV Latency',
    xlabel: str = 'Matrix dimension (N)',
    ylabel: str = 'Latency (ms)',
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot GEMM and GEMV latencies on two vertically stacked subplots.
    
    Args:
        sizes: Matrix sizes (N).
        gemm_latency_ms: Latencies for GEMM.
        gemv_latency_ms: Latencies for GEMV.
        title: Optional figure title.
        xlabel: Shared x-axis label.
        ylabel: Y-axis label for both subplots.
        
    Returns:
        Figure and array of Axes objects (GEMM on top, GEMV on bottom).
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax_gemm, ax_gemv = axes

    ax_gemm.plot(sizes, gemm_latency_ms, 'o-', label='GEMM', linewidth=2, markersize=8)
    ax_gemm.set_ylabel(ylabel, fontsize=12)
    ax_gemm.set_title('GEMM Latency', fontsize=13, fontweight='bold')
    ax_gemm.grid(True, alpha=0.3)

    ax_gemv.plot(sizes, gemv_latency_ms, 's-', label='GEMV', linewidth=2, markersize=8, color='tab:orange')
    ax_gemv.set_xticks(sizes)
    ax_gemv.set_xlabel(xlabel, fontsize=12)
    ax_gemv.set_ylabel(ylabel, fontsize=12)
    ax_gemv.set_title('GEMV Latency', fontsize=13, fontweight='bold')
    ax_gemv.grid(True, alpha=0.3)

    ax_gemm.legend(fontsize=10)
    ax_gemv.legend(fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig, axes


def plot_flash_attn_improvement(
    df: pd.DataFrame,
    *,
    title: Optional[str] = "FlashAttention vs Standard Attention",
    latency_cols: Tuple[str, str] = ("std_ms", "flash_ms"),
    memory_cols: Tuple[str, str] = ("std_mem_gb", "flash_mem_gb"),
    seq_len_col: str = "seq_len",
    annotate_speedup: bool = True,
    annotate_mem_reduction: bool = True,
    bar_width: float = 0.38,
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize the improvement of FlashAttention over standard attention with two bar-plot subplots:
    latency (ms) and memory (GB).

    Expected columns by default:
      - seq_len, std_ms, flash_ms, std_mem_gb, flash_mem_gb
    (Speedup is optional; if not present it is computed as std_ms / flash_ms for annotations.)

    Args:
        df: DataFrame with benchmark results.
        title: Optional figure title.
        latency_cols: (standard_latency_col, flash_latency_col).
        memory_cols: (standard_memory_col, flash_memory_col).
        seq_len_col: Column name for sequence length.
        annotate_speedup: If True, annotate latency subplot with speedup (std/flash).
        annotate_mem_reduction: If True, annotate memory subplot with reduction ratio (std/flash).
        bar_width: Width of each bar in grouped bar charts.
        ax: Optional array-like of 2 Axes to plot into (latency_ax, memory_ax).

    Returns:
        (fig, axes) where axes is an array: [latency_ax, memory_ax]
    """
    if df is None or len(df) == 0:
        raise ValueError("df must be a non-empty DataFrame.")

    std_lat_col, flash_lat_col = latency_cols
    std_mem_col, flash_mem_col = memory_cols

    required = {seq_len_col, std_lat_col, flash_lat_col, std_mem_col, flash_mem_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")

    data = df.copy()
    data[seq_len_col] = pd.to_numeric(data[seq_len_col], errors="coerce")
    data[std_lat_col] = pd.to_numeric(data[std_lat_col], errors="coerce")
    data[flash_lat_col] = pd.to_numeric(data[flash_lat_col], errors="coerce")
    data[std_mem_col] = pd.to_numeric(data[std_mem_col], errors="coerce")
    data[flash_mem_col] = pd.to_numeric(data[flash_mem_col], errors="coerce")
    data = data.dropna(subset=[seq_len_col, std_lat_col, flash_lat_col, std_mem_col, flash_mem_col])
    data = data.sort_values(seq_len_col)

    seq_lens = data[seq_len_col].astype(int).to_numpy()
    std_ms = data[std_lat_col].to_numpy()
    flash_ms = data[flash_lat_col].to_numpy()
    std_mem = data[std_mem_col].to_numpy()
    flash_mem = data[flash_mem_col].to_numpy()

    x = np.arange(len(seq_lens))
    offset = bar_width / 2

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    else:
        axes = np.asarray(ax)
        if axes.size != 2:
            raise ValueError("ax must be an array-like of exactly 2 matplotlib Axes: [latency_ax, memory_ax].")
        fig = axes.flat[0].figure

    ax_lat, ax_mem = axes.flat[0], axes.flat[1]

    # Latency subplot
    ax_lat.bar(x - offset, std_ms, width=bar_width, label="standard_attn", color="tab:blue", alpha=0.9)
    ax_lat.bar(x + offset, flash_ms, width=bar_width, label="flash_attn", color="tab:orange", alpha=0.9)
    ax_lat.set_title("Latency", fontsize=13, fontweight="bold")
    ax_lat.set_ylabel("Latency (ms)", fontsize=12)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([str(s) for s in seq_lens])
    ax_lat.set_xlabel("seq_len", fontsize=12)
    ax_lat.grid(True, axis="y", alpha=0.3)
    ax_lat.legend(fontsize=10)

    # Memory subplot
    ax_mem.bar(x - offset, std_mem, width=bar_width, label="standard_attn", color="tab:blue", alpha=0.9)
    ax_mem.bar(x + offset, flash_mem, width=bar_width, label="flash_attn", color="tab:orange", alpha=0.9)
    ax_mem.set_title("Memory", fontsize=13, fontweight="bold")
    ax_mem.set_ylabel("Peak memory (GB)", fontsize=12)
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels([str(s) for s in seq_lens])
    ax_mem.set_xlabel("seq_len", fontsize=12)
    ax_mem.grid(True, axis="y", alpha=0.3)
    ax_mem.legend(fontsize=10)

    # Optional annotations (ratio improvement)
    if annotate_speedup:
        speedup = np.divide(std_ms, flash_ms, out=np.full_like(std_ms, np.nan, dtype=float), where=flash_ms != 0)
        y_max = np.nanmax(np.maximum(std_ms, flash_ms))
        for xi, sp in zip(x, speedup):
            if np.isfinite(sp):
                ax_lat.text(xi, y_max * 1.02, f"{sp:.2f}×", ha="center", va="bottom", fontsize=10)
        ax_lat.set_ylim(top=y_max * 1.15)

    if annotate_mem_reduction:
        mem_ratio = np.divide(std_mem, flash_mem, out=np.full_like(std_mem, np.nan, dtype=float), where=flash_mem != 0)
        y_max = np.nanmax(np.maximum(std_mem, flash_mem))
        for xi, mr in zip(x, mem_ratio):
            if np.isfinite(mr):
                ax_mem.text(xi, y_max * 1.02, f"{mr:.2f}×", ha="center", va="bottom", fontsize=10)
        ax_mem.set_ylim(top=y_max * 1.15)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig, axes



# Initialize style when module is imported
set_style()
