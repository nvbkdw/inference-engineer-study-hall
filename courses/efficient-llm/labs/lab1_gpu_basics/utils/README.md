# Helper Functions Reference

All helper functions are imported via `from utils import *`. Below is the API documentation.

## Configuration (`utils/config.py`)

### `GEMMA3_270M_CONFIG`
Pre-configured `Gemma3TextConfig` for the Gemma-3-270m-it model with the following key parameters:
- `hidden_size`: 640
- `intermediate_size`: 2048
- `num_attention_heads`: 4
- `num_key_value_heads`: 1
- `head_dim`: 256
- `num_hidden_layers`: 18

### `GEMMA3_PESUDO_CONFIG`
Pre-configured `Gemma3TextConfig` for a pseudo Gemma-3 model with larger dimensions:
- `hidden_size`: 1152
- `intermediate_size`: 6912
- `num_attention_heads`: 8
- `num_key_value_heads`: 8
- `num_hidden_layers`: 26

### `GPU_SPECS`
Dictionary of GPU specifications for roofline analysis. Supported GPUs: `A6000`, `A10`, `A100`, `H100`, `H200`, `4090`.

Each entry contains:
- `fp16_tflops`, `bf16_tflops`, `fp32_tflops`: Peak compute in TFLOPS
- `memory_bandwidth_tb_s`: Memory bandwidth in TB/s
- `memory_gb`: Total GPU memory in GB
- `sram_kb`: SRAM per streaming multiprocessor in KB
- `num_sms`: Number of streaming multiprocessors

**Example:**
```python
gpu_spec = GPU_SPECS["A100"]
peak_flops = gpu_spec["fp32_tflops"]      # 19.5 TFLOPS
peak_bw = gpu_spec["memory_bandwidth_tb_s"]  # 2.0 TB/s
num_sms = gpu_spec["num_sms"]              # 108
```

### `get_ridge_point`
Calculate the ridge point (FLOPs/Byte) for the roofline model.

**Returns:** Ridge point where memory-bound transitions to compute-bound.

---

## Benchmarking (`utils/benchmark.py`)

### `warmup`
Warmup function to stabilize CUDA timing before benchmarking.

**Args:**
- `fn`: Function to warmup
- `n_warmup`: Number of warmup iterations (default: 10)

### `benchmark_latency`
Benchmark a function and return mean latency in milliseconds.

**Args:**
- `fn`: Function to benchmark
- `n_warmup`: Number of warmup iterations (default: 10)
- `n_repeat`: Number of timed iterations (default: 100)

**Returns:** Mean latency in ms.

### `benchmark_memory`
Benchmark function with memory tracking.

**Args:**
- `fn`: Function to benchmark
- `n_warmup`: Number of warmup iterations (default: 10)
- `n_repeat`: Number of timed iterations (default: 100)

**Returns:** `(peak_memory_gb, allocated_memory_gb)`

### `count_cuda_kernels`
Count CUDA kernels from a profiler result by category.

**Args:**
- `prof`: PyTorch profiler result

**Returns:** Dict with keys: `gemm`, `elementwise`, `softmax`, `layernorm`, `memory`, `other`, `total`

### `default_schedule`
Pre-configured profiler schedule: `wait=1, warmup=3, active=6`

---

## Visualization (`utils/visualization.py`)

### `set_style`
Set consistent plot style for all visualizations. Automatically called when the module is imported.

### `plot_roofline`
Plot a roofline model for the given GPU.

**Args:**
- `gpu_name`: Name of GPU for title
- `peak_flops`: Peak TFLOPS
- `peak_bandwidth`: Peak memory bandwidth in TB/s
- `operations`: Optional dict mapping operation names to arithmetic intensity (FLOPs/Byte)
- `x_range`: Optional tuple `(x_min, x_max)` to manually set x-axis limits
- `y_range`: Optional tuple `(y_min, y_max)` to manually set y-axis limits
- `title`: Optional custom title
- `ax`: Optional matplotlib Axes to plot on

**Returns:** matplotlib Axes object

**Example:**
```python
ai_to_plot = {
    'gemm_1024': 341.3,
    'gemv_1024': 1.0,
}
plot_roofline("A100", 19.5, 2.0, operations=ai_to_plot)
```

### `plot_gemm_gemv_latency`
Plot latency comparison between GEMM and GEMV operations.

**Args:**
- `sizes`: List of matrix dimensions
- `gemm_latency_ms`: List of GEMM latencies in ms
- `gemv_latency_ms`: List of GEMV latencies in ms
- `title`: Optional plot title
- `xlabel`, `ylabel`: Axis labels
- `ax`: Optional matplotlib Axes

**Returns:** matplotlib Axes object

### `plot_gemm_gemv_latency_split`
Plot GEMM and GEMV latencies on two vertically stacked subplots.

**Args:**
- `sizes`: List of matrix dimensions
- `gemm_latency_ms`: List of GEMM latencies in ms
- `gemv_latency_ms`: List of GEMV latencies in ms
- `title`: Optional figure title
- `xlabel`, `ylabel`: Axis labels

**Returns:** `(fig, axes)` where axes is array of 2 Axes objects

### `plot_gemm_torch_matmul_latency`
Plot latency comparison between `gemm_tiled` and `torch.matmul`.

**Args:**
- `sizes`: List of matrix dimensions
- `gemm_tiled_latency_ms`: List of tiled GEMM latencies in ms
- `torch_matmul_latency_ms`: List of torch.matmul latencies in ms
- `title`: Optional plot title
- `xlabel`, `ylabel`: Axis labels
- `ax`: Optional matplotlib Axes

**Returns:** matplotlib Axes object

### `plot_flash_attn_improvement`
Visualize FlashAttention vs standard attention with latency and memory bar plots.

**Args:**
- `df`: DataFrame with benchmark results
- `latency_cols`: Tuple of `(standard_latency_col, flash_latency_col)`
- `memory_cols`: Tuple of `(standard_memory_col, flash_memory_col)`
- `seq_len_col`: Column name for sequence length
- `title`: Optional figure title
- `annotate_speedup`: If True, annotate with speedup ratios
- `annotate_mem_reduction`: If True, annotate with memory reduction ratios
- `bar_width`: Width of bars in grouped bar charts
- `ax`: Optional array of 2 Axes `[latency_ax, memory_ax]`

**Returns:** `(fig, axes)` where axes is array of 2 Axes objects

**Expected DataFrame columns (defaults):** `seq_len`, `std_ms`, `flash_ms`, `std_mem_gb`, `flash_mem_gb`

---

## Model Utilities (`utils/model.py`)

### `create_prefill_inputs`
Create pseudo inputs for prefill stage.

**Args:**
- `batch_size`: Batch size
- `seq_len`: Sequence length
- `hidden_size`: Hidden dimension
- `dtype`: Data type (e.g., torch.float32, torch.bfloat16)
- `device`: Device (e.g., 'cuda', 'cpu')

**Returns:** `(hidden_states, position_ids)` tensors

### `create_decode_inputs_with_cache`
Create pseudo inputs for decode stage with KV cache.

**Args:**
- `batch_size`: Batch size
- `current_seq_len`: Current sequence length in cache
- `hidden_size`: Hidden dimension
- `dtype`: Data type
- `device`: Device

**Returns:** `(hidden_states, position_ids, cache_positions)` tensors

### `create_decode_inputs_without_cache`
Create pseudo inputs for decode stage without KV cache.

**Args:**
- `batch_size`: Batch size
- `current_seq_len`: Current sequence length
- `hidden_size`: Hidden dimension
- `dtype`: Data type
- `device`: Device

**Returns:** `(hidden_states, position_ids, prefill_hidden_states)` tensors

### `get_position_embeddings`
Compute rotary position embeddings.

**Args:**
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs tensor
- `rotary_emb`: Rotary embedding layer
- `layer_type`: Type of attention layer

**Returns:** `(cos, sin)` tensors

### `initialize_kv_cache`
Initialize a static KV cache for the given config.

**Args:**
- `config`: Model configuration
- `max_cache_len`: Maximum cache length

**Returns:** `StaticCache` object

### `prefill_forward`
Run prefill forward pass through a decoder layer.

**Args:**
- `layer`: Decoder layer
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `past_key_values`: KV cache

**Returns:** Output hidden states

### `decode_forward`
Run decode forward pass through a decoder layer.

**Args:**
- `layer`: Decoder layer
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `use_kv_cache`: Whether to use KV cache
- `past_key_values`: KV cache
- `cache_positions`: Cache position indices
- `prefill_hidden_states`: Hidden states from prefill (if not using cache)

**Returns:** Output hidden states

### `multi_layer_prefill_forward`
Run prefill forward pass through multiple decoder layers.

**Args:**
- `num_layers`: Number of layers to iterate through
- `layer`: Decoder layer (will be reused for all layers)
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `past_key_values`: KV cache

**Returns:** Output hidden states after all layers

### `multi_layer_decode_forward`
Run decode forward pass through multiple decoder layers.

**Args:**
- `num_layers`: Number of layers to iterate through
- `layer`: Decoder layer (will be reused for all layers)
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `use_kv_cache`: Whether to use KV cache
- `past_key_values`: KV cache
- `cache_positions`: Cache position indices
- `prefill_hidden_states`: Hidden states from prefill (if not using cache)

**Returns:** Output hidden states after all layers
