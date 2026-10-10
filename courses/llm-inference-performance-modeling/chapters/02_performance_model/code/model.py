"""Chapter 1 Qwen3 reference, exported for Chapter 2 model benchmarks.

The tagged definitions in Chapter 1's lab.ipynb remain the source of truth.
Importing this module executes only those definitions, never checkpoint downloads,
lab demonstrations, or timing cells; no previous notebook kernel is required.

Interface: Config specifies layers, hidden/intermediate widths, query/KV heads,
explicit head_dim, vocabulary, RMSNorm epsilon, and RoPE theta. TinyQwen3 accepts
integer token IDs [B, T] on the model device and returns (logits, cache).
decode=True selects only the final position's logits [B, 1, vocab]; it still
processes every input token. Otherwise logits have shape [B, T, vocab].

The caller owns cache: None starts an unpadded, equal-length batch at position 0;
otherwise pass one (K, V) tuple per layer, each [B, Hkv, P, R]. The returned cache
has length P+T and uses the parameter dtype/device. Positions are P..P+T-1.
Appending allocates new tensors and does not mutate the input cache. This is the
dense eager reference: GQA expands KV heads, QK/AV and softmax run in FP32, and
scores are materialized. It has no SDPA, paging, padding mask, or scaled RoPE.

Use model.to(device=..., dtype=...) for random fixtures, or load_checkpoint for
local safetensors. Call under torch.inference_mode(); loading and correctness
checks belong outside the timed interval. Despite its historical name,
TinyQwen3 also supports the full course 8B and 32B configurations.
"""

import hashlib
from pathlib import Path
import sys

import torch

CHAPTER1_CODE = Path(__file__).resolve().parents[2] / '01_reconstruct_qwen3' / 'code'
if str(CHAPTER1_CODE) not in sys.path:
    sys.path.insert(0, str(CHAPTER1_CODE))

from checkpoint import configuration, load_custom
from notebook_utils import load_notebook_implementation

_implementation = load_notebook_implementation(CHAPTER1_CODE / 'lab.ipynb')
Config = _implementation.Config
RMSNorm = _implementation.RMSNorm
rope = _implementation.rope
Block = _implementation.Block
TinyQwen3 = _implementation.TinyQwen3
model_weight_name_mapping = _implementation.model_weight_name_mapping

__all__ = [
    'Config', 'RMSNorm', 'rope', 'Block', 'TinyQwen3',
    'model_weight_name_mapping', 'load_checkpoint', 'source_hashes',
]


def source_hashes():
    """Record the actual imported definitions and reject edits since import."""
    paths = [Path(__file__), *(CHAPTER1_CODE / name for name in
                             ['lab.ipynb', 'notebook_utils.py', 'checkpoint.py'])]
    hashes = {str(p.relative_to(CHAPTER1_CODE.parents[2])):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if hashes['chapters/01_reconstruct_qwen3/code/lab.ipynb'] != TinyQwen3.__notebook_source_sha256__:
        raise RuntimeError('Chapter 1 definitions changed; restart the kernel before benchmarking.')
    return hashes


def load_checkpoint(snapshot, device='cpu', dtype=torch.float32, *, audit_only=False):
    """Audit/load local dense Qwen3 weights; return (model or None, inventory).

    Reuses Chapter 1's strict header audit and shard loader. No weights are
    allocated for audit_only=True. Loading returns an eval model on the explicit
    device/dtype; CPU is useful for untimed correctness checks only.
    """
    model, inventory = load_custom(
        snapshot, device, dtype, audit_only,
        model_factory=TinyQwen3, name_map_factory=model_weight_name_mapping,
    )
    if model is not None:
        _, metadata = configuration(snapshot)
        model.max_position_embeddings = metadata.get('max_position_embeddings', 32768)
    return model, inventory
