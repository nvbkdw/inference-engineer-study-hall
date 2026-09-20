"""Untimed RoPE regression against the installed Transformers oracle."""

import importlib
from pathlib import Path
import sys
import unittest

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
reference = importlib.import_module('chapters.03_kernels.exp.model_ref')


class RotaryOracle(unittest.TestCase):
    def check_device(self, device):
        generator = torch.Generator().manual_seed(33)
        for head_dim in (80, 128):
            with self.subTest(device=device, head_dim=head_dim):
                config = Qwen3Config(head_dim=head_dim, rope_theta=1e6)
                oracle = Qwen3RotaryEmbedding(config).to(device)
                # Nonzero absolute positions also exercise cached decoding.
                positions = torch.tensor([[0, 1, 21, 25], [127, 128, 2048, 8191]], device=device)
                q = torch.randn(2, 4, 4, head_dim, generator=generator).to(device)
                k = torch.randn(2, 2, 4, head_dim, generator=generator).to(device)
                cos, sin = oracle(q, positions)
                expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
                torch.testing.assert_close(reference.rope(q, positions, 1e6), expected_q, rtol=0, atol=0)
                torch.testing.assert_close(reference.rope(k, positions, 1e6), expected_k, rtol=0, atol=0)

    def test_cpu_matches_oracle(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GPU pow regression')
    def test_cuda_matches_oracle(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            self.check_device('cuda')
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


if __name__ == '__main__':
    unittest.main()
