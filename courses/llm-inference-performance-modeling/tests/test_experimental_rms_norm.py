"""Untimed GPU correctness fixtures for the experimental tiled RMSNorm."""

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


KERNELS = Path(__file__).resolve().parents[1] / "chapters/03_kernels/exp/kernels"
CUDA = torch.cuda.is_available() and importlib.util.find_spec("cutlass") is not None


@unittest.skipUnless(CUDA, "CuTe and CUDA required for the experimental RMSNorm")
class ExperimentalRMSNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The experimental module currently supports sibling imports directly.
        sys.path.insert(0, str(KERNELS))
        cls.addClassCleanup(sys.path.remove, str(KERNELS))
        cls.kernel = importlib.import_module("rms_norm")

    def setUp(self):
        torch.manual_seed(2026)

    def check_oracle(self, x, weight, eps):
        actual = self.kernel.rms_norm(x, weight, eps)
        # Independent FP64 oracle checks the reduction and final BF16 rounding.
        x64 = x.double()
        expected = x64 / (x64.square().mean(-1, keepdim=True) + eps).sqrt()
        expected = expected * weight.double()
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.device, x.device)
        torch.testing.assert_close(actual.double(), expected, rtol=4e-3, atol=1e-5)

    def test_all_warps_reuse_the_same_weights(self):
        x = torch.ones((8, 128), device="cuda", dtype=torch.bfloat16)
        weight = torch.arange(1, 129, device="cuda", dtype=torch.bfloat16)
        actual = self.kernel.rms_norm(x, weight, 1e-6)
        torch.testing.assert_close(actual, weight.expand_as(x), rtol=0, atol=0)

    def test_partial_blocks_and_original_fixture(self):
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        for shape in ((1, 128), (5, 128), (3, 2, 128), (1024, 16, 128)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                self.check_oracle(x, weight, 1e-6)

    def test_widths_zero_rows_and_epsilon(self):
        for width in (32, 128, 4096, 5120):
            x = torch.randn((5, width), device="cuda", dtype=torch.bfloat16)
            x[-1].zero_()
            weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            for eps in (1e-6, 1e-3):
                with self.subTest(width=width, eps=eps):
                    self.check_oracle(x, weight, eps)

    def test_rejects_unmasked_partial_columns(self):
        for width in (1, 31, 33, 127, 129):
            with self.subTest(width=width):
                x = torch.ones((4, width), device="cuda", dtype=torch.bfloat16)
                weight = torch.ones(width, device="cuda", dtype=torch.bfloat16)
                with self.assertRaisesRegex(ValueError, "width divisible by 32"):
                    self.kernel.rms_norm(x, weight, 1e-6)


if __name__ == "__main__":
    unittest.main()
