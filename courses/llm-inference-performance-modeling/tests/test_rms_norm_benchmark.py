"""Synthetic GPU checks for prepared RMSNorm work and graph timing artifacts."""

import csv
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch


KERNELS = Path(__file__).resolve().parents[1] / "chapters/03_kernels/exp/kernels"
CUDA = torch.cuda.is_available() and importlib.util.find_spec("cutlass") is not None


@unittest.skipUnless(CUDA, "CuTe and CUDA required for RMSNorm benchmark checks")
class RMSNormBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(KERNELS))
        cls.addClassCleanup(sys.path.remove, str(KERNELS))
        cls.kernel = importlib.import_module("rms_norm")
        cls.utils = importlib.import_module("utils")

    def test_prepared_backends_reuse_output_and_read_updated_inputs(self):
        torch.manual_seed(2026)
        selected = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(selected):
            x = torch.randn((5, 128), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            for prepare in (self.kernel._prepare_cute_benchmark, self.kernel._prepare_torch_benchmark):
                with self.subTest(backend=prepare.__name__):
                    run, output = prepare(x, weight, 1e-6)
                    pointer = output.data_ptr()
                    for multiplier in (1.0, 0.5):
                        x.mul_(multiplier)
                        weight.add_(0.25)
                        samples = self.utils.benchmark_cuda(
                            run, stream=selected, warmup=2, repeats=3, inner=4,
                        )
                        x64 = x.double()
                        expected = x64 / (x64.square().mean(-1, keepdim=True) + 1e-6).sqrt()
                        expected *= weight.double()
                        torch.testing.assert_close(output.double(), expected, rtol=4e-3, atol=1e-5)
                        self.assertEqual(output.data_ptr(), pointer)
                        self.assertTrue(all(math.isfinite(ms) and ms > 0 for ms in samples))

    def test_public_output_buffer_and_legacy_allocation(self):
        x = torch.ones((5, 128), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x)
        self.assertIs(self.kernel.rms_norm(x, weight, 1e-6, output), output)
        torch.testing.assert_close(output, self.kernel.rms_norm(x, weight, 1e-6))
        with self.assertRaisesRegex(ValueError, "Output shape"):
            self.kernel.rms_norm(x, weight, 1e-6, torch.empty_like(x[:1]))

    def test_benchmark_records_graph_and_buffer_scope(self):
        with tempfile.TemporaryDirectory(prefix="rmsnorm-graph-test-") as directory:
            out = Path(directory)
            summary = self.kernel.benchmark(shape=(5, 128), repeats=3, inner=4, out=out)
            manifest = json.loads((out / "manifest.json").read_text())
            with (out / "results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(manifest["cuda_graphs"])
            self.assertTrue(manifest["preallocated_buffers"])
            self.assertEqual(manifest["workspace_count"], 1)
            self.assertTrue(summary["cuda_graphs"])
            self.assertTrue(summary["preallocated_buffers"])
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(float(row["cuda_ms"]) > 0 for row in rows))
            self.assertTrue((out / "prediction.json").exists())


if __name__ == "__main__":
    unittest.main()
