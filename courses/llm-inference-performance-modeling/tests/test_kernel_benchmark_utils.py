"""GPU checks for reusable benchmark stream, workspace, and result contracts."""

import importlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch


KERNELS = Path(__file__).resolve().parents[1] / "chapters/03_kernels/exp/kernels"
CUDA = torch.cuda.is_available() and importlib.util.find_spec("cutlass") is not None


@unittest.skipUnless(CUDA, "CuTe and CUDA required for kernel benchmark checks")
class KernelBenchmarkUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(KERNELS))
        cls.addClassCleanup(sys.path.remove, str(KERNELS))
        cls.utils = importlib.import_module("utils")

    def check_samples(self, samples):
        self.assertTrue(samples)
        self.assertTrue(all(math.isfinite(ms) and ms > 0 for ms in samples))

    def test_explicit_stream_and_completion(self):
        original = torch.cuda.current_stream()
        selected = torch.cuda.Stream()
        x = torch.arange(1024, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        observed_streams = []

        def run():
            observed_streams.append(torch.cuda.current_stream())
            torch.square(x, out=out)

        samples = self.utils.benchmark_cuda(run, warmup=2, repeats=3, inner=4, stream=selected)
        self.check_samples(samples)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(stream == selected for stream in observed_streams))
        self.assertEqual(torch.cuda.current_stream(), original)
        torch.testing.assert_close(out, x.square())

    def test_rotating_workspaces_are_created_once(self):
        workspaces = []

        def workspace_generator():
            self.assertFalse(torch.cuda.is_current_stream_capturing())
            x = torch.full((1024,), len(workspaces) + 1, device="cuda", dtype=torch.float32)
            out = torch.empty_like(x)
            workspaces.append((x, out))
            return self.utils.cute_testing.JitArguments(x, out)

        samples = self.utils.benchmark_cuda(
            lambda x, out: torch.mul(x, 2, out=out), warmup=2, repeats=3, inner=4,
            workspace_generator=workspace_generator, workspace_count=2,
        )
        self.check_samples(samples)
        self.assertEqual(len(workspaces), 2)
        for x, out in workspaces:
            torch.testing.assert_close(out, x * 2)

    def test_graph_replays_reuse_buffers_without_calling_python(self):
        original = torch.cuda.current_stream()
        workspaces = []
        calls = []

        def prepare():
            self.assertFalse(torch.cuda.is_current_stream_capturing())
            x = torch.arange(1024, device="cuda", dtype=torch.float32)
            out = torch.empty_like(x)
            count = torch.zeros((), device="cuda", dtype=torch.int32)
            workspaces.append((x, out, count))
            return self.utils.cute_testing.JitArguments(x, out, count)

        def run(x, out, count):
            calls.append((torch.cuda.is_current_stream_capturing(), out.data_ptr()))
            torch.square(x, out=out)
            count.add_(1)

        samples = self.utils.benchmark_cuda(
            run, workspace_generator=prepare, warmup=2, repeats=3, inner=4,
        )
        self.check_samples(samples)
        self.assertEqual(len(workspaces), 1)
        x, out, count = workspaces[0]
        self.assertTrue(any(capturing for capturing, _ in calls))
        self.assertEqual({ptr for _, ptr in calls}, {out.data_ptr()})
        # Replays execute the GPU work without invoking the Python callback.
        self.assertGreater(count.item(), len(calls))
        self.assertEqual(torch.cuda.current_stream(), original)
        torch.testing.assert_close(out, x.square())

    @unittest.skipUnless(importlib.util.find_spec("triton"), "Triton required for L2 eviction timing")
    def test_triton_cold_cache_on_explicit_stream(self):
        original = torch.cuda.current_stream()
        selected = torch.cuda.Stream()
        x = torch.arange(1024, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        observed_streams = []
        captured = []

        def run():
            observed_streams.append(torch.cuda.current_stream())
            captured.append(torch.cuda.is_current_stream_capturing())
            torch.add(x, 1, out=out)

        samples = self.utils.benchmark_cuda_cold(run, warmup_ms=1, repeat_ms=2, stream=selected)
        self.check_samples(samples)
        self.assertTrue(all(stream == selected for stream in observed_streams))
        self.assertTrue(any(captured))
        self.assertEqual(torch.cuda.current_stream(), original)
        torch.testing.assert_close(out, x + 1)

    def test_invalid_configuration(self):
        for kwargs in ({"warmup": 0}, {"repeats": 0}, {"inner": 1.5}, {"workspace_count": 2}, {"device": "cpu"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.utils.benchmark_cuda(lambda: None, **kwargs)
        with self.assertRaisesRegex(TypeError, "torch.cuda.Stream"):
            self.utils.benchmark_cuda(lambda: None, stream=0)
        with self.assertRaisesRegex(ValueError, "cover every workspace"):
            self.utils.benchmark_cuda(
                lambda: None, inner=1, workspace_count=2,
                workspace_generator=lambda: self.utils.cute_testing.JitArguments(),
            )
        with self.assertRaisesRegex(TypeError, "JitArguments"):
            self.utils.benchmark_cuda(lambda: None, workspace_generator=lambda: ())
        with self.assertRaisesRegex(TypeError, "workspace_generator must be callable"):
            self.utils.benchmark_cuda(lambda: None, workspace_generator=1)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            self.utils.benchmark_cuda_cold(lambda: None, repeat_ms=float("nan"))

    def test_rejects_capture_without_cuda_work(self):
        with self.assertRaisesRegex(ValueError, "capture CUDA work"):
            self.utils.benchmark_cuda(lambda: None, warmup=1, repeats=1, inner=1)


if __name__ == "__main__":
    unittest.main()
