"""Mechanism/call-contract tests; GPU fixtures are not real-model measurements."""

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

opt = importlib.import_module("chapters.03_kernels.code.model_opt")
CUDA = torch.cuda.is_available() and importlib.util.find_spec("cutlass") is not None


class OptimizedContract(unittest.TestCase):
    def test_parameters_fallback_and_cache_ownership(self):
        torch.manual_seed(33)
        c = opt.reference.Config(
            layers=1,
            hidden=64,
            intermediate=96,
            q_heads=4,
            kv_heads=2,
            head_dim=16,
            vocab=128,
        )
        baseline = opt.reference.TinyQwen3(c).eval()
        candidate = opt.OptimizedQwen3(c, strict=False).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        ids = torch.tensor([[11, 25, 76, 93], [5, 9, 12, 42]])
        with torch.inference_mode():
            expected, _ = baseline(ids)
            _, cache = candidate(ids[:, :2])
            saved = [(k.clone(), v.clone()) for k, v in cache]
            actual, new = candidate(ids[:, 2:], cache)
        torch.testing.assert_close(actual, expected[:, 2:])
        self.assertTrue(candidate.fallbacks)
        self.assertEqual(new[0][0].shape, (2, 2, 4, 16))
        for old, copy in zip(cache, saved):
            for a, b in zip(old, copy):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        strict = opt.OptimizedQwen3(c, strict=True)
        with self.assertRaisesRegex(RuntimeError, "Unexpected.*fallback"):
            strict(ids)

    def test_aggregation_never_mixes_implementations(self):
        from shared.performance import (
            summarize_model_rows,
            summarize_tradeoff_observations,
        )

        rows = []
        for impl, ms in [("baseline", 10.0), ("optimized", 5.0)]:
            for repeat in range(3):
                for step in range(8):
                    rows.append(
                        dict(
                            model="8b",
                            implementation=impl,
                            batch=1,
                            prompt=2048,
                            phase="decode",
                            repeat=repeat,
                            step=step,
                            prefix_tokens=2048 + step,
                            token_ready_ms=ms,
                            cuda_ms=ms,
                            throughput_tokens=1,
                            flops=100,
                            bytes_proxy=100,
                        )
                    )
        summary = summarize_model_rows(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual([r["latency_ms"] for r in summary], [10.0, 5.0])
        tradeoffs = summarize_tradeoff_observations(rows, 2048)
        self.assertEqual([r["repeats"] for r in tradeoffs], [3, 3])


@unittest.skipUnless(
    CUDA, "CuTe and CUDA required; this is unmeasured GPU scope otherwise"
)
class CuteOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = opt.kernels()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(2026)

    def tensor(self, *shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    def test_widths_and_partial_rows(self):
        for width in (128, 4096, 5120, 12288, 25600):
            x = self.tensor(5, width)
            u = self.tensor(5, width)
            norm = opt.reference.RMSNorm(width).to("cuda", torch.bfloat16)
            norm.weight.data.uniform_(0.5, 1.5)
            torch.testing.assert_close(
                self.k.rmsnorm(x, norm.weight), norm(x), rtol=0.02, atol=0.02
            )
            for block in (128, 256):
                torch.testing.assert_close(
                    self.k.swiglu(x, u, block),
                    torch.nn.functional.silu(x) * u,
                    rtol=0.02,
                    atol=0.02,
                )

    def test_qk_head_norm_rotation_and_absolute_positions(self):
        for heads in (32, 64):
            q = self.tensor(2, 3, heads, 128)
            k = self.tensor(2, 3, 8, 128)
            norm = opt.reference.RMSNorm(128).to("cuda", torch.bfloat16)
            norm.weight.data.uniform_(0.5, 1.5)
            for prefix in (0, 17, 2048):
                positions = torch.arange(prefix, prefix + 3, device="cuda").expand(2, 3)
                a, b = self.k.qk_norm_rope(q, k, norm.weight, norm.weight, positions)
                for source, actual in [(q, a), (k, b)]:
                    expected = opt.reference.rope(
                        norm(source).transpose(1, 2), positions, 1e6
                    )
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    def test_attention_tiles_prefix_extremes_and_strides(self):
        dense = importlib.import_module("chapters.03_kernels.code.lab").dense_attention
        for heads in (32, 64):
            for s in (1, 31, 32, 33, 127, 128, 129):
                for t in sorted({1, min(3, s), s}):
                    # Projection transpose stride + contiguous cache are both exercised.
                    q = self.tensor(1, t, heads, 128).transpose(1, 2)
                    k = self.tensor(1, s, 8, 128).transpose(1, 2)
                    v = self.tensor(1, s, 8, 128).transpose(1, 2)
                    if s % 2:
                        k = k.contiguous()
                        v = v.contiguous()
                    if s == 33:
                        q = q * 10
                        k = k * 10
                    expected = dense(q.float(), k.float(), v.float(), s - t)
                    for tile in (17, 32):
                        actual = self.k.attention(q, k, v, key_tile=tile)
                        torch.testing.assert_close(
                            actual.float(), expected, rtol=0.02, atol=0.02
                        )
        q = torch.full((1, 4, 1, 128), 1e17, device="cuda", dtype=torch.bfloat16)
        k = torch.full((1, 2, 3, 128), -1e17, device="cuda", dtype=torch.bfloat16)
        v = self.tensor(1, 2, 3, 128)
        torch.testing.assert_close(
            self.k.attention(q, k, v).float(),
            dense(q.float(), k.float(), v.float(), 2),
            rtol=0.02,
            atol=0.02,
        )
        q = self.tensor(4, 32, 1, 128)
        k = self.tensor(4, 8, 2049, 128)
        v = self.tensor(4, 8, 2049, 128)
        torch.testing.assert_close(
            self.k.attention(q, k, v).float(),
            dense(q.float(), k.float(), v.float(), 2048),
            rtol=0.02,
            atol=0.02,
        )

    def test_rotary_products_round_before_addition(self):
        # Integer squares make the RMS reduction exact; isolate rotary rounding.
        # A BF16 FMA skips one required intermediate round and fails this check.
        torch.manual_seed(44)
        q = torch.randint(-4, 5, (2, 7, 32, 128), device="cuda").bfloat16()
        k = q[:, :, :8].contiguous()
        positions = torch.arange(7, device="cuda").expand(2, 7)
        norm = opt.reference.RMSNorm(128).to("cuda", torch.bfloat16)
        actual, _ = self.k.qk_norm_rope(q, k, norm.weight, norm.weight, positions)
        expected = opt.reference.rope(norm(q).transpose(1, 2), positions, 1e6)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_head_norm_matches_reference_reduction_order(self):
        # Zero rotation isolates FP32 reduction order at BF16 rounding boundaries.
        for heads in (32, 64):
            q = self.tensor(4, 7, heads, 128)
            k = self.tensor(4, 7, 8, 128)
            positions = torch.zeros((4, 7), device="cuda", dtype=torch.int64)
            norm = opt.reference.RMSNorm(128).to("cuda", torch.bfloat16)
            norm.weight.data.uniform_(0.5, 1.5)
            aq, ak = self.k.qk_norm_rope(q, k, norm.weight, norm.weight, positions)
            torch.testing.assert_close(aq, norm(q).transpose(1, 2), atol=0, rtol=0)
            torch.testing.assert_close(ak, norm(k).transpose(1, 2), atol=0, rtol=0)

    def test_jit_cache_distinguishes_broadcast_position_strides(self):
        # B=1 positions have a dynamic nonzero first stride; expand(B>1,T)
        # has compile-time zero stride in CuTe. Exercise both compilation orders.
        for order in ((1, 4, 2, 1), (4, 1, 2, 4)):
            self.k.COMPILED.clear()
            for batch in order:
                t = 129
                q = self.tensor(batch, t, 32, 128)
                k = self.tensor(batch, t, 8, 128)
                positions = torch.arange(17, 17 + t, device="cuda").expand(batch, t)
                norm = opt.reference.RMSNorm(128).to("cuda", torch.bfloat16)
                actual, _ = self.k.qk_norm_rope(
                    q, k, norm.weight, norm.weight, positions
                )
                torch.testing.assert_close(
                    actual,
                    opt.reference.rope(norm(q).transpose(1, 2), positions, 1e6),
                    atol=0.02,
                    rtol=0.02,
                )

    def test_current_stream_and_contract_rejection(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = self.tensor(5, 128)
            x.mul_(2)
            result = self.k.swiglu(x, x)
            expected = torch.nn.functional.silu(x) * x
        stream.synchronize()
        torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
        with self.assertRaisesRegex(ValueError, "CUDA BF16"):
            self.k.swiglu(x.float(), x.float())
        with self.assertRaises(ValueError):
            self.k.attention(
                self.tensor(1, 3, 1, 128),
                self.tensor(1, 2, 3, 128),
                self.tensor(1, 2, 3, 128),
            )

    def test_block_and_cached_logits_with_all_ablations(self):
        c = opt.reference.Config(
            layers=2,
            hidden=256,
            intermediate=512,
            q_heads=4,
            kv_heads=2,
            head_dim=128,
            vocab=128,
        )
        reference = opt.reference.TinyQwen3(c).to("cuda", torch.bfloat16).eval()
        ids = torch.tensor([[11, 25, 76, 93], [5, 9, 12, 42]], device="cuda")
        with torch.inference_mode():
            expected, _ = reference(ids)
            for preset in opt.PRESETS.values():
                model = (
                    opt.OptimizedQwen3(c, preset, strict=True)
                    .to("cuda", torch.bfloat16)
                    .eval()
                )
                model.load_state_dict(reference.state_dict())
                actual, _ = model(ids)
                _, cache = model(ids[:, :2])
                saved = [(k.clone(), v.clone()) for k, v in cache]
                chunk, new = model(ids[:, 2:], cache)
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=0.02, atol=0.03
                )
                torch.testing.assert_close(
                    chunk.float(), actual[:, 2:].float(), rtol=0.02, atol=0.03
                )
                for old, copy in zip(cache, saved):
                    for a, b in zip(old, copy):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                self.assertFalse(model.fallbacks)
                self.assertEqual(new[0][0].shape, (2, 2, 4, 128))


if __name__ == "__main__":
    unittest.main()
