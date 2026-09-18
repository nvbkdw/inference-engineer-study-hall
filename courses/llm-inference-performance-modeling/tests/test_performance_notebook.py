"""Check the notebook's actual equations and benchmark boundaries without full weights."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'chapters/02_performance_model/code/lab.ipynb'
AVAILABLE = all(importlib.util.find_spec(m) for m in ['numpy', 'matplotlib', 'IPython'])


def notebook_module():
    """Execute only definition cells: no calibration, download, or full-model work."""
    module = types.ModuleType('performance_notebook')
    sys.modules[module.__name__] = module
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if 'definitions' in cell.get('metadata', {}).get('tags', []):
            exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), module.__dict__)
    return module


@unittest.skipUnless(AVAILABLE, 'Notebook numerical/plot dependencies required')
class PerformanceNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = notebook_module()

    def test_hardware_units_and_boundaries(self):
        m = self.m
        self.assertEqual(m.roofline_ms(2e12, 1e9, 2, 100), 1000)
        self.assertAlmostEqual(m.roofline_ms(0, 1e9, 2, 100, 5), 10.005)
        self.assertEqual(m.gemm(4, 8, 16), (1024, 448))
        with self.assertRaises(ValueError):
            m.roofline_ms(1, 1, float('nan'), 100)
        with self.assertRaises(ValueError):
            m.decoder_layer_flops(m.MODELS['8b'], 1, 0)
        with self.assertRaises(ValueError):
            m.model_work(m.MODELS['8b'], 1, 2, logits_tokens=3)
        with self.assertRaises(RuntimeError):
            m.require_cuda('cpu', 'bfloat16')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
    def test_insufficient_capacity_is_rejected_before_loading(self):
        import torch
        from unittest.mock import patch
        with patch.object(torch.cuda, 'mem_get_info', return_value=(8*1024**3, 8*1024**3)), \
             patch.object(torch.cuda, 'get_device_name', return_value='8 GiB fixture GPU'):
            with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, 'Capacity preflight'):
                self.m.check_capacity(self.m.MODELS['8b'], 'cuda:0', [(1, 128)], 8)

    def test_qwen_inventory_and_worked_examples(self):
        m = self.m
        for name, parameters, prefill, decode in [
            ('8b', 8190735360, 29688662589440, 16344743936),
            ('32b', 32762123264, 132219976548352, 68264132608),
        ]:
            c = m.MODELS[name]
            self.assertEqual(c.parameters, parameters)
            self.assertEqual(m.model_work(c, 1, 2048)['flops'], prefill)
            self.assertEqual(m.model_work(c, 1, 1, 2048)['flops'], decode)
        self.assertEqual(m.MODELS['32b'].query_width, 8192)
        self.assertNotEqual(m.MODELS['32b'].query_width, m.MODELS['32b'].hidden)

    def test_causal_pairs_chunking_and_off_by_one(self):
        m = self.m
        c = m.MODELS['32b']
        batch, prefix, tokens = 2, 7, 5
        # Enumerate legal pairs as an independent oracle, including each query itself.
        pairs = batch * sum(sum(k <= q for k in range(prefix+tokens)) for q in range(prefix, prefix+tokens))
        work = m.decoder_layer_flops(c, batch, tokens, prefix)
        self.assertEqual(work['qk'] + work['av'], 4*c.query_width*pairs)
        chunk = sum(work.values())
        steps = sum(sum(m.decoder_layer_flops(c, batch, 1, prefix+j).values()) for j in range(tokens))
        self.assertEqual(chunk, steps)
        for mode in ['causal', 'dense']:
            one = m.decoder_layer_flops(c, 1, 1, 2048, mode)
            self.assertEqual(one['qk'], 2*2049*c.query_width)
        dense = m.decoder_layer_flops(c, batch, tokens, prefix, 'dense')
        self.assertGreater(dense['qk'], work['qk'])

    def test_logits_and_traffic_scaling(self):
        m = self.m
        c = m.MODELS['8b']
        last = m.model_work(c, 2, 16)
        all_logits = m.model_work(c, 2, 16, logits_tokens=16)
        self.assertEqual(all_logits['flops']-last['flops'], 2*2*15*c.hidden*c.vocab)
        one = m.model_work(c, 1, 1, 100)
        two = m.model_work(c, 2, 1, 100)
        self.assertEqual(two['flops'], 2*one['flops'])
        self.assertEqual(two['weight_bytes'], one['weight_bytes'])
        self.assertEqual(two['kv_read_bytes'], 2*one['kv_read_bytes'])
        self.assertEqual(one['kv_write_bytes'], 147456)
        self.assertGreater(two['intensity'], one['intensity'])

    def test_mfu_aggregates_work_and_time(self):
        m = self.m
        hw = m.Hardware('fixture', 200, {'bfloat16': 100}, 'assumed', 'fixture', 'fixture')
        measured = m.metrics(dict(flops=1e12, bytes_proxy=1e9, intensity=1000), .1, hw, 'bfloat16')
        self.assertAlmostEqual(measured['mfu'], .1)
        rows = []
        for repeat in range(3):
            for f, t in [(1e12, 100), (9e12, 300)]:
                rows.append(dict(model='8b', batch=1, prompt=2, phase='decode', repeat=repeat,
                                 flops=f, bytes_proxy=1e9, cuda_ms=t, token_ready_ms=t+1))
        summary = m.summarize_model_rows(rows, hw, 'bfloat16')[0]
        self.assertEqual(summary['achieved_tflops'], 25)  # 10 TFLOP / 0.4 seconds, not mean(10, 30).
        self.assertEqual(summary['mfu'], .25)
        self.assertEqual(summary['mfu_basis'], 'assumed')
        self.assertEqual(summary['token_ready_ms'], 201)

    @unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('torch'), 'PyTorch and Transformers required')
    def test_ledger_matches_actual_qwen_linear_shapes(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1)
        # Query width deliberately differs from hidden width, as in 32B.
        config = Qwen3Config(vocab_size=128, hidden_size=48, intermediate_size=80,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, tie_word_embeddings=False)
        model = Qwen3ForCausalLM(config).eval()
        c = self.m.ModelDimensions.from_config(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), c.parameters)
        counts = []
        def count_linear(module, inputs, output):
            counts.append(2 * output.numel() * module.in_features)
        hooks = [module.register_forward_hook(count_linear) for module in model.modules() if isinstance(module, torch.nn.Linear)]
        with torch.inference_mode():
            ids = torch.tensor([[11, 25, 76, 93]])
            full = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
        for hook in hooks:
            hook.remove()
        ledger = self.m.decoder_layer_flops(c, 1, 4)
        attention = c.layers * (ledger['qk'] + ledger['av'])
        self.assertEqual(sum(counts), self.m.model_work(c, 1, 4)['flops']-attention)
        # Exercise the same shard-at-a-time loader used by the full GPU notebook.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, max_shard_size='30KB')
            loaded = self.m.load_resident_model(directory, 'cpu', 'float32', 'eager')
            reference = self.m.reference_model_module()
            self.assertIsInstance(loaded, reference.TinyQwen3)
            mapping = reference.model_weight_name_mapping(loaded.c)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(dict(loaded.named_parameters())[mapping[name]], parameter, rtol=0, atol=0)
            hooks = [module.register_forward_hook(count_linear) for module in loaded.modules()
                     if isinstance(module, torch.nn.Linear)]
            counts.clear()
            with torch.inference_mode():
                restored, _ = loaded(ids, decode=True)
            for hook in hooks:
                hook.remove()
            self.assertEqual(sum(counts), self.m.model_work(c, 1, 4)['flops']-attention)
            torch.testing.assert_close(restored, full, rtol=1e-5, atol=1e-6)
            self.assertLess(self.m.check_cached_logits(loaded, 'cpu'), 1e-5)
            self.assertIn('chapters/01_reconstruct_qwen3/code/lab.ipynb', reference.source_hashes())
            with self.assertRaisesRegex(ValueError, 'only eager'):
                self.m.load_resident_model(directory, 'cpu', 'float32', 'sdpa')

            # Batched chunk append must preserve the prefix and use absolute positions.
            ids = torch.tensor([[11, 25, 76, 93], [5, 9, 12, 42]])
            with torch.inference_mode():
                expected = model(input_ids=ids, use_cache=False).logits
                _, prefix = loaded(ids[:, :2])
                saved = [(k.clone(), v.clone()) for k, v in prefix]
                actual, cache = loaded(ids[:, 2:], prefix)
            torch.testing.assert_close(actual, expected[:, 2:], rtol=1e-5, atol=1e-6)
            for (k, v), (old_k, old_v), (saved_k, saved_v) in zip(cache, prefix, saved):
                self.assertEqual(k.shape, (2, 2, 4, 16))
                self.assertEqual(v.shape, k.shape)
                torch.testing.assert_close(old_k, saved_k, rtol=0, atol=0)
                torch.testing.assert_close(old_v, saved_v, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
