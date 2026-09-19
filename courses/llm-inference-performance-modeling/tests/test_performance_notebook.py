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

    def test_analytical_tradeoff_units_and_batch_scaling(self):
        m = self.m
        # Deliberately compute-bound fixture: 1 TFLOP/s and effectively unlimited bandwidth.
        hw = m.Hardware('analytical fixture', 1e12, {'bfloat16': 1}, 'assumed', 'fixture', 'fixture')
        points = m.predict_tradeoffs(m.MODELS['8b'], [1, 2], 2048, hw, 'bfloat16')
        prefill = [r for r in points if r['phase'] == 'prefill']
        decode = [r for r in points if r['phase'] == 'decode']
        self.assertAlmostEqual(prefill[0]['latency_ms'], 29688.66258944)
        self.assertAlmostEqual(prefill[0]['total_tokens_per_second'], 2048/29.68866258944)
        self.assertIsNone(prefill[0]['interactivity_tokens_per_second'])
        self.assertEqual(prefill[0]['token_kind'], 'input')
        self.assertAlmostEqual(decode[0]['latency_ms'], 16.344743936)
        self.assertEqual(decode[0]['attended_positions'], 2049)
        self.assertEqual(decode[0]['prefix_tokens'], 2048)
        self.assertEqual(decode[0]['token_kind'], 'output')
        for phase in [prefill, decode]:
            self.assertAlmostEqual(phase[1]['latency_ms'], 2*phase[0]['latency_ms'])
            self.assertAlmostEqual(phase[1]['total_tokens_per_second'], phase[0]['total_tokens_per_second'])
            self.assertEqual(phase[0]['limiting_term'], 'compute')
        self.assertAlmostEqual(decode[1]['interactivity_tokens_per_second'],
                               decode[0]['interactivity_tokens_per_second']/2)
        for r in decode:
            self.assertAlmostEqual(r['total_tokens_per_second'],
                                   r['batch']*r['interactivity_tokens_per_second'])

    def test_analytical_decode_weight_amortization(self):
        m = self.m
        hw = m.Hardware('memory-bound fixture', 100, {'bfloat16': 1e6}, 'assumed', 'fixture', 'fixture')
        rows = m.predict_tradeoffs(m.MODELS['32b'], [1, 4], 2048, hw, 'bfloat16')
        one, four = [r for r in rows if r['phase'] == 'decode']
        self.assertEqual(one['limiting_term'], 'memory')
        self.assertGreater(four['total_tokens_per_second'], one['total_tokens_per_second'])
        self.assertLess(four['interactivity_tokens_per_second'], one['interactivity_tokens_per_second'])
        for batches, context in [([], 2048), ([0], 2048), ([1], 0)]:
            with self.assertRaises(ValueError):
                m.predict_tradeoffs(m.MODELS['8b'], batches, context, hw, 'bfloat16')

    def test_decode_throughput_aggregates_tokens_and_time(self):
        m = self.m
        measured = m.metrics(dict(flops=1e12, bytes_proxy=1e9), .1)
        self.assertAlmostEqual(measured['achieved_tflops'], 10)
        for invalid in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                m.metrics(dict(flops=1, bytes_proxy=1), invalid)
        rows = []
        for repeat, scale in enumerate([1, 2, 3]):
            for step, (f, t) in enumerate([(1e12, 100), (9e12, 300)]):
                rows.append(dict(model='8b', batch=4, prompt=128, phase='decode', repeat=repeat,
                                 flops=f, bytes_proxy=1e9, cuda_ms=t*scale, token_ready_ms=t*scale,
                                 throughput_tokens=4, prefix_tokens=128+step))
        summary = m.summarize_model_rows(rows)[0]
        # At the median repeat: 8 emitted tokens / 0.8 s, not mean(4/0.2, 4/0.6).
        self.assertEqual(summary['tokens_per_second'], 10)
        self.assertEqual(summary['tokens_per_second_max'], 20)
        self.assertAlmostEqual(summary['tokens_per_second_min'], 8/1.2)
        self.assertEqual(summary['latency_ms'], 400)
        self.assertEqual(summary['latency_min_ms'], 200)
        self.assertEqual(summary['latency_max_ms'], 600)
        self.assertEqual(summary['throughput_token_kind'], 'output')
        self.assertEqual(summary['prefix_tokens_min'], 128)
        self.assertEqual(summary['prefix_tokens_max'], 129)
        self.assertEqual(summary['achieved_tflops'], 12.5)  # 10 TFLOP / 0.8 s.
        self.assertEqual(summary['repeats'], 3)

    def test_tradeoff_overlay_matches_fixed_context_and_first_decode_call(self):
        rows = []
        for repeat, ms in enumerate([100, 200, 400]):
            for prompt in [128, 512]:
                for phase, prefix in [('prefill', 0), ('decode', prompt), ('decode', prompt+1)]:
                    rows.append(dict(model='8b', batch=2, prompt=prompt, phase=phase, repeat=repeat,
                        prefix_tokens=prefix, flops=1e9, bytes_proxy=1e8,
                        cuda_ms=ms, token_ready_ms=1 if prefix == prompt+1 else ms,
                        throughput_tokens=2*prompt if phase == 'prefill' else 2))
        points = self.m.summarize_tradeoff_observations(rows, 128)
        self.assertEqual(len(points), 2)
        prefill = next(r for r in points if r['phase'] == 'prefill')
        decode = next(r for r in points if r['phase'] == 'decode')
        self.assertEqual(prefill['latency_ms'], 200)
        self.assertEqual(prefill['total_tokens_per_second'], 1280)
        self.assertEqual(decode['interactivity_tokens_per_second'], 5)
        self.assertEqual(decode['total_tokens_per_second'], 10)
        self.assertEqual(decode['interactivity_tokens_per_second_min'], 2.5)
        self.assertEqual(decode['interactivity_tokens_per_second_max'], 10)
        self.assertEqual(decode['repeats'], 3)
        self.assertEqual(self.m.summarize_tradeoff_observations(rows, 2048), [])

        # With two repeats, median(1/t) differs from 1/median(t).
        even = self.m.summarize_tradeoff_observations([r for r in rows if r['repeat'] < 2], 128)
        decode = next(r for r in even if r['phase'] == 'decode')
        self.assertEqual(decode['interactivity_tokens_per_second'], 7.5)
        self.assertEqual(decode['total_tokens_per_second'], 15)

        hw = self.m.Hardware('fixture', 273, {'bfloat16': 125}, 'assumed', 'fixture', 'fixture')
        predictions = [dict(model='8b', **r) for r in
                      self.m.predict_tradeoffs(self.m.MODELS['8b'], [1, 2], 128, hw, 'bfloat16')]
        fig, axes = self.m.plot_tradeoffs(predictions, 128, 'bfloat16', 'assumed', observed=points)
        try:
            # Real observations are drawn as errorbar markers on both panels.
            self.assertEqual(len(axes[0].containers), 1)
            self.assertEqual(len(axes[1].containers), 1)
            x, y = axes[1].containers[0].lines[0].get_data()
            self.assertEqual(list(x), [5])
            self.assertEqual(list(y), [10])
            fig.canvas.draw()
        finally:
            self.m.plt.close(fig)

    def test_prefill_throughput_counts_input_tokens_and_uses_wall_time(self):
        rows = [dict(model='8b', batch=2, prompt=128, phase='prefill', repeat=r,
                     flops=1e12, bytes_proxy=1e9, cuda_ms=100, token_ready_ms=t,
                     throughput_tokens=256, prefix_tokens=0)
                for r, t in enumerate([200, 400, 600])]
        summary = self.m.summarize_model_rows(rows)[0]
        self.assertEqual(summary['tokens_per_second'], 640)  # 256 input tokens / 0.4 s.
        self.assertEqual(summary['throughput_token_kind'], 'input')
        self.assertEqual(summary['latency_ms'], 400)
        self.assertEqual(summary['forward_ms'], 100)
        rows[0]['token_ready_ms'] = 0
        with self.assertRaises(ValueError):
            self.m.summarize_model_rows(rows)

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
