"""Behavioral checks for course reference code, without weights or GPUs."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(chapter):
    if chapter == '01':
        sys.path.insert(0, str(ROOT/'chapters/01_reconstruct_qwen3/code'))
        from notebook_utils import load_notebook_implementation
        return load_notebook_implementation()
    path = next((ROOT/'chapters').glob(f'{chapter}_*/code/lab.py'))
    name = f'chapter_{chapter}'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class StandardLibraryLabs(unittest.TestCase):
    def test_pool_ownership_and_exhaustion(self):
        module('04').check()

    def test_exact_speculative_distribution(self):
        module('05').check()

    def test_speculative_history_and_bonus(self):
        m = module('05')
        import random
        def alternate(history):
            return [0.,1.] if len(history) % 2 else [1.,0.]
        self.assertEqual(m.cycle([],alternate,alternate,3,random.Random(3),8),([0,1,0,1],3))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required for tensor exercises')
class TensorLabs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_qwen_cache_and_parameter_inventory(self):
        m = module('01')
        import torch
        if not torch.cuda.is_available() or not (ROOT/'models/qwen3-tiny/provenance.json').is_file():
            self.skipTest('Prepared two-layer Qwen3 weights and Spark CUDA required')
        sys.path.insert(0, str(ROOT/'chapters/01_reconstruct_qwen3/code'))
        from notebook_utils import check_cache
        check_cache(m.TinyQwen3, m.model_weight_name_mapping)
        for c, parameters, kv in [
            (m.Config(36,4096,12288,32,8,128,151936),8190735360,147456),
            (m.Config(64,5120,25600,64,8,128,151936),32762123264,262144),
        ]:
            self.assertEqual(c.parameters(),parameters)
            self.assertEqual(c.kv_bytes(1),kv)

    def test_online_attention_partial_tiles(self):
        module('03').check()

    def test_int4_roundtrip_and_error_bound(self):
        module('06').check()

    def test_tensor_parallel_equivalence(self):
        module('07').check()

    def test_handoff_ownership_and_validation(self):
        m = module('08')
        m.check()
        state = m.fixture()
        with self.assertRaises(ValueError):
            state.validate('wrong-revision')
        state.processed += 1
        with self.assertRaises(ValueError):
            state.validate('tiny-fixture-v1')
        self.assertAlmostEqual(m.transfer_ms(2*2**30,25),85.89934592)


if __name__ == '__main__':
    unittest.main()
