"""Artifact validation, with CUDA and two-GPU requirements explicitly gated."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import csv

ROOT=Path(__file__).resolve().parents[1]
AVAILABLE=all(importlib.util.find_spec(name) for name in ['torch','matplotlib'])
CUDA=False
TWO_GPU=False
if importlib.util.find_spec('torch'):
    import torch
    CUDA=torch.cuda.is_available()
    TWO_GPU=CUDA and torch.cuda.device_count()>=2


@unittest.skipUnless(AVAILABLE,'PyTorch and Matplotlib required for experiment artifacts')
class StandaloneExperiments(unittest.TestCase):
    def run_chapters(self,numbers):
        expected_rows={'01':30,'02':9,'03':24,'04':36,'05':36,'06':30,'07':12,'08':12}
        with tempfile.TemporaryDirectory(prefix='course-experiments-') as directory:
            for number in numbers:
                chapter=next((ROOT/'chapters').glob(number+'_*'))
                with self.subTest(chapter=number):
                    out=Path(directory)/number
                    command=[sys.executable]
                    if int(number)>=7:
                        command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
                    command += [str(chapter/'code/experiment.py'),'--out',str(out),'--repeats','3']
                    if int(number)>=7:
                        command+=['--backend','nccl']
                    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
                    self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                    with (out/'results.csv').open() as file:
                        rows=list(csv.DictReader(file))
                    self.assertEqual(len(rows),expected_rows[number])
                    manifest=json.loads((out/'manifest.json').read_text())
                    self.assertEqual(manifest['chapter'],number)
                    self.assertEqual(manifest['repeats'],3)
                    self.assertTrue(manifest['source_sha256'])
                    if number in ['01','03','04','06']:
                        self.assertTrue(manifest['hardware']['device'].startswith('cuda'))
                        self.assertIn('CUDA',manifest['result_kind'])
                    elif number in ['02','05']:
                        self.assertIsNone(manifest['timing'])
                    else:
                        self.assertEqual(manifest['backend'],'nccl')
                        self.assertEqual(len(manifest['rank_hardware']),2)
                    self.assertTrue(json.loads((out/'prediction.json').read_text()))
                    summary=json.loads((out/'summary.json').read_text())
                    self.assertTrue(list(out.glob('*.svg')))
                    self.assertTrue(list(out.glob('*.png')))
                    for png in out.glob('*.png'):
                        self.assertTrue(png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                    if number=='01':
                        self.assertEqual(summary['exact_cache_byte_matches'],5)
                    elif number=='02':
                        self.assertTrue(summary['ownership_released'])
                        self.assertTrue(all(int(row['completed'])==90 for row in rows))
                    elif number=='03':
                        self.assertEqual(summary['heldout_points'],6)
                        self.assertEqual(len(summary['relative_errors']),6)
                    elif number=='04':
                        self.assertEqual(summary['matched_cases'],12)
                    elif number=='05':
                        self.assertLessEqual(summary['max_first_token_deviation'],summary['hoeffding_epsilon'])
                    elif number=='06':
                        self.assertEqual(summary['packing'],'passed')
                    else:
                        self.assertEqual(summary['correctness'],'all payloads matched')

    def test_untimed_model_artifacts(self):
        self.run_chapters(['02','05'])

    @unittest.skipUnless(CUDA,'DGX Spark/CUDA GPU required; no CPU timing substitute')
    def test_spark_measurement_artifacts(self):
        self.run_chapters(['01','03','04','06'])

    @unittest.skipUnless(TWO_GPU,'two physical GPUs required for same-host NCCL test; use documented two-Spark launch separately')
    def test_nccl_measurement_artifacts(self):
        self.run_chapters(['07','08'])

    def test_cpu_measurement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            command=[sys.executable,str(ROOT/'chapters/03_performance_model/code/experiment.py'),
                     '--device','cpu','--out',str(Path(directory)/'run')]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('CPU timing fallback is disabled',result.stderr)
            self.assertFalse((Path(directory)/'run').exists())

    @unittest.skipUnless(CUDA and not TWO_GPU,'single-GPU CUDA preflight test')
    def test_one_spark_cannot_be_two_gpu_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            env=dict(os.environ,LOCAL_WORLD_SIZE='2',LOCAL_RANK='0')
            result=subprocess.run([sys.executable,str(ROOT/'chapters/07_tensor_parallelism/code/experiment.py'),
                                   '--out',str(Path(directory)/'run')],env=env,cwd=ROOT,
                                  capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('One DGX Spark has one GPU',result.stderr)

    def test_nonempty_output_is_preserved(self):
        chapter=ROOT/'chapters/02_runtime_and_kv/code/experiment.py'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            marker=path/'prediction.json'
            marker.write_text('{"original": true}')
            result=subprocess.run([sys.executable,str(chapter),'--out',directory],cwd=ROOT,
                                  capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(json.loads(marker.read_text()),{'original':True})


if __name__=='__main__':
    unittest.main()
