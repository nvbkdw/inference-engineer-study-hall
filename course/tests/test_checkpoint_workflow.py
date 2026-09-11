"""Offline real-format checkpoint audit and independent HF numerical oracle."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'chapters/01_reconstruct_qwen3/code'
AVAILABLE=all(importlib.util.find_spec(name) for name in ['torch','transformers','safetensors','accelerate'])


@unittest.skipUnless(AVAILABLE,'optional checkpoint tools required')
class CheckpointWorkflow(unittest.TestCase):
    def run_command(self,*args,success=True):
        result=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,capture_output=True,text=True,timeout=90)
        if success:
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0)
        return result

    def test_sharded_loader_against_transformers_and_identity_rejection(self):
        with tempfile.TemporaryDirectory(prefix='qwen-loader-') as directory:
            root=Path(directory)
            snapshot=root/'tiny'
            self.run_command(CODE/'make_fixture.py','--out',snapshot)
            self.assertTrue((snapshot/'model.safetensors.index.json').exists())
            base=[CODE/'run_checkpoint.py','--snapshot',snapshot,'--tokens',snapshot/'tokens.json',
                  '--model-revision','local-tiny-v1','--device','cpu']
            self.run_command(*base,'--backend','custom','--audit-only','--out',root/'audit')
            self.assertFalse((root/'audit/logits.safetensors').exists())
            inventory=json.loads((root/'audit/inventory.json').read_text())
            self.assertEqual(len(inventory),25)
            for backend in ['transformers','custom']:
                self.run_command(*base,'--backend',backend,'--out',root/backend)
            compare=[CODE/'compare_logits.py','--reference',root/'transformers','--candidate',root/'custom',
                     '--rtol','0.0001','--atol','0.00001']
            result=self.run_command(*compare)
            summary=json.loads(result.stdout)
            self.assertEqual(summary['positions'],4)
            self.assertEqual(summary['vocabulary'],101)
            path=root/'custom/manifest.json'
            manifest=json.loads(path.read_text())
            manifest['model_revision']='different-checkpoint'
            path.write_text(json.dumps(manifest))
            self.run_command(*compare,success=False)
            # The shape audit must fail before uninitialized weights could run.
            config_path=snapshot/'config.json'
            config=json.loads(config_path.read_text())
            config['head_dim']=10
            config_path.write_text(json.dumps(config))
            result=self.run_command(*base,'--backend','custom','--audit-only','--out',root/'bad-shape',success=False)
            self.assertIn('shape mismatch',result.stderr)


if __name__=='__main__':
    unittest.main()
