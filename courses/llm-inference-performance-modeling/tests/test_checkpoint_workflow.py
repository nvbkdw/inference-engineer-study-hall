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
AVAILABLE=(ROOT/'models/qwen3-tiny/provenance.json').is_file() and all(importlib.util.find_spec(name) for name in ['torch','transformers','safetensors','accelerate'])


@unittest.skipUnless(AVAILABLE,'checkpoint tools and prepared real two-layer weights required')
class CheckpointWorkflow(unittest.TestCase):
    def test_notebook_definitions_skip_lab_actions_and_cover_model_parameters(self):
        import torch
        sys.path.insert(0, str(CODE))
        from notebook_utils import load_notebook_implementation
        notebook=json.loads((CODE/'lab.ipynb').read_text())
        notebook['cells'].insert(0,dict(cell_type='code',id='must-not-run',metadata={},
                                      source=["raise RuntimeError('lab action executed')"]))
        # Even a demonstration in a definition cell must not execute on import.
        definitions=next(cell for cell in notebook['cells']
                         if 'qwen3-definition' in cell.get('metadata',{}).get('tags',[]))
        definitions['source'] += ["\nraise RuntimeError('demonstration executed')\n"]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lab.ipynb'
            path.write_text(json.dumps(notebook))
            m=load_notebook_implementation(path)
            for layers,hidden,intermediate,q_heads,expected in [
                (2,4096,12288,32,1630556672),
                (36,4096,12288,32,8190735360),
                (64,5120,25600,64,32762123264),
            ]:
                config=m.Config(layers=layers,hidden=hidden,intermediate=intermediate,q_heads=q_heads)
                with torch.device('meta'):
                    model=m.TinyQwen3(config)
                names=m.model_weight_name_mapping(config)
                self.assertEqual(set(names.values()),set(dict(model.named_parameters())))
                self.assertEqual(len(set(names.values())),len(names))
                self.assertEqual(sum(p.numel() for p in model.parameters()),expected)

    def run_command(self,*args,success=True):
        result=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,capture_output=True,text=True,timeout=90)
        if success:
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0)
        return result

    def test_sharded_loader_against_transformers_and_identity_rejection(self):
        import torch
        from safetensors.torch import load_file, save_file
        sys.path.insert(0, str(CODE))
        from notebook_utils import compare_checkpoint_logits
        if not torch.cuda.is_available():
            self.skipTest('Spark CUDA required for the real-weight FP32 acceptance baseline')
        with tempfile.TemporaryDirectory(prefix='qwen-loader-') as directory:
            root=Path(directory)
            snapshot=root/'tiny'
            self.run_command(CODE/'make_fixture.py','--out',snapshot,'--offline')
            self.assertTrue((snapshot/'model.safetensors.index.json').exists())
            base=[CODE/'run_checkpoint.py','--snapshot',snapshot,'--tokens',snapshot/'tokens.json',
                  '--model-revision','Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218:layers-0-1','--device','cuda:0']
            self.run_command(*base,'--backend','custom','--audit-only','--out',root/'audit')
            self.assertFalse((root/'audit/logits.safetensors').exists())
            inventory=json.loads((root/'audit/inventory.json').read_text())
            self.assertEqual(len(inventory),25)
            for backend in ['transformers','custom']:
                self.run_command(*base,'--backend',backend,'--out',root/backend)
            def compare():
                return compare_checkpoint_logits(root/'transformers',root/'custom',
                                                 rtol=0.0001,atol=0.00001,display=False)
            summary=compare()
            self.assertEqual(summary['positions'],5)
            self.assertEqual(summary['vocabulary'],151936)
            # A changed low-ranking logit must fail even when the top token agrees.
            logits_path=root/'custom/logits.safetensors'
            logits=load_file(str(logits_path))['logits']
            top=logits[0].argmax().item()
            logits[0,(top+1)%logits.shape[1]]-=1.
            self.assertEqual(logits[0].argmax().item(),top)
            save_file({'logits':logits},str(logits_path))
            with self.assertRaises(AssertionError):
                compare()
            path=root/'custom/manifest.json'
            manifest=json.loads(path.read_text())
            manifest['model_revision']='different-checkpoint'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'different model_revision'):
                compare()
            # The shape audit must fail before uninitialized weights could run.
            config_path=snapshot/'config.json'
            config=json.loads(config_path.read_text())
            config['head_dim']=10
            config_path.write_text(json.dumps(config))
            result=self.run_command(*base,'--backend','custom','--audit-only','--out',root/'bad-shape',success=False)
            self.assertIn('shape mismatch',result.stderr)


if __name__=='__main__':
    unittest.main()
