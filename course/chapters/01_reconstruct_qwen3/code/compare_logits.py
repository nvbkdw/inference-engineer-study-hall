"""Compare saved full-vocabulary vectors only when checkpoint/fixture identities match."""
import argparse
import json
from pathlib import Path
from safetensors.torch import load_file
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--rtol',type=float,required=True)
    p.add_argument('--atol',type=float,required=True)
    a=p.parse_args()
    if min(a.rtol,a.atol)<0:
        p.error('nonnegative tolerances required')
    ref_meta=json.loads((a.reference/'manifest.json').read_text())
    new_meta=json.loads((a.candidate/'manifest.json').read_text())
    for key in ['model_revision','config_sha256','fixture_sha256','dtype']:
        if ref_meta[key]!=new_meta[key]:
            raise ValueError(f'cannot compare different {key}')
    ref=load_file(str(a.reference/'logits.safetensors'))['logits']
    candidate=load_file(str(a.candidate/'logits.safetensors'))['logits']
    torch.testing.assert_close(candidate,ref,rtol=a.rtol,atol=a.atol)
    relative=(candidate-ref).norm()/ref.norm().clamp_min(1e-20)
    print(json.dumps(dict(correctness='passed',positions=ref.shape[0],vocabulary=ref.shape[1],
                          max_abs_error=float((candidate-ref).abs().max()),relative_l2=float(relative))))


if __name__=='__main__':
    main()
