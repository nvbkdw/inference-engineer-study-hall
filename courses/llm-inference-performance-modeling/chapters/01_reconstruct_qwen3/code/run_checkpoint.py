"""Save selected full-vocabulary logits from a local Qwen3 snapshot, one model at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from functools import partial
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from experiment import write_json,write_csv
from checkpoint import configuration,load_custom
from notebook_utils import load_notebook_implementation
from safetensors.torch import save_file
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--tokens',type=Path,required=True)
    p.add_argument('--model-revision',required=True,help='Resolved checkpoint commit; demo may use a clearly named local fixture ID')
    p.add_argument('--backend',choices=['custom','transformers'],required=True)
    p.add_argument('--dtype',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    implementation=load_notebook_implementation()
    load_candidate=partial(load_custom,model_factory=implementation.TinyQwen3,
                           name_map_factory=implementation.model_weight_name_mapping)
    if a.out.exists() and any(a.out.iterdir()):
        p.error('output directory must be empty')
    c,data=configuration(a.snapshot)
    fixture=json.loads(a.tokens.read_text())
    prompt,continuation=fixture['prompt_ids'],fixture['continuation_ids']
    if not prompt or any(type(i)!=int or i<0 or i>=c.vocab for i in prompt+continuation):
        p.error('nonempty prompt and valid integer token IDs required')
    if len(prompt)+len(continuation)>data.get('max_position_embeddings',32768):
        p.error('fixture exceeds declared context capacity')
    a.out.mkdir(parents=True,exist_ok=True)
    dtype=torch.float32 if a.dtype=='fp32' else torch.bfloat16
    device=torch.device(a.device)
    if device.type=='cpu':
        torch.set_num_threads(1)
    elif device.type=='cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize(device)
    manifest=dict(backend=a.backend,snapshot=str(a.snapshot.resolve()),model_revision=a.model_revision,
        config_sha256=hashlib.sha256((a.snapshot/'config.json').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(a.tokens.read_bytes()).hexdigest(),fixture_kind=fixture.get('kind','unspecified'),
        dtype=a.dtype,device=str(device),torch=torch.__version__,prompt_tokens=len(prompt),
        continuation_tokens=len(continuation),sampling='teacher-forced supplied continuation',
        attention='eager dense reference',timing='cold, synchronized diagnostic; not a steady-state benchmark',
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ['lab.ipynb','notebook_utils.py','checkpoint.py','run_checkpoint.py']})
    (a.out/'implementation_notebook.ipynb').write_bytes((Path(__file__).parent/'lab.ipynb').read_bytes())
    write_json(a.out/'manifest.json',manifest)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    # Audit headers without allocating full weights, including for the HF reference path.
    _,inventory=load_candidate(a.snapshot,audit_only=True)
    write_json(a.out/'inventory.json',inventory)
    write_json(a.out/'memory_prediction.json',dict(parameters=c.parameters(),
        parameter_bytes=c.parameters()*(4 if a.dtype=='fp32' else 2),
        final_kv_bytes=c.kv_bytes(len(prompt)+len(continuation),4 if a.dtype=='fp32' else 2)))
    if a.audit_only:
        print(f'Audited {len(inventory)} tensors and {c.parameters()} parameters without allocating model storage.')
        return
    if a.backend=='custom':
        model,_=load_candidate(a.snapshot,device,dtype)
    else:
        import transformers
        from transformers import AutoModelForCausalLM
        manifest['transformers']=transformers.__version__
        write_json(a.out/'manifest.json',manifest)
        model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,dtype=dtype,
                                                 device_map={'':str(device)},attn_implementation='eager').eval()
    sync()
    loading_ms=1000*(time.perf_counter()-start)
    rows,saved,cache=[],[],None
    with torch.inference_mode():
        for step,ids in enumerate([prompt]+[[t] for t in continuation]):
            ids_tensor=torch.tensor([ids],device=device)
            sync()
            start=time.perf_counter()
            if a.backend=='custom':
                logits,cache=model(ids_tensor,cache,decode=True)
            else:
                output=model(input_ids=ids_tensor,past_key_values=cache,use_cache=True,logits_to_keep=1)
                logits,cache=output.logits,output.past_key_values
            sync()
            elapsed=1000*(time.perf_counter()-start)
            saved.append(logits[0,-1].float().cpu())
            rows.append(dict(step=step,phase='prefill' if step==0 else 'forced_decode',
                             processed_tokens=len(prompt)+step,diagnostic_ms=elapsed))
    save_file({'logits':torch.stack(saved)},str(a.out/'logits.safetensors'))
    write_csv(a.out/'diagnostic_timing.csv',rows)
    summary=dict(saved_positions=len(saved),vocabulary=c.vocab,loading_ms=loading_ms,
                 parameters=sum(p.numel() for p in model.parameters()),
                 scope='Selected-position numerical reference; cold timings are diagnostic only')
    if device.type=='cuda':
        summary.update(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    write_json(a.out/'summary.json',summary)
    print(f'Saved {len(saved)} complete vocabulary vectors to {a.out}')


if __name__=='__main__':
    main()
