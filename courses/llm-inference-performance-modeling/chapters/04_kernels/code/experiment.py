"""Check and time dense versus online attention as a CUDA algorithm experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from experiment import begin, finish, plot, time_cuda, write_csv, write_json
from lab import dense_attention,online_attention
import torch


def main():
    args=begin('04',__doc__,['CUDA FP32 prefill, B=1 Hq=8 Hkv=2 R=128',
        'Ordinary PyTorch loops; not a fused GPU kernel or an integrated Qwen result',
        'Score temporary counts exclude all other tensors and allocator state'])
    write_json(args.out/'prediction.json',dict(hypothesis='Online attention reduces score temporary size; Python tiling may increase latency.',
                                              lengths=[127,128,129,512],tiles=[32,128]))
    rows,storage=[],[]
    error=0.
    with torch.inference_mode():
        for s in [127,128,129,512]:
            q,k,v=torch.randn(1,8,s,128,device=args.device),torch.randn(1,2,s,128,device=args.device),torch.randn(1,2,s,128,device=args.device)
            ref=dense_attention(q,k,v,0)
            for method,tile in [('dense',s),('online32',32),('online128',128)]:
                fn=(lambda:dense_attention(q,k,v,0)) if method=='dense' else (lambda:online_attention(q,k,v,0,tile))
                actual=fn()
                error=max(error,float((actual-ref).abs().max()))
                torch.testing.assert_close(actual,ref,rtol=1e-4,atol=1e-5)
                storage.append(dict(length=s,method=method,score_temporary_bytes=4*8*s*min(s,tile)))
                rows.extend(dict(length=s,method=method,repeat=j,measured_ms=ms)
                            for j,ms in enumerate(time_cuda(fn,args.repeats,2)))
    write_csv(args.out/'score_storage.csv',storage)
    plot(args.out/'latency.svg',rows,'length','measured_ms','method','P4 CUDA attention latency: median and range')
    plot(args.out/'score_storage.svg',storage,'length','score_temporary_bytes','method','P4 modeled score temporary size')
    finish(args,rows,dict(correctness='passed',max_abs_error=error,matched_cases=12,
                         scope='CUDA algorithm exercise; CuTe and integration remain GPU milestones'))


if __name__ == '__main__':
    main()
