"""Measure scaled CUDA cached/recomputed decode and audit logical cache storage."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'shared'))
from experiment import begin, finish, plot, time_cuda, write_csv, write_json
from lab import Config, TinyQwen
import torch


def main():
    args = begin('01', __doc__, ['Tiny random FP32 CUDA model, not Qwen benchmark',
                                'Logical tensor bytes exclude allocator/workspace',
                                'Each timed region produces exactly one next-position logit'])
    c = Config(layers=2, hidden=512, intermediate=1536, q_heads=8, kv_heads=2, head_dim=128, vocab=4096)
    model = TinyQwen(c).to(args.device).eval()
    lengths = [128,256,512,1024,2048]
    write_json(args.out/'prediction.json', {
        'cache_bytes_per_token':c.kv_bytes(1,4),
        'hypothesis':'Cache bytes are linear in processed length; recomputation grows faster than cached decode.',
        'lengths':lengths})
    memory, rows, largest_error = [], [], 0.
    with torch.inference_mode():
        for length in lengths:
            ids = torch.randint(c.vocab,(1,length+1),device=args.device)
            _,cache = model(ids[:,:length])
            actual = sum(t.numel()*t.element_size() for pair in cache for t in pair)
            expected = c.kv_bytes(length,4)
            assert actual == expected
            memory.extend([dict(length=length,kind=kind,cache_bytes=value) for kind,value in
                           [('predicted logical bytes',expected),('counted tensor bytes',actual)]])
            cached = lambda:model(ids[:,length:],cache,last_only=True)
            full = lambda:model(ids,last_only=True)
            a,b = cached()[0],full()[0][:,-1:]
            largest_error = max(largest_error,float((a-b).abs().max()))
            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
            for name,fn in [('cached',cached),('recomputed',full)]:
                for repeat,ms in enumerate(time_cuda(fn,args.repeats)):
                    rows.append(dict(length=length,method=name,repeat=repeat,measured_ms=ms))
    write_csv(args.out/'memory.csv',memory)
    plot(args.out/'memory.svg',memory,'length','cache_bytes','kind','P1 logical KV accounting (CUDA FP32)')
    plot(args.out/'latency.svg',rows,'length','measured_ms','method','P1 measured scaled CUDA decode: median and range')
    finish(args,rows,dict(correctness='passed',max_abs_logit_error=largest_error,
                         exact_cache_byte_matches=len(lengths),scope='scaled CUDA model only'))


if __name__ == '__main__':
    main()
