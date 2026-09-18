"""Measure random-layer error/storage and CUDA reconstruction cost at several group sizes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from experiment import begin, finish, plot, time_cuda, write_csv, write_json
from lab import quantize,dequantize,pack,unpack
import statistics
import torch


def main():
    args=begin('06',__doc__,['Random FP32 layer N=1024 K=4096; no AWQ calibration or model quality claim',
        'Symmetric INT4 teaching layout with FP32 scales; KV not present',
        'Timing includes full dequantization to FP32 plus ordinary CUDA GEMM; not packed inference'])
    write_json(args.out/'prediction.json',dict(hypothesis='Smaller groups reduce metadata-adjusted compression; activation direction changes output error.',
                                              groups=[16,32,64,128,256],weight_shape=[1024,4096]))
    rows,timings=[],[]
    with torch.inference_mode():
        for repeat in range(args.repeats):
            w=torch.randn(1024,4096,device=args.device)
            w[:,0]*=20
            x=torch.randn(16,4096,device=args.device)
            for activation in ['isotropic','sensitive_channel']:
                current=x.clone()
                if activation=='sensitive_channel':
                    current[:,1]*=20
                reference=current@w.T
                for group in [16,32,64,128,256]:
                    q,scale=quantize(w,group)
                    packed=pack(q)
                    assert torch.equal(unpack(packed,q.shape),q)
                    reconstructed=dequantize(q,scale,group)
                    output=current@reconstructed.T
                    relative=float((output-reference).norm()/reference.norm())
                    rows.append(dict(repeat=repeat,activation=activation,group_size=group,
                                     relative_output_l2=relative,packed_bytes=packed.numel(),
                                     scale_bytes=scale.numel()*scale.element_size(),
                                     original_fp32_bytes=w.numel()*w.element_size()))
                    baseline=time_cuda(lambda:current@w.T,3,5)
                    rebuilt=time_cuda(lambda:current@dequantize(q,scale,group).T,3,5)
                    timings.extend([dict(repeat=repeat,activation=activation,group_size=group,
                                         method=method,measured_ms=statistics.median(values))
                                    for method,values in [('FP32 GEMM',baseline),('INT4 reconstruction + FP32 GEMM',rebuilt)]])
    write_csv(args.out/'timing.csv',timings)
    plot(args.out/'error.svg',rows,'group_size','relative_output_l2','activation','P6 random-layer error: median and fixture range')
    finish(args,rows,dict(packing='passed',random_weight_fixtures=args.repeats,
                         activation_conditions=2,group_sizes=5,quality_claim='none',
                         scope='numerical and CUDA reconstruction exercise; not packed GPU execution'))


if __name__ == '__main__':
    main()
