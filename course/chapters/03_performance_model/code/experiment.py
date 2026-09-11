"""Calibrate a DGX Spark BF16 roofline and measure six withheld GEMM shapes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from experiment import begin, finish, plot, time_cuda, write_csv, write_json
from lab import gemm,roofline
import statistics
import torch


def main():
    args = begin('03',__doc__,['CUDA BF16 Qwen3-8B Q-projection shape K=N=4096',
        'Copy calibration: 128 MiB buffer, effective rate; verify cache residency before claiming DRAM bandwidth',
        'Calibration M=1,16,512; held-out M=2,4,8,32,64,128',
        'Median relative error <= 25% is diagnostic, not a guaranteed result'])
    calibration=[]
    device=args.device
    with torch.inference_mode():
        one=torch.ones(1,1,device=device,dtype=torch.bfloat16)
        small_out=torch.empty_like(one)
        # A small-call floor, not an isolated hardware launch-latency measurement.
        overhead=min(time_cuda(lambda:torch.mm(one,one,out=small_out),args.repeats,50))
        source=torch.randn(32*1024*1024,device=device)
        dest=torch.empty_like(source)
        copy_samples=time_cuda(lambda:dest.copy_(source),args.repeats)
        copy_ms=statistics.median(copy_samples)
        gbps=2*source.numel()*source.element_size()/(copy_ms*1e6)
        write_csv(args.out/'bandwidth.csv',[dict(repeat=j,logical_bytes=2*source.numel()*source.element_size(),measured_ms=t) for j,t in enumerate(copy_samples)])
        rates=[]
        k=n=4096
        for m in [1,16,512]:
            x=torch.randn(m,k,device=device,dtype=torch.bfloat16)
            w=torch.randn(k,n,device=device,dtype=torch.bfloat16)
            out=torch.empty(m,n,device=device,dtype=torch.bfloat16)
            samples=time_cuda(lambda:torch.mm(x,w,out=out),args.repeats,10)
            elapsed=statistics.median(samples)
            f,q=gemm(m,k,n,2)
            rates.append(f/(max(elapsed-overhead,elapsed*.1)*1e9))
            calibration.extend(dict(m=m,repeat=j,measured_ms=t) for j,t in enumerate(samples))
        tflops=max(rates)
        heldout=[2,4,8,32,64,128]
        predicted={m:roofline(*gemm(m,k,n,2),tflops,gbps,overhead*1000) for m in heldout}
        write_csv(args.out/'calibration.csv',calibration)
        write_json(args.out/'prediction.json',dict(tflops=tflops,gbps=gbps,small_call_floor_ms=overhead,
                                                  k=k,n=n,dtype='bf16',predicted_ms=predicted,heldout=heldout))
        rows,errors=[],[]
        for m in heldout:
            x=torch.randn(m,k,device=device,dtype=torch.bfloat16)
            w=torch.randn(k,n,device=device,dtype=torch.bfloat16)
            out=torch.empty(m,n,device=device,dtype=torch.bfloat16)
            samples=time_cuda(lambda:torch.mm(x,w,out=out),args.repeats,10)
            rows.append(dict(m=m,kind='predicted',repeat=-1,latency_ms=predicted[m]))
            rows.extend(dict(m=m,kind='measured CUDA BF16',repeat=j,latency_ms=t) for j,t in enumerate(samples))
            errors.append(abs(predicted[m]-statistics.median(samples))/statistics.median(samples))
    plot(args.out/'prediction.svg',rows,'m','latency_ms','kind','P3 CUDA BF16 roofline vs held-out GEMMs')
    finish(args,rows,dict(heldout_points=6,relative_errors=errors,
                         median_relative_error=statistics.median(errors),
                         diagnostic_target_met=statistics.median(errors)<=.25,
                         scope='CUDA Q-projection GEMMs; actual GPU recorded in manifest, not full-model latency'))


if __name__ == '__main__':
    main()
