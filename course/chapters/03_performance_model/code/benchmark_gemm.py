"""CUDA-event GEMM baseline; preallocated output, separate warmup, raw repeat times."""
import argparse
import csv
import sys
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, default=4)
    p.add_argument('--k', type=int, default=4096)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--inner', type=int, default=10)
    a = p.parse_args()
    if min(a.m, a.k, a.n, a.repeats, a.inner) <= 0:
        p.error('all sizes/counts must be positive')
    if not torch.cuda.is_available():
        p.error('CUDA required; use lab.py for CPU analytical exercises')
    x = torch.randn(a.m, a.k, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(a.k, a.n, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(a.m, a.n, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        for _ in range(20):
            torch.mm(x, w, out=out)
        torch.cuda.synchronize()
        writer = csv.writer(sys.stdout)
        writer.writerow(['repeat', 'm', 'k', 'n', 'measured_ms'])
        for repeat in range(a.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.inner):
                torch.mm(x, w, out=out)
            end.record()
            end.synchronize()
            writer.writerow([repeat, a.m, a.k, a.n, start.elapsed_time(end)/a.inner])


if __name__ == '__main__':
    main()
