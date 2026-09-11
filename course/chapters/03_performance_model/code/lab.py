"""Illustrative component roofline. All hardware inputs must be supplied explicitly."""
import argparse
import csv
import sys


def roofline(flops, byte_count, tflops, gbps, launch_us=0):
    if min(flops, byte_count, launch_us) < 0 or min(tflops, gbps) <= 0:
        raise ValueError('nonnegative work and positive rates required')
    compute = flops/(tflops*1e12)
    memory = byte_count/(gbps*1e9)
    return 1000*(max(compute, memory) + launch_us*1e-6)


def gemm(m, k, n, element_bytes=2):
    return 2*m*k*n, element_bytes*(m*k+k*n+m*n)


def predict(model, batch, context, phase, tflops, gbps, launch_us):
    """Equal-length requests, separate projections, final-position LM head in prefill.

    Attention bytes are a unique-KV idealization, not measured physical traffic.
    Norm/RoPE/activation/scheduler costs are omitted and must be added after profiling.
    """
    if model == '8b':
        layers, d, i, h, g, r, vocab = 36, 4096, 12288, 32, 8, 128, 151936
    else:
        layers, d, i, h, g, r, vocab = 64, 5120, 25600, 64, 8, 128, 151936
    m = batch * (context if phase == 'prefill' else 1)
    rows = []
    for name, k, n, count in [('q', d, h*r, layers), ('k', d, g*r, layers),
                              ('v', d, g*r, layers), ('o', h*r, d, layers),
                              ('gate', d, i, layers), ('up', d, i, layers), ('down', i, d, layers),
                              ('lm_head', d, vocab, 1)]:
        rows_m = batch if name == 'lm_head' else m
        f, q = gemm(rows_m, k, n)
        rows.append((name, count, f*count, q*count,
                     count*roofline(f, q, tflops, gbps, launch_us)))
    pairs = batch * (context*(context+1)//2 if phase == 'prefill' else context)
    f = 4*h*r*pairs
    # Q read + output write, K/V reads once. Dense score materialization excluded.
    q = 2*(2*m*h*r + 2*batch*context*g*r)
    rows.append(('attention', layers, f*layers, q*layers,
                 layers*roofline(f, q, tflops, gbps, launch_us)))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['8b', '32b'], default='8b')
    p.add_argument('--phase', choices=['prefill', 'decode'], default='decode')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--tflops', type=float, required=True)
    p.add_argument('--gbps', type=float, required=True)
    p.add_argument('--launch-us', type=float, default=0)
    a = p.parse_args()
    if min(a.batch, a.context) <= 0:
        p.error('batch/context must be positive')
    rows = predict(a.model, a.batch, a.context, a.phase, a.tflops, a.gbps, a.launch_us)
    w = csv.writer(sys.stdout)
    w.writerow(['op', 'count', 'flops', 'logical_bytes', 'modeled_ms'])
    w.writerows(rows)
    print(f'MODELED total_ms={sum(row[-1] for row in rows):.6f}; omitted costs documented in source', file=sys.stderr)
