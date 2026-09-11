"""Two-rank column/row-sharded SwiGLU demo. CPU simulation or real torchrun."""
import argparse
import os
import torch
import torch.distributed as dist
from torch.nn import functional as F


def fixture(device='cpu'):
    # Generate on CPU for identical fixtures across CPU and GPU ranks.
    generator = torch.Generator().manual_seed(42)
    return [torch.randn(shape, generator=generator).to(device) for shape in
            [(3,16), (32,16), (32,16), (16,32)]]


def local_mlp(x, gate, up, down, rank, size):
    g = gate.chunk(size, dim=0)[rank]
    u = up.chunk(size, dim=0)[rank]
    d = down.chunk(size, dim=1)[rank]
    return (F.silu(x @ g.T)*(x @ u.T)) @ d.T


def check():
    x,g,u,d = fixture()
    ref = (F.silu(x@g.T)*(x@u.T))@d.T
    actual = sum(local_mlp(x,g,u,d,rank,2) for rank in range(2))
    torch.testing.assert_close(actual,ref,rtol=1e-4,atol=1e-4)
    print('PASS: simulated TP=2 column/row sharding equals unsharded SwiGLU')


def distributed(backend):
    local_rank = int(os.environ['LOCAL_RANK'])
    if backend == 'nccl':
        torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}' if backend == 'nccl' else 'cpu'
    dist.init_process_group(backend)
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size != 2:
            raise ValueError('this exercise requires exactly two ranks')
        x,g,u,d = fixture(device)
        partial = local_mlp(x,g,u,d,rank,size)
        dist.all_reduce(partial)
        ref = (F.silu(x@g.T)*(x@u.T))@d.T
        torch.testing.assert_close(partial,ref,rtol=1e-4,atol=1e-4)
        if rank == 0:
            print(f'PASS: real {backend} TP=2 SwiGLU all-reduce')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['gloo','nccl'])
    a = p.parse_args()
    distributed(a.backend) if a.backend else check()
