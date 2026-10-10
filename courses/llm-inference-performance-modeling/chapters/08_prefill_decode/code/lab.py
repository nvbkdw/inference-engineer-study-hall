"""KV handoff oracle and synchronous two-rank payload demo; no serving scheduler."""
import argparse
from dataclasses import dataclass
import os
import torch
import torch.distributed as dist


@dataclass
class State:
    model_revision: str
    layout: str
    processed: int
    history: list
    pending: int
    k: torch.Tensor
    v: torch.Tensor

    def validate(self, expected_revision):
        if self.model_revision != expected_revision or self.layout != 'L,B,H,S,R':
            raise ValueError('incompatible identity/layout')
        if self.k.shape != self.v.shape or self.k.ndim != 5 or self.k.shape[3] != self.processed:
            raise ValueError('cache shape/length mismatch')
        if len(self.history) != self.processed+1 or self.history[-1] != self.pending:
            raise ValueError('history must contain processed prefix plus one pending token')


def fixture():
    torch.manual_seed(42)
    return State('tiny-fixture-v1', 'L,B,H,S,R', 5, [1,2,3,4,5,6], 6,
                 torch.randn(2,1,2,5,8), torch.randn(2,1,2,5,8))


def continue_attention(state):
    """Deterministic attention continuation oracle; not a full Qwen forward pass."""
    generator = torch.Generator().manual_seed(state.pending)
    shape = (*state.k.shape[:3],1,state.k.shape[-1])
    q,k_new,v_new = [torch.randn(shape, generator=generator).to(state.k.device) for _ in range(3)]
    k, v = torch.cat((state.k,k_new),3), torch.cat((state.v,v_new),3)
    scores = q @ k.transpose(-1,-2)/k.shape[-1]**.5
    return scores.softmax(-1) @ v


def transfer_ms(byte_count, gbps, setup_ms=0):
    if byte_count < 0 or gbps <= 0 or setup_ms < 0:
        raise ValueError('invalid transfer parameters')
    return setup_ms + 1000*byte_count/(gbps*1e9)


def check():
    source = fixture()
    source.validate('tiny-fixture-v1')
    dest = State(source.model_revision,source.layout,source.processed,source.history.copy(),
                 source.pending,source.k.clone(),source.v.clone())
    dest.validate('tiny-fixture-v1')
    assert dest.k.data_ptr() != source.k.data_ptr()
    torch.testing.assert_close(continue_attention(source),continue_attention(dest))
    # Source release/mutation after destination installation cannot alter the copy.
    expected = dest.k.clone()
    source.k.zero_()
    assert torch.equal(dest.k,expected)
    print('PASS: state validation, independent destination storage, continuation')
    print(f'ILLUSTRATIVE 2 GiB at 200 Gbit/s: {transfer_ms(2*2**30,25):.3f} ms')


def distributed(backend):
    local_rank = int(os.environ['LOCAL_RANK'])
    if backend == 'nccl':
        torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}' if backend == 'nccl' else 'cpu'
    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        source = fixture()
        # Fixed fixture protocol: identity/layout/shape are pre-agreed by both ranks.
        # Real requests must exchange and validate a versioned header before allocation.
        payload_shape = (2,*source.k.shape)
        if rank == 0:
            payload = torch.stack((source.k,source.v)).to(device)
            dist.send(payload,1)
            ack = torch.zeros(1,dtype=torch.int64,device=device)
            dist.recv(ack,1)
            assert ack.item() == 1
            print('PASS: destination acknowledged installed KV; source can release')
        else:
            payload = torch.empty(payload_shape,device=device)
            dist.recv(payload,0)
            dest = State(source.model_revision,source.layout,source.processed,source.history,
                         source.pending,payload[0],payload[1])
            dest.validate('tiny-fixture-v1')
            torch.testing.assert_close(continue_attention(dest).cpu(),continue_attention(source),rtol=1e-4,atol=1e-5)
            dist.send(torch.ones(1,dtype=torch.int64,device=device),0)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['gloo','nccl'])
    a = p.parse_args()
    distributed(a.backend) if a.backend else check()
