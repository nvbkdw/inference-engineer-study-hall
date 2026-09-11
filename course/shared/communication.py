"""Two-rank communication experiment shared by standalone Projects 7 and 8."""
import argparse
from datetime import timedelta
import hashlib
import os
import socket
from pathlib import Path
import platform
import statistics
import time
import torch
import torch.distributed as dist
from experiment import finish,plot,write_csv,write_json,cuda_environment


def affine_fit(points):
    """Nonnegative least-squares fit y = intercept + slope*x (tiny two-parameter fit)."""
    xbar=statistics.mean(x for x,y in points)
    ybar=statistics.mean(y for x,y in points)
    denom=sum((x-xbar)**2 for x,y in points)
    if denom<=0:
        raise ValueError('need distinct payload sizes')
    slope=sum((x-xbar)*(y-ybar) for x,y in points)/denom
    intercept=ybar-slope*xbar
    candidates=[(max(ybar,0),0),(0,max(sum(x*y for x,y in points)/sum(x*x for x,y in points),0))]
    if slope>=0 and intercept>=0:
        candidates.append((intercept,slope))
    return min(candidates,key=lambda ab:sum((y-ab[0]-ab[1]*x)**2 for x,y in points))


def run(chapter,mode):
    parser=argparse.ArgumentParser(description=f'Two-rank {mode} calibration and withheld prediction')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--backend',choices=['nccl'],default='nccl')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<3:
        parser.error('at least three repeats required')
    torch.set_num_threads(1)
    local_rank=int(os.environ.get('LOCAL_RANK',0))
    if torch.cuda.is_available() and int(os.environ.get('LOCAL_WORLD_SIZE',1))>torch.cuda.device_count():
        parser.error('One DGX Spark has one GPU. Use two connected Sparks with one rank per host, or two physical GPUs; no shared-device timing.')
    try:
        hardware=cuda_environment(f'cuda:{local_rank}')
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if int(os.environ.get('LOCAL_WORLD_SIZE',1))>torch.cuda.device_count():
        parser.error('One DGX Spark has one GPU. Use two connected Sparks with one rank per host, or two physical GPUs; no shared-device timing.')
    device=torch.device(f'cuda:{local_rank}')
    dist.init_process_group(args.backend,timeout=timedelta(seconds=90))
    try:
        if dist.get_world_size()!=2:
            raise ValueError('launch exactly two ranks')
        rank=dist.get_rank()
        hardware.update(rank=rank,host=socket.gethostname(),local_rank=local_rank)
        rank_hardware=[None]*2
        dist.all_gather_object(rank_hardware,hardware)
        if len({(item['host'],item['device']) for item in rank_hardware})!=2:
            raise ValueError('each rank must own a distinct physical GPU')
        status=torch.tensor([1],device=device)
        if rank==0 and args.out.exists() and any(args.out.iterdir()):
            status.zero_()
        dist.broadcast(status,src=0)
        if not status.item():
            raise ValueError('output directory must be empty')
        calibrate=[4096,65536,1048576]
        withheld=[16384,262144,4194304]
        if rank==0:
            args.out.mkdir(parents=True,exist_ok=True)
            root=Path(__file__).resolve().parents[1]
            files=[Path(__file__),root/'shared/experiment.py',*root.glob(f'chapters/{chapter}_*/code/experiment.py')]
            write_json(args.out/'manifest.json',dict(chapter=chapter,backend=args.backend,world_size=2,
                mode=mode,rank_hardware=rank_hardware,python=platform.python_version(),torch=torch.__version__,platform=platform.platform(),
                calibration_bytes=calibrate,withheld_bytes=withheld,repeats=args.repeats,
                timing='max rank wall time' if mode=='all_reduce' else 'sender wall time through completion acknowledgment',
                source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                limits='Preallocated FP32 contiguous tensors; no model, serving queue, allocation, layout conversion, or host staging'))
        def synchronize():
            if device.type=='cuda':
                torch.cuda.synchronize()
        def measure(byte_count):
            payload=torch.empty(byte_count//4,device=device)
            ack=torch.ones(1,dtype=torch.int64,device=device)
            samples=[]
            for repeat in range(args.repeats+3):
                payload.fill_(rank+1)
                synchronize()
                dist.barrier()
                synchronize()
                start=time.perf_counter()
                if mode=='all_reduce':
                    dist.all_reduce(payload)
                    synchronize()
                elif rank==0:
                    dist.send(payload,1)
                    dist.recv(ack,1)
                    synchronize()
                else:
                    dist.recv(payload,0)
                    synchronize()  # acknowledge only after payload is ready
                    dist.send(ack,0)
                    synchronize()
                elapsed=1000*(time.perf_counter()-start)
                expected=3 if mode=='all_reduce' else 1
                if not bool((payload==expected).all()):
                    raise AssertionError('payload corruption')
                timing=torch.tensor([elapsed if mode=='all_reduce' or rank==0 else 0.],device=device,dtype=torch.float64)
                dist.all_reduce(timing,op=dist.ReduceOp.MAX)  # Outside timed region.
                if repeat>=3:
                    samples.append(timing.item())
            return samples
        calibration=[]
        for size in calibrate:
            values=measure(size)
            calibration.extend(dict(payload_bytes=size,repeat=j,measured_ms=t) for j,t in enumerate(values))
        fit=affine_fit([(size,statistics.median(row['measured_ms'] for row in calibration if row['payload_bytes']==size)) for size in calibrate])
        predictions={size:fit[0]+fit[1]*size for size in withheld}
        if rank==0:
            write_csv(args.out/'calibration.csv',calibration)
            write_json(args.out/'prediction.json',dict(intercept_ms=fit[0],slope_ms_per_byte=fit[1],
                predicted_ms=predictions,hypothesis='Affine local payload model generalizes to withheld sizes; protocols may change with size.'))
        dist.barrier()  # All held-out work begins after rank 0 persists the predictions.
        rows,errors=[],[]
        for size in withheld:
            samples=measure(size)
            rows.append(dict(payload_bytes=size,kind='predicted',repeat=-1,latency_ms=predictions[size]))
            rows.extend(dict(payload_bytes=size,kind=f'measured {args.backend}',repeat=j,latency_ms=t) for j,t in enumerate(samples))
            errors.append(abs(predictions[size]-statistics.median(samples))/statistics.median(samples))
        if rank==0:
            plot(args.out/'transfer.svg',rows,'payload_bytes','latency_ms','kind',f'P{chapter}: {mode}, two {args.backend} ranks')
            finish(args,rows,dict(correctness='all payloads matched',withheld_sizes=3,
                median_relative_error=statistics.median(errors),relative_errors=errors,
                scope='Communication primitive only; not full model scaling or P/D serving'))
    finally:
        dist.destroy_process_group()
