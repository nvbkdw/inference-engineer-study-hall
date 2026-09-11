"""DGX Spark experiment helpers. CPU execution is for untimed correctness/modeling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics


def cuda_environment(device='cuda:0'):
    import torch
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('DGX Spark measurement requires CUDA-enabled PyTorch and a GPU; CPU timing fallback is disabled. See shared/SETUP.md.')
    torch.cuda.set_device(selected)
    properties = torch.cuda.get_device_properties(selected)
    probe = torch.ones(8, device=selected)
    assert (probe + 1).sum().item() == 16
    return dict(device=str(selected), gpu_name=properties.name,
                compute_capability=list(torch.cuda.get_device_capability(selected)),
                cuda_runtime=torch.version.cuda, torch=torch.__version__,
                visible_device_count=torch.cuda.device_count(),
                device_total_memory_bytes=properties.total_memory)


def begin(chapter, description, assumptions, *, measurement=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--out', type=Path, required=True, help='New/empty output directory')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    if measurement:
        parser.add_argument('--device', default='cuda:0', help='CUDA device; no CPU timing fallback')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('use at least three repeats')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('output directory must be empty; preserve previous results')
    hardware = {}
    if measurement:
        try:
            hardware = cuda_environment(args.device)
        except (ImportError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = [*root.glob('shared/*.py'), *root.glob(f'chapters/{chapter}_*/code/*.py')]
    manifest = dict(chapter=chapter, description=description, assumptions=assumptions,
                    seed=args.seed, repeats=args.repeats, python=platform.python_version(),
                    platform=platform.platform(), hardware=hardware,
                    result_kind='measured CUDA workload' if measurement else 'untimed simulation / numerical model',
                    timing='CUDA events on current stream; warmed workload; dispatch gaps may be included' if measurement else None,
                    warmup_calls=10 if measurement else None,
                    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        if measurement:
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        manifest['torch'] = torch.__version__
        manifest['fp32_tf32_allowed'] = False if measurement else None
    except ImportError:
        pass
    write_json(args.out/'manifest.json', manifest)
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('no observations')
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def time_cuda(fn, repeats, inner=5, warmup=10):
    """Current-stream device interval, not CPU dispatch time or serving latency.

    fn must run on the selected CUDA device/current stream. Python multi-kernel
    references can include dispatch gaps. Keep fixture setup, host copies, .item(),
    correctness checks, and plotting outside timing.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA timing cannot run on CPU')
    if min(repeats, inner, warmup) <= 0:
        raise ValueError('positive timing counts required')
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end)/inner)
    return samples


def plot(path, rows, x, y, group, title):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for label in dict.fromkeys(row[group] for row in rows):
        points = {}
        for row in rows:
            if row[group] == label:
                points.setdefault(row[x], []).append(row[y])
        xs = sorted(points)
        ys = [statistics.median(points[t]) for t in xs]
        lo = [statistics.median(points[t])-min(points[t]) for t in xs]
        hi = [max(points[t])-statistics.median(points[t]) for t in xs]
        ax.errorbar(xs, ys, yerr=[lo,hi], marker='o', capsize=3, label=str(label))
    ax.set(xlabel=x, ylabel=y, title=title)
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=140)
    plt.close(fig)


def finish(args, rows, summary):
    write_csv(args.out/'results.csv', rows)
    write_json(args.out/'summary.json', summary)
    print(f'Wrote {len(rows)} rows to {args.out}; inspect summary.json and figures.')
