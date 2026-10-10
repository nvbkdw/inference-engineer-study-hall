"""Focused SwiGLU work-partition experiment: kernel and adapter timing boundaries."""

import importlib
from pathlib import Path
import torch
from shared import performance as p


def run(out):
    k = importlib.import_module("chapters.03_kernels.code.cute_kernels")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    p.require_cuda("cuda:0", "bfloat16")
    p.write_json(
        out / "prediction.json",
        dict(
            scope="synthetic BF16 component inputs at course MLP widths; not full-model latency",
            mechanism="256 versus 128 threads: fewer blocks, same arithmetic/logical bytes; direction is shape dependent",
            logical_bytes_unfused_per_element=10,
            logical_bytes_fused_per_element=6,
            warmup=2,
            repeats=3,
            tolerances=dict(atol=0.02, rtol=0.02),
        ),
    )
    rows = []
    with torch.inference_mode():
        for width in (12288, 25600):
            for tokens in (1, 128):
                torch.manual_seed(42)
                g = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
                u = torch.randn_like(g)
                expected = torch.nn.functional.silu(g) * u
                for block in (128, 256):
                    torch.testing.assert_close(
                        k.swiglu(g, u, block), expected, rtol=0.02, atol=0.02
                    )
                    y = torch.empty_like(g)
                    tensors = (g.view(-1), u.view(-1), y.view(-1))
                    compiled, args, stream = k._launch(
                        "swiglu", k.swiglu_launch, tensors, (block,)
                    )
                    for boundary, fn in [
                        ("kernel", lambda: compiled(*args, stream)),
                        ("adapter", lambda: k.swiglu(g, u, block)),
                        ("reference", lambda: torch.nn.functional.silu(g) * u),
                    ]:
                        samples = p.cuda_samples(fn, repeats=3, inner=1, warmup=2)
                        rows.extend(
                            dict(
                                width=width,
                                tokens=tokens,
                                block=block,
                                boundary=boundary,
                                repeat=i,
                                cuda_ms=ms,
                            )
                            for i, ms in enumerate(samples)
                        )
    p.write_csv(out / "results.csv", rows)
    p.write_json(out / "compilations.json", k.COMPILATIONS)
    return rows
