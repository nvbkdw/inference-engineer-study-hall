"""Instrumented captures, deliberately separate from ordinary benchmark records."""

from contextlib import contextmanager, ExitStack
import csv
import json
from pathlib import Path
import torch


@contextmanager
def annotations(model):
    """Enable Python operator names and NVTX only inside this context."""
    original = []
    handles = []
    stack = []

    def enter(name):
        scope = ExitStack()
        scope.enter_context(torch.profiler.record_function(name))
        scope.enter_context(torch.cuda.nvtx.range(name))
        return scope

    def pre(name):
        def hook(module, args):
            stack.append(enter(name))

        return hook

    def post(module, args, out):
        stack.pop().close()

    try:
        for name, module in model.named_modules():
            if not list(module.children()):
                handles.extend(
                    [
                        module.register_forward_pre_hook(pre(name)),
                        module.register_forward_hook(post, always_call=True),
                    ]
                )
        for layer in model.layers:
            for name in ("normalize", "normalize_rotary", "activation", "attention"):
                method = getattr(layer, name)
                had = name in layer.__dict__
                original.append((layer, name, method, had))

                def wrapper(*args, _method=method, _name=name, **kwargs):
                    with enter(_name):
                        return _method(*args, **kwargs)

                setattr(layer, name, wrapper)
        yield
    finally:
        for module, name, method, had in original:
            if had:
                setattr(module, name, method)
            else:
                delattr(module, name)
        for handle in handles:
            handle.remove()
        for scope in reversed(stack):
            scope.close()


def prepare_call(model, batch, context, phase, device="cuda:0"):
    generator = torch.Generator(device=device).manual_seed(42)
    ids = torch.randint(
        100, 10000, (batch, context), device=device, generator=generator
    )
    cache = None
    if phase == "decode":
        with torch.inference_mode():
            _, cache = model(ids, decode=True)
        ids = torch.full((batch, 1), 100, device=device, dtype=torch.long)

    def call():
        return model(ids, cache, decode=True)

    with torch.inference_mode():
        for _ in range(2):
            result = call()
            del result
    torch.cuda.synchronize(device)
    return call


def capture(model, batch, context, phase, profiler, out, device="cuda:0"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    call = prepare_call(model, batch, context, phase, device)
    with torch.inference_mode(), annotations(model):
        if profiler == "torch":
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with torch.profiler.record_function(f"{phase}_B{batch}_S{context}"):
                    result = call()
                    torch.cuda.synchronize(device)
                del result
            prof.export_chrome_trace(str(out / "trace.json"))
            events = prof.key_averages(group_by_input_shape=True)
            (out / "operators.txt").write_text(
                events.table(sort_by="self_cuda_time_total", row_limit=100)
            )
            rows = [
                dict(
                    operator=e.key,
                    shapes=str(e.input_shapes),
                    calls=e.count,
                    self_cpu_us=e.self_cpu_time_total,
                    self_cuda_us=e.self_device_time_total,
                    cuda_memory_bytes=e.device_memory_usage,
                )
                for e in events
            ]
            with (out / "operators.csv").open("w") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            trace = json.loads((out / "trace.json").read_text())
            kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
            names = sorted(set(e["name"] for e in kernels))
            (out / "kernel_summary.json").write_text(
                json.dumps(dict(count=len(kernels), names=names), indent=2)
            )
            if hasattr(model, "optimizations") and any(
                vars(model.optimizations)[f]
                for f in (
                    "rmsnorm",
                    "swiglu",
                    "qk_rope",
                    "prefill_attention",
                    "decode_attention",
                )
            ):
                if not any(
                    "kernel" in name
                    and any(
                        k in name for k in ("rms", "swiglu", "qk_rope", "attention")
                    )
                    for name in names
                ):
                    raise AssertionError("Trace contains no custom kernels")
        elif profiler in ("nsys", "ncu"):
            torch.cuda.cudart().cudaProfilerStart()
            try:
                result = call()
                torch.cuda.synchronize(device)
                del result
            finally:
                torch.cuda.cudart().cudaProfilerStop()
        else:
            raise ValueError(profiler)


def capture_grid(model, cases, out, device="cuda:0"):
    """One external-profiler process/model; every call warms before capture starts."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    calls = [
        (
            case,
            prepare_call(model, case["batch"], case["context"], case["phase"], device),
        )
        for case in cases
    ]
    torch.cuda.synchronize(device)
    with torch.inference_mode(), annotations(model):
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for case, call in calls:
                label = f"{case['phase']}_B{case['batch']}_S{case['context']}"
                with (
                    torch.cuda.nvtx.range(label),
                    torch.profiler.record_function(label),
                ):
                    result = call()
                    torch.cuda.synchronize(device)
                    del result
        finally:
            torch.cuda.cudart().cudaProfilerStop()
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
