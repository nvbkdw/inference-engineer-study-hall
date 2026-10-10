"""Reproducible real-checkpoint profiling and matched Chapter 2 sweeps.

Run --help; notebooks call the same functions. No CPU timing fallback.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from shared import performance as p

opt = importlib.import_module("chapters.03_kernels.code.model_opt")
profiling = importlib.import_module("chapters.03_kernels.code.profiling")
DEVICE = "cuda:0"
CASES = [(b, s) for b in (1, 2, 4) for s in (128, 512, 2048)]
TOLERANCES = dict(
    operator_atol=0.02, operator_rtol=0.02, logits_atol=0.25, logits_rtol=0.02
)


def hardware():
    return p.Hardware(
        "DGX Spark / GB10",
        273,
        {"bfloat16": 125.0, "float16": 125.0},
        "assumed",
        "Teaching assumption: 125 dense BF16 TFLOP/s, not a vendor BF16 specification",
        "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
    )


def source_hashes():
    paths = [
        ROOT / "shared/performance.py",
        ROOT / "pyproject.toml",
        *Path(__file__).parent.glob("*.py"),
    ]
    return {
        **opt.reference.source_hashes(),
        **{
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }


def save_compilation_metadata(out):
    module = sys.modules.get("chapters.03_kernels.code.cute_kernels")
    p.write_json(Path(out) / "compilations.json", module.COMPILATIONS if module else [])


def manifest(stage):
    p.require_cuda(DEVICE, "bfloat16")
    return dict(
        stage=stage,
        utc=datetime.now(timezone.utc).isoformat(),
        hardware=asdict(hardware()),
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        python=platform.python_version(),
        platform=platform.platform(),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        versions={
            name: version(name)
            for name in (
                "nvidia-cutlass-dsl",
                "cuda-python",
                "cuda-bindings",
                "transformers",
            )
        },
        revisions=p.REVISIONS,
        source_hashes=source_hashes(),
        tolerances=TOLERANCES,
        config_hashes={
            k: hashlib.sha256(
                (ROOT / "models" / f"qwen3-{k}" / rev / "config.json").read_bytes()
            ).hexdigest()
            for k, rev in p.REVISIONS.items()
        },
        compiler_target="sm_121 (auto-detected by pinned DSL)",
        precision="bfloat16",
        warmup_runs=2,
        repeats=3,
        decode_steps=8,
        timing="CUDA events: forward only; synchronized wall: forward + greedy token selection. Profilers excluded.",
        cache="caller owned, empty before prefill, append allocates; no prefix reuse",
        tokens="seed 42 CUDA randint prompt, teacher-forced saved continuation IDs; greedy selection still timed",
        work_model="causal matmul FLOPs / compulsory-byte proxy; not counter-measured traffic",
    )


def new_run(out, stage, cases):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    p.write_json(out / "manifest.json", manifest(stage))
    predictions = []
    for key in p.MODELS:
        for batch, prompt in cases:
            for step in range(-1, 8):
                work = p.model_work(
                    p.MODELS[key],
                    batch,
                    prompt if step == -1 else 1,
                    0 if step == -1 else prompt + step,
                )
                predictions.append(
                    dict(
                        model=key,
                        batch=batch,
                        prompt=prompt,
                        phase="prefill" if step == -1 else "decode",
                        step=max(step, 0),
                        predicted_ms=p.roofline_ms(
                            work["flops"], work["bytes_proxy"], 125, 273
                        ),
                        **work,
                    )
                )
    p.write_json(out / "predictions.json", predictions)
    tradeoffs = [
        dict(model=key, **r)
        for key, c in p.MODELS.items()
        for r in p.predict_tradeoffs(
            c, [1, 2, 4, 8, 16, 32, 64], 2048, hardware(), "bfloat16"
        )
    ]
    p.write_json(out / "tradeoff_predictions.json", tradeoffs)
    return out


def load(key, implementation):
    p.require_cuda(DEVICE, "bfloat16")
    p.check_capacity(p.MODELS[key], DEVICE, [(1, 128)], 8)
    snapshot = ROOT / "models" / f"qwen3-{key}" / p.REVISIONS[key]
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(
            f"Prepare pinned checkpoint with Chapter 1 Lab 2: {snapshot}"
        )
    from transformers import AutoConfig

    if (
        p.ModelDimensions.from_config(
            AutoConfig.from_pretrained(snapshot, local_files_only=True)
        )
        != p.MODELS[key]
    ):
        raise ValueError("Checkpoint dimensions differ from course ledger")
    print("Loading", key, implementation, flush=True)
    if implementation == "baseline":
        model, _ = opt.reference.load_checkpoint(snapshot, DEVICE, torch.bfloat16)
    else:
        model, _ = opt.load_checkpoint(
            snapshot, DEVICE, optimizations=opt.PRESETS[implementation], strict=True
        )
    return model


def set_flags(model, flags):
    model.optimizations = flags
    for layer in model.layers:
        layer.optimizations = flags


@torch.inference_mode()
def _validate_model_batch(model, batch):
    """Full-vocabulary real-weight checks, outside all timing and capture."""
    ids = torch.tensor([[11, 25, 76, 93, 45, 67, 100]], device=DEVICE).expand(batch, -1)
    flags = getattr(model, "optimizations", None)
    if flags is not None:
        set_flags(model, opt.Optimizations(False, False, False, False, False))
    baseline, _ = model(ids, decode=False)
    if flags is not None:
        set_flags(model, flags)
    actual, _ = model(ids, decode=False)
    _, cache = model(ids[:, :3], decode=True)
    saved = [(k.clone(), v.clone()) for k, v in cache]
    chunk, new = model(ids[:, 3:], cache, decode=False)
    for a, b in [(actual, baseline), (chunk, actual[:, 3:])]:
        torch.testing.assert_close(
            a.float(),
            b.float(),
            atol=TOLERANCES["logits_atol"],
            rtol=TOLERANCES["logits_rtol"],
        )
    for old, copy in zip(cache, saved):
        for a, b in zip(old, copy):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    if any(k.shape[2] != ids.shape[1] for k, v in new):
        raise AssertionError("Cache append mismatch")
    return dict(
        status="passed",
        optimized_vs_baseline_max_abs=float(
            (actual.float() - baseline.float()).abs().max()
        ),
        cached_vs_full_max_abs=float(
            (chunk.float() - actual[:, 3:].float()).abs().max()
        ),
        fixture=ids.cpu().tolist(),
        tolerances=TOLERANCES,
    )


def validate_model(model):
    """Check real full-model logits after changing batch and broadcast layout."""
    cases = [
        dict(batch=batch, **_validate_model_batch(model, batch)) for batch in (1, 4)
    ]
    return dict(
        status="passed",
        cases=cases,
        optimized_vs_baseline_max_abs=max(
            r["optimized_vs_baseline_max_abs"] for r in cases
        ),
        cached_vs_full_max_abs=max(r["cached_vs_full_max_abs"] for r in cases),
        tolerances=TOLERANCES,
    )


def tokens(out, batch, prompt):
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    ids = torch.randint(100, 10000, (batch, prompt), device=DEVICE, generator=gen)
    forced = torch.randint(100, 10000, (batch, 8), device=DEVICE, generator=gen)
    p.write_json(
        out / f"tokens-B{batch}-S{prompt}.json",
        dict(prompt=ids.cpu().tolist(), continuation=forced.cpu().tolist()),
    )
    return forced


def external_profile(
    key,
    implementation,
    batch,
    context,
    phase,
    kind,
    out,
    cases=None,
    permission_evidence=None,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if kind == "ncu" and cases is not None and permission_evidence is None:
        # A launch-count limit applies to the entire process. Separate targets
        # ensure an early prefill kernel cannot consume the decode capture quota.
        statuses = [
            dict(
                **case,
                **external_profile(
                    key,
                    implementation,
                    case["batch"],
                    case["context"],
                    case["phase"],
                    kind,
                    out / f"B{case['batch']}-S{case['context']}-{case['phase']}",
                ),
            )
            for case in cases
        ]
        status = dict(
            profiler=kind,
            status="captured"
            if all(item["status"] == "captured" for item in statuses)
            else "incomplete; inspect per-case statuses",
            cases=statuses,
        )
        p.write_json(out / "status.json", status)
        return status
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "capture",
        "--model",
        key,
        "--implementation",
        implementation,
        "--batch",
        str(batch),
        "--context",
        str(context),
        "--phase",
        phase,
        "--profiler",
        kind,
        "--out",
        str(out / "capture"),
    ]
    if cases is not None:
        p.write_json(out / "cases.json", cases)
        cmd += ["--capture-grid", str(out / "cases.json")]
    if kind == "nsys":
        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=false",
            "-o",
            str(out / "report"),
            *cmd,
        ]
    else:
        cmd = [
            "ncu",
            "--profile-from-start",
            "off",
            "--target-processes",
            "all",
            "--kernel-name",
            "regex:.*(attention_kernel|softmax).*",
            "--launch-count",
            "2",
            "--set",
            "full",
            "-o",
            str(out / "report"),
            *cmd,
        ]
    if kind == "ncu" and permission_evidence is not None:
        evidence = Path(permission_evidence)
        if "ERR_NVGPUCTRPERM" not in evidence.read_text():
            raise ValueError("Permission evidence must contain ERR_NVGPUCTRPERM")
        (out / "command.log").write_text(
            json.dumps(cmd)
            + "\nNot launched: counter permission preflight failed. See "
            + str(evidence)
            + "\n"
        )
        status = dict(
            profiler=kind,
            returncode=None,
            status="unmeasured counters: ERR_NVGPUCTRPERM",
            permission_evidence=str(evidence),
        )
        p.write_json(out / "status.json", status)
        return status
    with (out / "command.log").open("w") as f:
        f.write(json.dumps(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    report = out / ("report.nsys-rep" if kind == "nsys" else "report.ncu-rep")
    status = dict(
        profiler=kind,
        returncode=result.returncode,
        status="captured"
        if result.returncode == 0 and report.is_file() and report.stat().st_size
        else "failed or no report; see command.log",
    )
    if kind == "nsys" and result.returncode == 0:
        with (out / "summary.csv").open("w") as f:
            stats = subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--report",
                    "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum",
                    "--format",
                    "csv",
                    str(out / "report.nsys-rep"),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        status["stats_returncode"] = stats.returncode
    if "ERR_NVGPUCTRPERM" in (out / "command.log").read_text():
        status["status"] = "unmeasured counters: ERR_NVGPUCTRPERM"
    p.write_json(out / "status.json", status)
    return status


def counter_preflight(out):
    """A tiny separate NCU process avoids repeatedly loading weights without permission."""
    log = Path(out) / "ncu-permission-preflight.log"
    cmd = [
        "ncu",
        "--launch-count",
        "1",
        "--set",
        "full",
        sys.executable,
        "-c",
        'import torch; x=torch.ones(128,device="cuda"); torch.cuda.synchronize()',
    ]
    with log.open("w") as f:
        f.write(json.dumps(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log if "ERR_NVGPUCTRPERM" in log.read_text() else None


def execute(
    out, stage="sweep", model_keys=("8b", "32b"), implementations=None, external=True
):
    cases = CASES if stage == "sweep" else [(1, 2048), (4, 2048)]
    implementations = implementations or (
        ["baseline", "optimized"]
        if stage == "sweep"
        else ["baseline"]
        if stage == "baseline"
        else ["optimized", "fusion_only", "attention_only"]
    )
    out = new_run(out, stage, cases)
    rows = []
    statuses = []
    validation = []
    profile_jobs = []
    external_statuses = []
    for key in model_keys:
        retained_model = None
        for implementation in implementations:
            model = retained_model
            try:
                if model is None:
                    model = load(key, implementation)
                else:
                    set_flags(model, opt.PRESETS[implementation])
                    model.fallbacks.clear()
                    model.dispatches.clear()
                validation.append(
                    dict(
                        model=key,
                        implementation=implementation,
                        **validate_model(model),
                    )
                )
                p.write_json(out / "correctness.json", validation)
                selected = (
                    cases
                    if implementation in ("baseline", "optimized")
                    else [(1, 2048)]
                )
                for batch, prompt in selected:
                    try:
                        forced = tokens(out, batch, prompt)
                        print("Measure", key, implementation, batch, prompt, flush=True)
                        before = (
                            len(opt.kernels().COMPILATIONS)
                            if implementation != "baseline"
                            else 0
                        )
                        measurements = p.benchmark_model(
                            model,
                            key,
                            [(batch, prompt)],
                            3,
                            2,
                            8,
                            DEVICE,
                            "bfloat16",
                            implementation,
                            forced,
                            (lambda: len(opt.kernels().COMPILATIONS))
                            if implementation != "baseline"
                            else None,
                        )
                        rows.extend(measurements)
                        status = dict(
                            model=key,
                            implementation=implementation,
                            batch=batch,
                            prompt=prompt,
                            status="measured",
                        )
                        if implementation != "baseline":
                            status.update(model.execution_metadata())
                            status["compilations"] = opt.kernels().COMPILATIONS[before:]
                            if model.fallbacks:
                                raise AssertionError(
                                    "Unexpected fallback in strict run"
                                )
                        statuses.append(status)
                        if stage != "sweep":
                            for phase in ("prefill", "decode"):
                                target = (
                                    out
                                    / f"{key}-{implementation}-B{batch}-S{prompt}-{phase}"
                                )
                                try:
                                    profiling.capture(
                                        model,
                                        batch,
                                        prompt,
                                        phase,
                                        "torch",
                                        target / "torch",
                                    )
                                except torch.cuda.OutOfMemoryError as error:
                                    status.setdefault("profiling_failures", []).append(
                                        dict(
                                            phase=phase,
                                            profiler="torch",
                                            status="unmeasured: memory failure",
                                            detail=str(error),
                                        )
                                    )
                                profile_jobs.append(
                                    (key, implementation, batch, prompt, phase, target)
                                )
                    except torch.cuda.OutOfMemoryError as error:
                        statuses.append(
                            dict(
                                model=key,
                                implementation=implementation,
                                batch=batch,
                                prompt=prompt,
                                status="unmeasured: memory failure",
                                detail=str(error),
                            )
                        )
                    finally:
                        p.write_csv(out / "results.csv", rows)
                        p.write_json(out / "status.json", statuses)
                        gc.collect()
                        torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as error:
                for batch, prompt in cases:
                    statuses.append(
                        dict(
                            model=key,
                            implementation=implementation,
                            batch=batch,
                            prompt=prompt,
                            status="unmeasured: model load memory failure",
                            detail=str(error),
                        )
                    )
            finally:
                retained_model = model if stage == "optimized" else None
                del model
                gc.collect()
                torch.cuda.empty_cache()
                p.write_json(out / "status.json", statuses)
                save_compilation_metadata(out)
        del retained_model
        gc.collect()
        torch.cuda.empty_cache()
    if stage == "baseline":
        importlib.import_module("chapters.03_kernels.code.evidence").freeze_ranked(out)
    # Release resident weights before each separately launched profiler process.
    if external and profile_jobs:
        permission_evidence = counter_preflight(out)
        for key, implementation in dict.fromkeys(
            (job[0], job[1]) for job in profile_jobs
        ):
            cases = [
                dict(batch=b, context=s, phase=phase)
                for k, i, b, s, phase, _ in profile_jobs
                if (k, i) == (key, implementation)
            ]
            for kind in ("nsys", "ncu"):
                print("Profile", kind, key, implementation, cases, flush=True)
                external_statuses.append(
                    dict(
                        model=key,
                        implementation=implementation,
                        **external_profile(
                            key,
                            implementation,
                            1,
                            2048,
                            "prefill",
                            kind,
                            out / f"{key}-{implementation}-{kind}",
                            cases,
                            permission_evidence,
                        ),
                    )
                )
    if not rows:
        # An empty run still has a raw artifact and per-case failure records.
        (out / "results.csv").write_text(
            "model,implementation,batch,prompt,phase,repeat,step,cuda_ms,token_ready_ms\n"
        )
    plots = importlib.import_module("chapters.03_kernels.code.plots")
    plots.render(out, rows)
    p.write_json(
        out / "completion.json",
        dict(
            benchmark_cases=len([r for r in statuses if r["status"] == "measured"]),
            profile_cases=len(profile_jobs),
            external_profilers_requested=external,
            external_statuses=external_statuses,
            pytorch_traces_written=len(list(out.glob("*/torch/trace.json"))),
        ),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "optimized", "sweep", "capture"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=["8b", "32b"])
    parser.add_argument("--implementation", choices=["baseline", *opt.PRESETS])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--profiler", choices=["torch", "nsys", "ncu"], default="torch")
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--capture-grid", type=Path)
    args = parser.parse_args()
    if args.stage == "capture":
        if not args.model or not args.implementation:
            parser.error("capture requires --model and --implementation")
        args.out.mkdir(parents=True, exist_ok=False)
        p.write_json(
            args.out / "manifest.json",
            dict(
                **manifest("capture"),
                model=args.model,
                implementation=args.implementation,
                profiler=args.profiler,
            ),
        )
        model = load(args.model, args.implementation)
        if args.capture_grid:
            profiling.capture_grid(
                model, json.loads(args.capture_grid.read_text()), args.out
            )
        else:
            profiling.capture(
                model, args.batch, args.context, args.phase, args.profiler, args.out
            )
        save_compilation_metadata(args.out)
    else:
        execute(
            args.out,
            args.stage,
            [args.model] if args.model else ("8b", "32b"),
            [args.implementation] if args.implementation else None,
            not args.no_external,
        )


if __name__ == "__main__":
    main()
