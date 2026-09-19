"""Trace attribution and explicitly assumed Amdahl forecasts, frozen before changes."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from shared import performance as p

MECHANISMS = {
    "attention": "CuTe causal GQA: no global scores/probabilities or expanded KV",
    "normalize_rotary": "Fuse Q/K head norm with absolute-position rotate-half",
    "normalize": "Fuse RMS reduction, normalization and scale",
    "activation": "Fuse SiLU and multiplication",
    "cache": "Retain dynamic concatenation in this chapter; runtime ownership is later work",
    "host_idle": "Investigate launch gaps; fusion may reduce dispatch, but idle is not all host overhead",
}


def attribute_trace(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise ValueError("No CUDA kernels in trace")
    cpu = {
        e.get("args", {}).get("External id"): e
        for e in events
        if e.get("cat") in ("cpu_op", "user_annotation")
        and e.get("args", {}).get("External id") is not None
    }
    ranges = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e["name"]
        in ("attention", "normalize_rotary", "normalize", "activation", "norm")
    ]
    cats = [e for e in events if e.get("cat") == "cpu_op" and e["name"] == "aten::cat"]
    totals = defaultdict(float)
    counts = defaultdict(int)
    for kernel in kernels:
        operator = cpu.get(kernel.get("args", {}).get("External id"))
        # Direct CuTe driver launches need not have a PyTorch External id.
        # Their generated names identify the operator without CPU correlation.
        custom = next(
            (
                operation
                for tag, operation in (
                    ("attention", "attention"),
                    ("qk_rope", "normalize_rotary"),
                    ("rms", "normalize"),
                    ("swiglu", "activation"),
                )
                if f"cutlass_{tag}_kernel" in kernel["name"]
            ),
            None,
        )
        name = custom or "other"
        if operator and custom is None:
            ts = operator["ts"]
            enclosing = [
                r
                for r in ranges
                if r["tid"] == operator["tid"] and r["ts"] <= ts <= r["ts"] + r["dur"]
            ]
            if enclosing:
                name = min(enclosing, key=lambda r: r["dur"])["name"]
                if name == "norm":
                    name = "normalize"
            elif any(
                r["tid"] == operator["tid"] and r["ts"] <= ts <= r["ts"] + r["dur"]
                for r in cats
            ):
                name = "cache"
        totals[name] += kernel["dur"]
        counts[name] += 1
    begin = min(k["ts"] for k in kernels)
    end = max(k["ts"] + k["dur"] for k in kernels)
    span = end - begin
    # Union, not sum, of kernel intervals: robust to overlap on different streams.
    covered = 0.0
    right = begin
    for k in sorted(kernels, key=lambda e: e["ts"]):
        stop = k["ts"] + k["dur"]
        covered += max(0, stop - max(right, k["ts"]))
        right = max(right, stop)
    totals["host_idle"] = max(0, span - covered)
    return dict(
        span_us=span,
        kernel_busy_us=covered,
        kernel_count=len(kernels),
        categories_us=dict(totals),
        categories_kernel_count=dict(counts),
    )


def freeze_ranked(out):
    """Local 2x assumptions are hypotheses, not profiler speed measurements."""
    out = Path(out)
    rows = []
    for trace in sorted(out.glob("*-baseline-B*-S*-*/torch/trace.json")):
        key, impl, batch, prompt, phase = trace.parent.parent.name.split("-")
        b, s = int(batch[1:]), int(prompt[1:])
        c = p.MODELS[key]
        attribution = attribute_trace(trace)
        p.write_json(trace.parent / "attribution.json", attribution)
        t = s if phase == "prefill" else 1
        prefix = 0 if phase == "prefill" else s
        traffic = {
            "attention": 16 * b * c.query_heads * t * (prefix + t)
            + 4 * b * c.query_heads * (prefix + t) * c.head_dim,
            "activation": 4 * b * t * c.intermediate,
            "normalize": 8 * b * t * c.hidden,
            "normalize_rotary": 4 * b * t * (c.query_heads + c.kv_heads) * c.head_dim,
            "cache": 0,
            "host_idle": 0,
        }
        for name, mechanism in MECHANISMS.items():
            fraction = (
                attribution["categories_us"].get(name, 0) / attribution["span_us"]
            )
            proposed = name in (
                "attention",
                "normalize_rotary",
                "normalize",
                "activation",
            )
            local_speedup = 2.0 if proposed else 1.0
            integrated = 1 / ((1 - fraction) + fraction / local_speedup)
            rows.append(
                dict(
                    model=key,
                    batch=b,
                    prompt=s,
                    phase=phase,
                    operation=name,
                    evidence=str(trace.relative_to(out)),
                    mechanism=mechanism,
                    expected_logical_byte_reduction_per_layer=traffic[name],
                    bytes_scope="illustrative intermediate write/read ledger; not measured traffic",
                    baseline_time_fraction=fraction,
                    fraction_scope="attributed kernel time / instrumented CUDA span; idle separate",
                    baseline_kernel_count=attribution["categories_kernel_count"].get(
                        name, 0
                    ),
                    predicted_local_speedup=local_speedup,
                    local_speedup_basis="assumed hypothesis, frozen before optimized run",
                    predicted_integrated_speedup=integrated,
                )
            )
    rows.sort(
        key=lambda r: (
            r["model"],
            r["batch"],
            r["phase"],
            -r["predicted_integrated_speedup"],
        )
    )
    p.write_csv(out / "ranked_optimizations.csv", rows)
    p.write_json(out / "ranked_optimizations.json", rows)
    return rows


def compare(baseline, optimized, out, measurements=None):
    """Join baseline trace forecasts, both measured sweeps and trace changes."""
    from importlib import import_module

    plots = import_module("chapters.03_kernels.code.plots")
    baseline, optimized, out = Path(baseline), Path(optimized), Path(out)
    forecasts = json.loads((baseline / "ranked_optimizations.json").read_text())
    observations = plots.read_rows(baseline / "results.csv") + plots.read_rows(
        optimized / "results.csv"
    )
    if measurements is not None:
        observations = list(measurements) + [
            r
            for r in observations
            if r["implementation"] in ("fusion_only", "attention_only")
        ]
    summary = p.summarize_model_rows(observations)
    rows = plots.speedups(summary)
    groups = {
        "optimized": {"attention", "normalize", "activation", "normalize_rotary"},
        "fusion_only": {"normalize", "activation", "normalize_rotary"},
        "attention_only": {"attention"},
    }
    operator_records = {}
    for r in rows:
        matches = [
            f
            for f in forecasts
            if all(f[k] == r[k] for k in ("model", "batch", "prompt", "phase"))
            and f["operation"] in groups[r["implementation"]]
        ]
        saved = sum(
            f["baseline_time_fraction"] * (1 - 1 / f["predicted_local_speedup"])
            for f in matches
        )
        r["amdahl_predicted_speedup"] = 1 / (1 - saved) if matches else None
        r["amdahl_scope"] = (
            "frozen assumed 2x replacements, perturbed trace fractions, no assumed adapter benefit"
            if matches
            else "unmeasured baseline trace fraction at this workload"
        )
        for directory, implementation in [
            (baseline, "baseline"),
            (optimized, r["implementation"]),
        ]:
            trace = (
                directory
                / f"{r['model']}-{implementation}-B{r['batch']}-S{r['prompt']}-{r['phase']}"
                / "torch/trace.json"
            )
            if trace.exists():
                attr = attribute_trace(trace)
                r[f"{implementation}_kernel_count"] = attr["kernel_count"]
                r[f"{implementation}_trace_span_us"] = attr["span_us"]
                r[f"{implementation}_trace_idle_us"] = attr["categories_us"].get(
                    "host_idle", 0
                )
                r[f"{implementation}_trace_attention_us"] = attr["categories_us"].get(
                    "attention", 0
                )
                if str(trace) not in operator_records:
                    selected = {
                        "aten::repeat_interleave",
                        "aten::bmm",
                        "aten::_softmax",
                        "aten::cat",
                        "aten::silu",
                        "aten::mul",
                    }
                    with (trace.parent / "operators.csv").open() as file:
                        operator_records[str(trace)] = [
                            dict(
                                model=r["model"],
                                batch=r["batch"],
                                prompt=r["prompt"],
                                phase=r["phase"],
                                implementation=implementation,
                                evidence=str(trace.parent / "operators.csv"),
                                **operator,
                            )
                            for operator in csv.DictReader(file)
                            if operator["operator"] in selected
                        ]
        candidate_count = r.get(f"{r['implementation']}_kernel_count")
        baseline_count = r.get("baseline_kernel_count")
        r["kernel_count_reduction"] = (
            baseline_count - candidate_count
            if baseline_count is not None and candidate_count is not None
            else None
        )
        r["speedup_relative_to_amdahl"] = (
            r["speedup"] / r["amdahl_predicted_speedup"] if matches else None
        )
        if r["kernel_count_reduction"] is None:
            r["interpretation"] = (
                "Matched latency measured; no trace at this exact workload. Mechanism extrapolation from 2048-token cases remains a hypothesis."
            )
        elif r["speedup"] > 1:
            r["interpretation"] = (
                "Fewer launches accompany lower latency. Attention score/KV expansion removal and fusion are supported by source and traces; GEMMs/cache concatenation remain. Counter traffic is unmeasured."
            )
        else:
            r["interpretation"] = (
                "Fewer launches do not guarantee a speedup. Use attention-only/fusion-only ablations and the component adapter timings to distinguish SIMT work, rereads, and dispatch cost. Counter traffic is unmeasured."
            )

    # Implementation-specific columns: write a stable union schema.
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    if rows:
        p.write_csv(
            out / "amdahl_comparison.csv", [{k: r.get(k) for k in fields} for r in rows]
        )
    p.write_json(out / "amdahl_comparison.json", rows)
    p.write_csv(
        out / "intermediate_operator_records.csv",
        [row for records in operator_records.values() for row in records],
    )
    p.write_json(
        out / "profile_comparison_scope.json",
        dict(
            analysis_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            trace_times="Instrumented CUDA spans/kernel times; excluded from benchmark observations. Idle is not entirely host overhead.",
            allocations="Operator device allocation bytes are inclusive, may overlap, and must not be summed across nested operators. These are not counter-measured DRAM bytes.",
            absent_operators="An absent selected operator has no recorded invocation in this capture; inspect shapes and the linked full table.",
        ),
    )
    return rows
