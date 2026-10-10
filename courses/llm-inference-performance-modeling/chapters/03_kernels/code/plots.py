"""Matched implementation plots and speedup tables; missing cases stay missing."""

import csv
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from shared import performance as p


def save(fig, path):
    for suffix in ("png", "svg"):
        fig.savefig(str(path) + "." + suffix, bbox_inches="tight")
    plt.close(fig)


def speedups(summary):
    paired = []
    index = {
        (r["model"], r["batch"], r["prompt"], r["phase"], r["implementation"]): r
        for r in summary
    }
    for (model, batch, prompt, phase, implementation), r in index.items():
        if implementation == "baseline":
            continue
        b = index.get((model, batch, prompt, phase, "baseline"))
        if b is None:
            continue
        paired.append(
            dict(
                model=model,
                batch=batch,
                prompt=prompt,
                phase=phase,
                implementation=implementation,
                baseline_ms=b["latency_ms"],
                candidate_ms=r["latency_ms"],
                speedup=b["latency_ms"] / r["latency_ms"],
                baseline_min_ms=b["latency_min_ms"],
                baseline_max_ms=b["latency_max_ms"],
                candidate_min_ms=r["latency_min_ms"],
                candidate_max_ms=r["latency_max_ms"],
                range_low=b["latency_min_ms"] / r["latency_max_ms"],
                range_high=b["latency_max_ms"] / r["latency_min_ms"],
                range_kind="observed extrema ratios, not confidence interval",
            )
        )
    return paired


def amdahl(baseline_ms, fraction, local_speedup, adapter_ms=0):
    if (
        not 0 <= fraction <= 1
        or baseline_ms <= 0
        or local_speedup <= 0
        or adapter_ms < 0
    ):
        raise ValueError("Invalid Amdahl inputs")
    predicted_ms = (
        baseline_ms * ((1 - fraction) + fraction / local_speedup) + adapter_ms
    )
    return dict(predicted_ms=predicted_ms, predicted_speedup=baseline_ms / predicted_ms)


def render(out, rows=None):
    out = Path(out)
    if rows is None:
        # Preserve schema types by reading the JSON summary for report regeneration.
        raise ValueError("Pass raw typed rows; notebooks retain these via read_rows")
    summary = p.summarize_model_rows(rows)
    p.write_json(out / "summary.json", summary)
    p.write_csv(out / "summary.csv", summary)
    p.write_csv(out / "speedups.csv", speedups(summary))
    hw = p.Hardware(**json.loads((out / "manifest.json").read_text())["hardware"])
    fig, _ = p.plot_roofline(summary, hw, "bfloat16", compact=True)
    save(fig, out / "roofline")
    for model, fig in p.plot_latency_sweeps(summary, x_axis="batch").items():
        save(fig, out / f"latency_throughput_{model}")
    predictions = json.loads((out / "tradeoff_predictions.json").read_text())
    observed = p.summarize_tradeoff_observations(rows, 2048)
    p.write_json(out / "tradeoff_observations.json", observed)
    fig, _ = p.plot_tradeoffs(
        predictions, 2048, "bfloat16", hw.compute_basis, observed, compact_labels=True
    )
    save(fig, out / "tradeoffs")
    p.write_json(
        out / "plot_sources.json",
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(p.__file__))
        },
    )


def read_rows(path):
    rows = list(csv.DictReader(Path(path).open()))
    strings = {"model", "implementation", "phase", "throughput_token_kind"}
    integers = {
        "batch",
        "prompt",
        "repeat",
        "step",
        "prefix_tokens",
        "attended_positions",
        "throughput_tokens",
    }
    return [
        {
            k: v if k in strings else int(v) if k in integers else float(v)
            for k, v in row.items()
        }
        for row in rows
    ]
