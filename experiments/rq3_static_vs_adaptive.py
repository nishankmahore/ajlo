"""RQ3 (comparison): static layouts vs AJLO's adaptive re-clustering after drift.

After the structural shift to regime D (part-key joins), static layouts keep
their original keys; AJLO re-clusters (rq3_physical.py). This step benchmarks
the regime-D queries (+ two regime-A queries) against every existing layout,
in one session so timings are comparable. Nothing is rebuilt.

Layouts (must already exist under LAYOUTS_DIR):
  b0            no layout
  b2            per-table filter columns (ALC-style), static
  b3            per-table most-predicated column incl. join keys, static
  ajlo          AJLO plan before drift (pre-shift), static
  rq3           AJLO after its drift-triggered re-clustering (re-clustered; run rq3phys first)

Output:
  results/rq3_static_vs_adaptive.csv
  results/rq3_static_vs_adaptive.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from experiments.config import LAYOUTS_DIR, RESULTS_DIR
from experiments.experiment_queries import get_rq3_queries
from experiments.rq3_physical import _benchmark
from experiments.spark_utils import get_spark_session

console = Console()

LAYOUTS = [
    ("b0",   "B0\n(no layout)",          "#d62728"),
    ("b2",   "B2\n(filter keys,\nstatic)", "#ff7f0e"),
    ("b3",   "B3\n(per-table join-\naware, static)", "#9467bd"),
    ("ajlo", "AJLO\n(pre-shift)",       "#98df8a"),
    ("rq3",  "AJLO\n(re-clustered)",    "#2ca02c"),
]


def run(spark=None, sf: int = 1) -> list[dict]:
    console.print(f"\n=== RQ3 (comparison): static layouts vs adaptive AJLO (SF={sf}) ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if spark is None:
        spark = get_spark_session("rq3-static-vs-adaptive")

    queries = get_rq3_queries(sf)
    rows = []
    for name, _, _ in LAYOUTS:
        path = LAYOUTS_DIR / name
        if not path.exists():
            console.print(f"  [SKIP] {name}: layout not found at {path}")
            continue
        console.print(f"\n  Layout: {name}")
        for r in _benchmark(spark, path, queries, phase=name):
            r["layout"] = r.pop("phase")
            rows.append(r)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "rq3_static_vs_adaptive.csv"
    df.to_csv(out_csv, index=False)
    console.print(f"  CSV → {out_csv}")
    if not df.empty:
        _summarise(df)
        _plot(df)
    return rows


def _summarise(df: pd.DataFrame) -> None:
    d = df[df.query_id.str.startswith("D")]
    a = df[~df.query_id.str.startswith("D")]
    console.print("\n  Mean per query (regime D | regime A):")
    for name, _, _ in LAYOUTS:
        if name in set(df.layout):
            dd, aa = d[d.layout == name], a[a.layout == name]
            console.print(f"    {name:5s}: D {dd.bytes_read.mean()/1e9:6.2f} GB, "
                          f"{dd.skip_pct.mean():5.1f}% skip, {dd.wall_time_ms.mean():6.0f} ms  |  "
                          f"A {aa.bytes_read.mean()/1e9:6.2f} GB")


def _plot(df: pd.DataFrame) -> None:
    present = [l for l in LAYOUTS if l[0] in set(df.layout)]
    d = df[df.query_id.str.startswith("D")]
    means = [d[d.layout == n].bytes_read.mean() / 1e9 for n, _, _ in present]
    lat = d.groupby("layout").apply(
        lambda g: (g.wall_time_ms.mean() / 1e3, np.sqrt((g.wall_time_std_ms ** 2).mean()) / 1e3))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar([l[1] for l in present], means, color=[l[2] for l in present],
                       edgecolor="black", linewidth=0.5)
    axes[0].bar_label(bars, fmt="%.1f GB", padding=2, fontsize=9)
    axes[0].set_ylabel("Mean Bytes Read per Query (GB)\n[Delta min/max statistics]")
    axes[0].set_title("Regime-D queries after the workload shift (lower is better)", fontsize=10)

    m = [lat[n][0] for n, _, _ in present]
    e = [lat[n][1] for n, _, _ in present]
    bars = axes[1].bar([l[1] for l in present], m, yerr=e, capsize=5,
                       color=[l[2] for l in present], edgecolor="black", linewidth=0.5)
    axes[1].bar_label(bars, labels=[f"{x:.2f} s" for x in m], padding=3, fontsize=9)
    axes[1].set_ylabel("Mean Query Latency (s)\n[error bar: pooled run-to-run std]")
    axes[1].set_title("Regime-D query latency (lower is better)", fontsize=10)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("RQ3 — Static Layouts vs. AJLO Adaptive Re-clustering After Structural Drift")
    plt.tight_layout()
    out = RESULTS_DIR / "rq3_static_vs_adaptive.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    plt.savefig(RESULTS_DIR / "eps" / "rq3_static_vs_adaptive.eps", format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {out}")


if __name__ == "__main__":
    run()
