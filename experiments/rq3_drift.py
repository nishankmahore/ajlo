"""RQ3: Does AJLO adapt effectively to workload drift, and at what cost?

Simulates three workload regimes (A → B → C) from Section 9.3.
Measures:
  - Drift score per window transition
  - Whether drift was correctly detected
  - Number of reorg ops triggered
  - Estimated rewrite cost per re-optimization

Output:
  results/rq3.csv
  results/rq3_drift_score.png
  results/rq3_reorg_cost.png
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()


def _link(path) -> str:
    return f"[link=file://{path}]{path}[/link]"

from ajlo.algorithms.join_graph import extract_join_graph
from ajlo.algorithms.jis import compute_jis
from ajlo.algorithms.adaptive_reopt import adaptive_reoptimize
from ajlo.models import LayoutPlan
from ajlo.cost_model import expected_scan_bytes, scan_reduction
from experiments.query_logs import generate_drift_log, generate_query_log, get_cardinalities

RESULTS_DIR = Path(__file__).parent.parent / "results"

DRIFT_DELTA    = 0.10
WINDOW_DAYS    = 7
QUERIES_PER_WINDOW = 100


def _build_graph(log: list[dict], start: datetime, end: datetime):
    g = extract_join_graph(log, (start, end))
    return compute_jis(g)


def run(sf: int = 1) -> list[dict]:
    console.print(f"\n=== RQ3: Workload Drift Detection & Adaptation (SF={sf}) ===")
    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    RESULTS_DIR.mkdir(exist_ok=True)

    base = datetime(2024, 1, 1)

    # Generate logs for all three regimes, each covering one window
    regimes: list[tuple[str, str]] = [
        ("W1", "A"), ("W2", "A"),                # stable regime A
        ("W3", "B"), ("W4", "B"),                # shift to B
        ("W5", "C"), ("W6", "C"),                # shift to C
    ]

    logs = {}
    for wid, regime in regimes:
        idx = int(wid[1]) - 1
        start = base + timedelta(days=idx * WINDOW_DAYS)
        end   = start + timedelta(days=WINDOW_DAYS)
        logs[wid] = (generate_query_log(regime, QUERIES_PER_WINDOW,
                                        start_time=start, seed=idx * 7),
                     start, end, regime)

    # Build a graph per window
    graphs = {}
    for wid, (log, start, end, regime) in logs.items():
        graphs[wid] = _build_graph(log, start, end)

    # Simulate AJLO's adaptive re-optimization across windows
    current_plan = LayoutPlan()
    rows = []
    prev_wid = "W1"

    for i, (wid, (log, start, end, regime)) in track(
        enumerate(logs.items()), description="Windows", total=len(logs)
    ):
        if i == 0:
            # Seed initial plan from first window
            from ajlo.algorithms.layout_planner import recommend_layout
            g0 = graphs[wid]
            current_plan = recommend_layout(
                g0, LayoutPlan(),
                rewrite_budget=1e12, storage_budget=1e12,
                table_sizes=table_sizes,
            )
            rows.append({
                "window": wid, "regime": regime,
                "drift_score": 0.0, "drift_detected": False,
                "reorg_ops": 0, "rewrite_cost_gb": 0.0,
                "expected_scan_reduction_mb": 0.0,
            })
            continue

        g_old = graphs[prev_wid]
        g_new = graphs[wid]

        new_plan, ops = adaptive_reoptimize(
            g_old, g_new, current_plan,
            rewrite_budget=1e12, storage_budget=1e12,
            delta=DRIFT_DELTA,
            table_sizes=table_sizes,
        )

        from ajlo.algorithms.drift_detector import detect_drift
        detected, drift_score = detect_drift(g_old, g_new, DRIFT_DELTA)

        # Estimate total rewrite cost for triggered ops
        rewrite_bytes = sum(
            table_sizes.get(op.table, 1e9) for op in ops
        )

        # Estimate scan reduction from the new plan
        scan_red = 0.0
        for edge in g_new.edges:
            strat = new_plan.get(edge.table_i).strategy_type
            exp = expected_scan_bytes(edge.frequency, edge.data_volume)
            scan_red += scan_reduction(exp, strat)

        rows.append({
            "window":                    wid,
            "regime":                    regime,
            "drift_score":               round(drift_score, 4),
            "drift_detected":            detected,
            "reorg_ops":                 len(ops),
            "rewrite_cost_gb":           round(rewrite_bytes / 1e9, 3),
            "expected_scan_reduction_mb": round(scan_red / 1e6, 1),
        })

        if detected:
            current_plan = new_plan
        prev_wid = wid

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "rq3.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"  CSV → {_link(csv_path)}")
    for r in rows:
        console.print(f"  {r['window']} ({r['regime']}): drift={r['drift_score']:.3f} "
                      f"detected={r['drift_detected']} ops={r['reorg_ops']}")

    _plot_drift_score(rows)
    _plot_reorg_cost(rows)

    return rows


def _plot_drift_score(rows):
    windows = [r["window"] for r in rows]
    scores  = [r["drift_score"] for r in rows]
    detected = [r["drift_detected"] for r in rows]
    regimes  = [r["regime"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(windows, scores, "o-", color="#1f77b4", linewidth=2, markersize=8,
            label="Drift Score")
    ax.axhline(y=0.10, color="red", linestyle="--", linewidth=1.5,
               label=f"Drift threshold δ={DRIFT_DELTA}")

    for i, (w, s, d, reg) in enumerate(zip(windows, scores, detected, regimes)):
        color = "#d62728" if d else "#2ca02c"
        marker = "▲" if d else "●"
        ax.annotate(f"Regime {reg}\n{marker}",
                    xy=(w, s), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color=color)

    ax.set_xlabel("Window")
    ax.set_ylabel("Drift Score")
    ax.set_title("RQ3 — Workload Drift Score Across Windows\n"
                 "(red ▲ = drift detected, re-optimization triggered)")
    ax.legend()
    ax.set_ylim(-0.05, max(scores + [0.15]) * 1.2)
    plt.tight_layout()
    out = RESULTS_DIR / "rq3_drift_score.png"
    plt.savefig(out, dpi=150)
    plt.close()
    console.print(f"  Chart → {_link(out)}")


def _plot_reorg_cost(rows):
    windows = [r["window"] for r in rows]
    costs   = [r["rewrite_cost_gb"] for r in rows]
    red     = [r["expected_scan_reduction_mb"] / 1024 for r in rows]  # GB

    x = np.arange(len(windows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, costs, width, label="Rewrite Cost (GB)", color="#d62728",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, red, width, label="Expected Scan Reduction (GB)",
           color="#2ca02c", edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Data Volume (GB)")
    ax.set_title("RQ3 — Rewrite Cost vs. Expected Scan Reduction per Window")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "rq3_reorg_cost.png"
    plt.savefig(out, dpi=150)
    plt.close()
    console.print(f"  Chart → {_link(out)}")


if __name__ == "__main__":
    run()
