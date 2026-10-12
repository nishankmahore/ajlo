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
from experiments.query_logs import generate_query_log, get_cardinalities

from experiments.config import RESULTS_DIR

DRIFT_DELTA    = 0.10
WINDOW_DAYS    = 7
QUERIES_PER_WINDOW = 100


def _build_graph(log: list[dict], start: datetime, end: datetime):
    g = extract_join_graph(log, (start, end))
    return compute_jis(g)


# W1–W2: regime A (orderkey joins); W3–W4: regime B (same join predicates,
# different frequencies); W5–W6: regime D (new dominant predicate: part-key joins).
SCHEDULE: list[tuple[str, str]] = [
    ("W1", "A"), ("W2", "A"),
    ("W3", "B"), ("W4", "B"),
    ("W5", "D"), ("W6", "D"),
]


def _key_of(strategy) -> str:
    return (strategy.cluster_cols[0] if strategy.cluster_cols
            else strategy.partition_col or "—")


def simulate_drift(sf: int = 1) -> list[dict]:
    """Replay the window schedule through Algorithms 1→2→4→5.

    Returns one dict per window with the graph, the plan in force after the
    window, and the reorganisation ops executed at that window. W1 builds the
    initial layout (all of its tables count as rewritten). Shared by rq3_drift
    (cost model view) and rq3_physical (executes the same ops on Delta tables).
    """
    from ajlo.algorithms.drift_detector import detect_drift
    from ajlo.algorithms.layout_planner import recommend_layout
    from ajlo.models import ReorgOp, LayoutStrategy

    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    base = datetime(2024, 1, 1)
    steps = []
    prev_graph = None
    plan = LayoutPlan()

    for idx, (wid, regime) in enumerate(SCHEDULE):
        start = base + timedelta(days=idx * WINDOW_DAYS)
        end   = start + timedelta(days=WINDOW_DAYS)
        log = generate_query_log(regime, QUERIES_PER_WINDOW, start_time=start,
                                 seed=idx * 7, sf=sf)
        graph = _build_graph(log, start, end)

        if prev_graph is None:
            plan = recommend_layout(graph, LayoutPlan(), rewrite_budget=1e18,
                                    storage_budget=1e18, table_sizes=table_sizes)
            ops = [ReorgOp(table=t, from_strategy=LayoutStrategy(table=t),
                           to_strategy=s, priority=0.0)
                   for t, s in plan.strategies.items() if s.strategy_type != "NOOP"]
            detected, score = False, 0.0
        else:
            detected, score = detect_drift(prev_graph, graph, DRIFT_DELTA)
            plan, ops = adaptive_reoptimize(
                prev_graph, graph, plan,
                rewrite_budget=1e18, storage_budget=1e18,
                delta=DRIFT_DELTA, table_sizes=table_sizes,
            )

        steps.append({"window": wid, "regime": regime, "graph": graph,
                      "plan": plan.copy(), "ops": ops,
                      "drift_detected": detected, "drift_score": score,
                      "initial": prev_graph is None})
        prev_graph = graph
    return steps


def run(sf: int = 1) -> list[dict]:
    console.print(f"\n=== RQ3: Workload Drift Detection & Adaptation (SF={sf}) ===")
    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for step in track(simulate_drift(sf), description="Windows"):
        ops = step["ops"]
        rewrite_bytes = sum(table_sizes.get(op.table, 1e9) for op in ops)

        # Expected per-window scan reduction of the plan in force (real bytes)
        scan_red = 0.0
        for edge in step["graph"].edges:
            strat = step["plan"].get(edge.table_i).strategy_type
            exp = expected_scan_bytes(edge.frequency, edge.data_volume)
            scan_red += scan_reduction(exp, strat)

        rows.append({
            "window":                     step["window"],
            "regime":                     step["regime"],
            "phase":                      "initial layout" if step["initial"] else "drift check",
            "drift_score":                round(step["drift_score"], 4),
            "drift_detected":             step["drift_detected"],
            "reorg_ops":                  len(ops),
            "reorg_tables":               "; ".join(f"{op.table}→{_key_of(op.to_strategy)}"
                                                    for op in ops),
            "rewrite_cost_gb":            round(rewrite_bytes / 1e9, 3),
            "expected_scan_reduction_gb": round(scan_red / 1e9, 3),
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "rq3.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"  CSV → {_link(csv_path)}")
    for r in rows:
        console.print(f"  {r['window']} ({r['regime']}, {r['phase']}): "
                      f"drift={r['drift_score']:.3f} detected={r['drift_detected']} "
                      f"ops={r['reorg_ops']} [{r['reorg_tables']}]")

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
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq3_drift_score.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {_link(out)}")


def _plot_reorg_cost(rows):
    windows = [r["window"] for r in rows]
    costs   = [r["rewrite_cost_gb"] for r in rows]
    red     = [r["expected_scan_reduction_gb"] for r in rows]

    x = np.arange(len(windows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, costs, width, label="Rewrite Cost (GB)", color="#d62728",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, red, width, label="Expected Scan Reduction (GB)",
           color="#2ca02c", edgecolor="black", linewidth=0.6)

    for xi, r in zip(x, rows):
        if r["reorg_ops"]:
            ax.annotate(r["reorg_tables"].replace("; ", "\n"),
                        xy=(xi - width / 2, r["rewrite_cost_gb"]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['window']}\n({r['regime']}, initial layout)" if r["phase"] == "initial layout"
                        else f"{r['window']}\n({r['regime']})" for r in rows])
    ax.set_ylabel("Data Volume per Window (GB)")
    ax.set_title("RQ3 — Rewrite Cost vs. Expected Scan Reduction per Window\n"
                 "(W1 builds the initial layout; later bars are drift-triggered rewrites)")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "rq3_reorg_cost.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq3_reorg_cost.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {_link(out)}")


if __name__ == "__main__":
    run()
