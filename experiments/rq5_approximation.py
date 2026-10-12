"""RQ5: How close is the greedy approximation to optimal?

Generates small synthetic join graphs (5–8 tables, 6–10 edges) where
brute-force optimal is feasible (3^n combinations with budget filtering).
Compares greedy NPV vs optimal NPV and reports the approximation ratio.

No Spark needed — pure algorithm evaluation.

Output:
  results/rq5.csv
  results/rq5_approx_ratio.png
  results/rq5_npv_comparison.png
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()


def _link(path) -> str:
    return f"[link=file://{path}]{path}[/link]"

from ajlo.models import JoinEdge, JoinGraph, LayoutPlan, LayoutStrategy
from ajlo.algorithms.jis import compute_jis
from ajlo.algorithms.layout_planner import recommend_layout
from ajlo.cost_model import (
    expected_scan_bytes, scan_reduction, rewrite_cost,
    storage_cost, npv, BYTES_PER_ROW, SCAN_OVERHEAD
)

from experiments.config import RESULTS_DIR
THEORETICAL_BOUND = 1.0 - 1.0 / np.e   # ≈ 0.6321 (Sviridenko bound)


# ── Synthetic graph generator ─────────────────────────────────────────────────

def _make_synthetic_graph(
    n_tables: int,
    n_edges: int,
    seed: int = 0,
) -> tuple[JoinGraph, dict[str, float]]:
    """Generate a random join graph with known properties."""
    rng = np.random.default_rng(seed)
    tables = [f"t{i}" for i in range(n_tables)]
    table_sizes = {t: float(rng.uniform(1e8, 5e9)) for t in tables}  # 100 MB – 5 GB

    edges = []
    used = set()
    attempts = 0
    while len(edges) < n_edges and attempts < 1000:
        i, j = rng.integers(0, n_tables, size=2)
        if i == j:
            attempts += 1
            continue
        ti, tj = tables[min(i, j)], tables[max(i, j)]
        key = f"{ti}-{tj}"
        if key in used:
            attempts += 1
            continue
        used.add(key)
        edges.append(JoinEdge(
            table_i=ti, col_i="id",
            table_j=tj, col_j="id",
            frequency=int(rng.integers(5, 100)),
            data_volume=float(rng.uniform(1e5, 1e8)),
            exec_cost=float(rng.uniform(500, 20000)),
        ))
        attempts = 0

    from datetime import datetime
    g = JoinGraph(
        edges=edges,
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 1, 8),
    )
    return compute_jis(g), table_sizes


# ── Brute-force optimal ───────────────────────────────────────────────────────

def _plan_npv(
    assignments: dict[str, str],   # table → strategy_type
    graph: JoinGraph,
    table_sizes: dict[str, float],
    H: int = 4,
    r: float = 0.10,
) -> float:
    """Compute total NPV for a given table→strategy assignment."""
    seen: set[str] = set()
    total = 0.0
    for edge in graph.edges:
        strat_i = assignments.get(edge.table_i, "NOOP")
        strat_j = assignments.get(edge.table_j, "NOOP")
        # Edge is "covered" if both sides use the same non-NOOP strategy
        if strat_i == strat_j and strat_i != "NOOP":
            strat = strat_i
        else:
            strat = "NOOP"

        exp = expected_scan_bytes(edge.frequency, edge.data_volume,
                                  BYTES_PER_ROW, SCAN_OVERHEAD)
        sr = scan_reduction(exp, strat)

        rc = 0.0
        for table, s in ((edge.table_i, strat_i), (edge.table_j, strat_j)):
            if table not in seen and s != "NOOP":
                rc += rewrite_cost(table_sizes.get(table, 1e9), "NOOP", s)
                seen.add(table)

        sc = sum(
            storage_cost(table_sizes.get(t, 1e9), s)
            for t, s in ((edge.table_i, strat_i), (edge.table_j, strat_j))
            if s != "NOOP"
        )
        total += npv(sr, rc, sc, H, r)
    return total


def brute_force_optimal(
    graph: JoinGraph,
    table_sizes: dict[str, float],
    rewrite_budget: float,
    storage_budget: float,
    H: int = 4,
    r: float = 0.10,
) -> tuple[float, dict[str, str]]:
    """Enumerate all strategy combinations. Feasible for ≤8 tables (3^8 = 6561)."""
    tables = list(graph.tables())
    strategies = ["NOOP", "CO_CLUSTER", "CO_PARTITION"]

    best_npv_val = float("-inf")
    best_assign: dict[str, str] = {t: "NOOP" for t in tables}

    for combo in itertools.product(strategies, repeat=len(tables)):
        assignments = dict(zip(tables, combo))

        # Check budget
        rc = sum(
            rewrite_cost(table_sizes.get(t, 1e9), "NOOP", s)
            for t, s in assignments.items() if s != "NOOP"
        )
        sc = sum(
            storage_cost(table_sizes.get(t, 1e9), s)
            for t, s in assignments.items() if s != "NOOP"
        )
        if rc > rewrite_budget or sc > storage_budget:
            continue

        val = _plan_npv(assignments, graph, table_sizes, H, r)
        if val > best_npv_val:
            best_npv_val = val
            best_assign = assignments

    return best_npv_val, best_assign


def _greedy_npv(
    graph: JoinGraph,
    table_sizes: dict[str, float],
    rewrite_budget: float,
    storage_budget: float,
) -> tuple[float, LayoutPlan]:
    plan = recommend_layout(
        graph, LayoutPlan(), rewrite_budget, storage_budget,
        table_sizes=table_sizes,
    )
    assignments = {t: plan.get(t).strategy_type for t in graph.tables()}
    val = _plan_npv(assignments, graph, table_sizes)
    return val, plan


# ── Main experiment ───────────────────────────────────────────────────────────

def run() -> list[dict]:
    console.print("\n=== RQ5: Greedy Approximation Quality ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Test configurations: (n_tables, n_edges, budget_fraction)
    configs = [
        (5, 6,  0.50),
        (5, 7,  0.50),
        (6, 8,  0.40),
        (6, 9,  0.40),
        (7, 9,  0.30),
        (7, 10, 0.30),
        (8, 10, 0.25),
        (8, 11, 0.25),
        (8, 12, 0.20),
    ]

    rows = []
    for i, (n_tables, n_edges, budget_frac) in track(
        enumerate(configs), description="Configs", total=len(configs)
    ):
        graph, table_sizes = _make_synthetic_graph(n_tables, n_edges, seed=i * 13)
        total_size = sum(table_sizes.values())
        rewrite_budget = total_size * budget_frac
        storage_budget = total_size * 0.10

        opt_npv, opt_assign = brute_force_optimal(
            graph, table_sizes, rewrite_budget, storage_budget
        )
        greedy_npv_val, greedy_plan = _greedy_npv(
            graph, table_sizes, rewrite_budget, storage_budget
        )

        ratio = (greedy_npv_val / opt_npv) if opt_npv > 0 else 1.0
        rows.append({
            "config_id":        i + 1,
            "n_tables":         n_tables,
            "n_edges":          n_edges,
            "budget_frac":      budget_frac,
            "optimal_npv_gb":   round(opt_npv / 1e9, 3),
            "greedy_npv_gb":    round(greedy_npv_val / 1e9, 3),
            "approx_ratio":     round(ratio, 4),
            "theoretical_bound": round(THEORETICAL_BOUND, 4),
            "above_bound":      ratio >= THEORETICAL_BOUND,
        })
        console.print(f"  Config {i+1} ({n_tables}T/{n_edges}E): "
                      f"optimal={opt_npv/1e9:.2f}GB, "
                      f"greedy={greedy_npv_val/1e9:.2f}GB, "
                      f"ratio={ratio:.3f} "
                      f"({'≥' if ratio >= THEORETICAL_BOUND else '<'} {THEORETICAL_BOUND:.3f})")

    csv_path = RESULTS_DIR / "rq5.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"  CSV → {_link(csv_path)}")

    _plot_approx_ratios(rows)
    _plot_npv_comparison(rows)

    return rows


def _plot_approx_ratios(rows):
    labels = [f"C{r['config_id']}\n({r['n_tables']}T/{r['n_edges']}E)" for r in rows]
    ratios = [r["approx_ratio"] for r in rows]
    colors = ["#2ca02c" if r["above_bound"] else "#d62728" for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, ratios, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(y=THEORETICAL_BOUND, color="navy", linestyle="--", linewidth=1.5,
               label=f"Sviridenko bound (1−1/e ≈ {THEORETICAL_BOUND:.3f})")
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Approximation Ratio (greedy / optimal)")
    ax.set_title("RQ4 — Greedy Approximation Quality vs. Brute-Force Optimal\n"
                 "Green = above theoretical bound; Red = below bound")
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "rq5_approx_ratio.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq5_approx_ratio.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {_link(out)}")


def _plot_npv_comparison(rows):
    labels = [f"C{r['config_id']}" for r in rows]
    opt    = [r["optimal_npv_gb"] for r in rows]
    greedy = [r["greedy_npv_gb"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, opt,    width, label="Optimal NPV",      color="#1f77b4",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, greedy, width, label="Greedy NPV (AJLO)", color="#2ca02c",
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NPV (GB-equivalent)")
    ax.set_title("RQ4 — Greedy vs. Optimal NPV Across Configurations")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "rq5_npv_comparison.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq5_npv_comparison.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {_link(out)}")


if __name__ == "__main__":
    run()
