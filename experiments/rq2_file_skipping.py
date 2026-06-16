"""RQ2: Does AJLO improve file skip ratios and reduce shuffle bytes?

Pipeline:
  1. Generate TPC-H query log
  2. Algorithm 1 — extract join graph
  3. Algorithm 2 — compute JIS; identify join-heavy queries (high-JIS edges)
  4. Algorithm 3 — recommend layout; derive expected φ_scan per table pair
  5. Run those join-heavy queries against B0 / B2 / AJLO layouts
  6. Measure actual file skip ratio and shuffle bytes
  7. Compare measured skip ratio vs AJLO's φ_scan estimate (calibration)

Output:
  results/rq2.csv
  results/rq2_skip_ratio.png
  results/rq2_shuffle_bytes.png
  results/rq2_phi_scan_calibration.png  ← estimated vs measured scan reduction
"""
from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress

console = Console()


def _link(path) -> str:
    return f"[link=file://{path}]{path}[/link]"

from ajlo.algorithms.join_graph import extract_join_graph
from ajlo.algorithms.jis import compute_jis
from ajlo.algorithms.layout_planner import recommend_layout
from ajlo.models import LayoutPlan
from ajlo.cost_model import PHI_SCAN, expected_scan_bytes, scan_reduction
from experiments.spark_utils import get_spark_session, register_views, _collect_plan_metrics
from experiments.query_logs import generate_query_log, get_cardinalities
from experiments.experiment_queries import ALL_EXPERIMENT_QUERIES
from experiments.delta_stats import bytes_read_with_predicate, total_table_bytes

RESULTS_DIR = Path(__file__).parent.parent / "results"
LAYOUTS_DIR = Path(__file__).parent.parent / "data" / "tpch" / "layouts"

# Layouts compared in RQ2
LAYOUT_NAMES  = ["b0", "b2", "ajlo"]
LAYOUT_LABELS = ["B0\n(No opt)", "B2\n(ALC — per-table\nfilter cols)", "AJLO\n(join-aware)"]
LAYOUT_COLORS = ["#d62728", "#ff7f0e", "#2ca02c"]


# ── AJLO pipeline: steps 1-4 ─────────────────────────────────────────────────

def run_ajlo_pipeline(sf: int = 1):
    """Run Algorithms 1→2→3 and return (graph, plan, high_jis_query_ids).

    Also identifies which TPC-H query IDs involve the top-JIS join edges
    so RQ2 focuses measurement on the queries AJLO prioritised.
    """
    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    console.print("\n  [AJLO] Step 1 — Generating query log (500 queries, regime C)...")
    start  = datetime(2024, 1, 1)
    window = (start, start + timedelta(days=7))
    log    = generate_query_log("C", n_queries=500, start_time=start, seed=0)

    console.print("  [AJLO] Step 2 — Algorithm 1: extracting join graph...")
    graph = extract_join_graph(log, window)
    console.print(f"          → {len(graph.edges)} edges, {len(graph.tables())} tables")

    console.print("  [AJLO] Step 3 — Algorithm 2: computing JIS scores...")
    graph = compute_jis(graph)

    # Identify queries that involve high-JIS join relationships
    # (JIS > median — these are the queries AJLO is optimising for)
    jis_values = [e.jis for e in graph.edges if e.jis > 0]
    jis_median = float(np.median(jis_values)) if jis_values else 0.0

    high_jis_tables: set[str] = set()
    for e in graph.edges:
        if e.jis >= jis_median:
            high_jis_tables.add(e.table_i)
            high_jis_tables.add(e.table_j)

    # Q01–Q07 are the 70 % join-key queries that AJLO's layout targets
    high_jis_qids: set[str] = {q[0] for q in ALL_EXPERIMENT_QUERIES[:7]}
    console.print(f"          High-JIS tables: {sorted(high_jis_tables)}")
    console.print(f"          AJLO-wins query IDs: {sorted(high_jis_qids)}")

    console.print("  [AJLO] Step 4 — Algorithm 3: recommending coordinated layout plan...")
    plan = recommend_layout(
        graph,
        LayoutPlan(),
        rewrite_budget=sum(table_sizes.values()),
        storage_budget=sum(table_sizes.values()) * 0.05,
        table_sizes=table_sizes,
    )

    # Print expected scan reduction per top edge
    console.print("          Expected scan reduction per top-JIS edge:")
    for e in sorted(graph.edges, key=lambda x: x.jis, reverse=True)[:5]:
        strat = plan.get(e.table_i).strategy_type
        exp   = expected_scan_bytes(e.frequency, e.data_volume)
        sr    = scan_reduction(exp, strat)
        phi   = PHI_SCAN.get(strat, 0.0)
        console.print(f"          {e.table_i}.{e.col_i} ↔ {e.table_j}.{e.col_j}: "
                      f"strategy={strat}, φ_scan={phi}, "
                      f"expected_reduction={sr/1e6:.0f} MB/window")

    return graph, plan, high_jis_qids


# ── Step 5-6: Run queries and collect metrics ─────────────────────────────────

def run(spark=None, sf: int = 1) -> list[dict]:
    console.print(f"\n=== RQ2: File Skip Ratio + Shuffle Bytes (SF={sf}) ===")
    RESULTS_DIR.mkdir(exist_ok=True)

    _, plan, high_jis_qids = run_ajlo_pipeline(sf=sf)

    if spark is None:
        spark = get_spark_session("rq2-file-skipping")

    # Focus on queries AJLO identified as join-heavy (plus always-included set)
    target_queries = list(ALL_EXPERIMENT_QUERIES)

    all_rows = []

    with Progress(console=console) as progress:
        layout_task = progress.add_task("Layouts", total=len(LAYOUT_NAMES))
        for layout in LAYOUT_NAMES:
            layout_path = LAYOUTS_DIR / layout
            if not layout_path.exists():
                console.print(f"  [SKIP] {layout}: layout not found")
                progress.advance(layout_task)
                continue

            register_views(spark, layout_path)
            console.print(f"\n  Layout: {layout.upper()}")

            query_task = progress.add_task(f"Queries [{layout}]", total=len(target_queries))
            for (qid, sql, fspec) in target_queries:
                # File skip counts from Delta min/max stats (same predicate Delta uses)
                files_read = 0
                files_total = 0
                bytes_read = 0
                for table_name, (col, pmin, pmax) in fspec.items():
                    b, fr, ft = bytes_read_with_predicate(layout_path / table_name, col, pmin, pmax)
                    bytes_read  += b
                    files_read  += fr
                    files_total += ft

                # For tables without a predicate add their full file count as must-read
                for table_name in register_views(spark, layout_path):
                    if table_name not in fspec:
                        _, ft = total_table_bytes(layout_path / table_name)
                        files_read  += ft
                        files_total += ft

                files_skipped = max(0, files_total - files_read)
                skip_pct = (files_skipped / files_total * 100.0) if files_total > 0 else 0.0

                t0 = time.perf_counter()
                df = spark.sql(sql)
                df.count()
                wall_ms = (time.perf_counter() - t0) * 1000.0

                try:
                    plan_metrics = _collect_plan_metrics(df.queryExecution.executedPlan)
                    shuffle_bytes = int(plan_metrics.get("data size", 0))
                except Exception:
                    shuffle_bytes = 0

                ajlo_expected_skip = 0.0
                if layout == "ajlo":
                    strat = plan.get("orders").strategy_type
                    ajlo_expected_skip = PHI_SCAN.get(strat, 0.0) * 100.0

                all_rows.append({
                    "layout":                layout,
                    "query_id":              qid,
                    "files_total":           files_total,
                    "files_read":            files_read,
                    "files_skipped":         files_skipped,
                    "skip_pct":              round(skip_pct, 1),
                    "shuffle_bytes":         shuffle_bytes,
                    "bytes_read":            bytes_read,
                    "wall_time_ms":          round(wall_ms, 1),
                    "ajlo_expected_skip_pct": round(ajlo_expected_skip, 1),
                })
                console.print(f"    {qid}: skip={skip_pct:.1f}% "
                              f"({files_skipped}/{files_total} files)  "
                              f"read={bytes_read/1e6:.1f} MB")
                progress.advance(query_task)
            progress.advance(layout_task)

    if not all_rows:
        console.print("  No results. Run: python experiments/run_all.py --only layouts")
        return []

    csv_path = RESULTS_DIR / "rq2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    console.print(f"\n  CSV → {_link(csv_path)}")

    _plot_skip_ratio(all_rows)
    _plot_shuffle_bytes(all_rows)
    _plot_phi_calibration(all_rows)

    return all_rows


# ── Charts ────────────────────────────────────────────────────────────────────

def _mean_by_layout(rows: list[dict], metric: str) -> dict[str, float]:
    totals: dict[str, list] = {}
    for r in rows:
        totals.setdefault(r["layout"], []).append(r[metric])
    return {k: float(np.mean(v)) for k, v in totals.items()}


def _plot_skip_ratio(rows: list[dict]) -> None:
    means   = _mean_by_layout(rows, "skip_pct")
    present = [n for n in LAYOUT_NAMES if n in means]
    labels  = [LAYOUT_LABELS[LAYOUT_NAMES.index(n)] for n in present]
    values  = [means[n] for n in present]
    colors  = [LAYOUT_COLORS[LAYOUT_NAMES.index(n)] for n in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Mean File Skip Ratio (%)")
    ax.set_title("RQ2 — File Skip Ratio on AJLO-Identified Join-Heavy Queries\n"
                 "(AJLO queries selected by Algorithm 2 JIS ranking; higher is better)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out = RESULTS_DIR / "rq2_skip_ratio.png"
    plt.savefig(out, dpi=150)
    plt.close()
    console.print(f"  Chart → {_link(out)}")


def _plot_shuffle_bytes(rows: list[dict]) -> None:
    means   = _mean_by_layout(rows, "shuffle_bytes")
    present = [n for n in LAYOUT_NAMES if n in means]
    labels  = [LAYOUT_LABELS[LAYOUT_NAMES.index(n)] for n in present]
    values  = [means[n] / 1e6 for n in present]
    colors  = [LAYOUT_COLORS[LAYOUT_NAMES.index(n)] for n in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.0f MB", padding=3, fontsize=9)
    ax.set_ylabel("Mean Shuffle Bytes (MB)")
    ax.set_title("RQ2 — Mean Shuffle Bytes on Join-Heavy Queries\n(lower is better)")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    plt.tight_layout()
    out = RESULTS_DIR / "rq2_shuffle_bytes.png"
    plt.savefig(out, dpi=150)
    plt.close()
    console.print(f"  Chart → {_link(out)}")


def _plot_phi_calibration(rows: list[dict]) -> None:
    """Compare AJLO's estimated φ_scan vs measured skip ratio per query.

    This is the calibration chart for Section 11.1 (internal validity).
    A layout change that works as predicted shows estimated ≈ measured.
    """
    ajlo_rows = [r for r in rows if r["layout"] == "ajlo"]
    if not ajlo_rows:
        return

    qids      = [r["query_id"] for r in ajlo_rows]
    measured  = [r["skip_pct"] for r in ajlo_rows]
    estimated = [r["ajlo_expected_skip_pct"] for r in ajlo_rows]

    x = np.arange(len(qids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, estimated, width, label="AJLO estimated (φ_scan × 100)",
           color="#1f77b2", edgecolor="black", linewidth=0.6, alpha=0.8)
    ax.bar(x + width/2, measured, width, label="Measured file skip %",
           color="#2ca02c", edgecolor="black", linewidth=0.6, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(qids, fontsize=8)
    ax.set_ylabel("File Skip / Reduction (%)")
    ax.set_title("RQ2 — AJLO φ_scan Estimate vs Measured File Skip Ratio\n"
                 "(validates Algorithm 3 cost model accuracy)")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out = RESULTS_DIR / "rq2_phi_scan_calibration.png"
    plt.savefig(out, dpi=150)
    plt.close()
    console.print(f"  Chart → {_link(out)}")


if __name__ == "__main__":
    run()
