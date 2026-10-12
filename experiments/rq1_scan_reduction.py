"""RQ1: Does AJLO reduce bytes scanned per query compared to per-table baselines?

Pipeline:
  1. Generate TPC-H query log
  2. Algorithm 1 → extract join graph
  3. Algorithm 2 → compute JIS scores
  4. Algorithm 3 → recommend coordinated layout plan
  5. Run ALL_EXPERIMENT_QUERIES against B0 / B2 / AJLO
  6. Measure bytes_read using Delta min/max file statistics (real data skipping)
  7. Show per-query winner: ALC vs AJLO tradeoff

bytes_read = sum of Parquet file sizes for files that PASS Delta's min/max
             predicate check — the actual bytes Delta must read from storage.
             This is the real data-skipping measurement.

Output:
  results/rq1.csv
  results/rq1_bytes_scanned.png         ← mean bytes by layout, by query category
  results/rq1_query_latency.png
  results/rq1_jis_ranked_edges.png      ← JIS ranking that drove AJLO's decisions
  results/rq1_tradeoff.png              ← per-query: B2 vs AJLO, winner highlighted
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

sns.set_theme(style="whitegrid", palette="tab10")
from rich.progress import track

from ajlo.algorithms.join_graph import extract_join_graph
from ajlo.algorithms.jis import compute_jis
from ajlo.algorithms.layout_planner import recommend_layout
from ajlo.models import LayoutPlan, JoinGraph
from experiments.spark_utils import get_spark_session, register_views, time_query
from experiments.query_logs import generate_query_log, get_cardinalities
from experiments.delta_stats import bytes_read_with_predicate, total_table_bytes
from experiments.experiment_queries import ALL_EXPERIMENT_QUERIES, get_experiment_queries

console = Console()

from experiments.config import RESULTS_DIR, LAYOUTS_DIR

LAYOUT_NAMES  = ["b0", "b2", "b3", "ajlo"]
LAYOUT_LABELS = ["B0\n(No opt)", "B2\n(ALC — per-table\nfilter cols)",
                 "B3\n(per-table\nindependent opt)", "AJLO\n(join-aware)"]
LAYOUT_COLORS = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c"]




# ── AJLO library pipeline ─────────────────────────────────────────────────────

def run_ajlo_pipeline(sf: int = 1) -> tuple[JoinGraph, LayoutPlan]:
    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    console.print("\n  [AJLO] Step 1 — Generating query log (500 queries, regime C)...")
    start  = datetime(2024, 1, 1)
    window = (start, start + timedelta(days=7))
    log    = generate_query_log("C", n_queries=500, start_time=start, seed=0, sf=sf)

    console.print("  [AJLO] Step 2 — Algorithm 1: extracting join graph...")
    graph = extract_join_graph(log, window)
    console.print(f"          → {len(graph.edges)} join relationships, "
                  f"{len(graph.tables())} tables")

    console.print("  [AJLO] Step 3 — Algorithm 2: computing JIS scores...")
    graph = compute_jis(graph)
    console.print("          Top-5 edges by JIS:")
    for e in sorted(graph.edges, key=lambda e: e.jis, reverse=True)[:5]:
        console.print(f"          JIS={e.jis:.4f}  {e.table_i}.{e.col_i} ↔ "
                      f"{e.table_j}.{e.col_j}  (freq={e.frequency})")

    console.print("  [AJLO] Step 4 — Algorithm 3: recommending coordinated layout...")
    plan = recommend_layout(
        graph, LayoutPlan(),
        rewrite_budget=sum(table_sizes.values()),
        storage_budget=sum(table_sizes.values()) * 0.05,
        table_sizes=table_sizes,
    )
    console.print("          Layout recommended by Algorithm 3:")
    for table in sorted(graph.tables()):
        strat = plan.get(table)
        col   = (strat.cluster_cols[0] if strat.cluster_cols
                 else strat.partition_col or "—")
        console.print(f"          {table:12s}: {strat.strategy_type:14s} → {col}")
    return graph, plan


# ── Bytes read via Delta statistics ──────────────────────────────────────────

def _measure_bytes(layout_path: Path, filter_spec: dict) -> int:
    """Return total bytes that must be read, using Delta min/max statistics.

    filter_spec: {table_name: (col, pred_min, pred_max)}
    Only tables that appear in filter_spec are counted.
    """
    total = 0
    for table_path in layout_path.iterdir():
        if not table_path.is_dir():
            continue
        t = table_path.name
        if t in filter_spec:
            col, pmin, pmax = filter_spec[t]
            b, _, _ = bytes_read_with_predicate(table_path, col, pmin, pmax)
            total += b
    return total


# ── Main run ──────────────────────────────────────────────────────────────────

def run(spark=None, sf: int = 1) -> list[dict]:
    console.print(f"\n=== RQ1: Bytes Scanned per Query (SF={sf}) ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    graph, plan = run_ajlo_pipeline(sf=sf)

    if spark is None:
        spark = get_spark_session("rq1-scan-reduction")

    all_rows = []
    run_rows = []   # one row per timed execution (for variance reporting)

    for layout in LAYOUT_NAMES:
        layout_path = LAYOUTS_DIR / layout
        if not layout_path.exists():
            console.print(f"  [SKIP] {layout}: layout not found")
            continue

        register_views(spark, layout_path)
        console.print(f"\n  Layout: {layout.upper()}")

        for (qid, sql, filter_spec) in track(
            get_experiment_queries(sf), description=f"{layout} queries"
        ):
            bytes_read = _measure_bytes(layout_path, filter_spec)
            times = time_query(spark, sql)
            std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

            all_rows.append({
                "layout":              layout,
                "query_id":            qid,
                "bytes_read":          bytes_read,
                "wall_time_ms":        round(float(np.mean(times)), 1),
                "wall_time_std_ms":    round(std, 1),
                "wall_time_median_ms": round(float(np.median(times)), 1),
                "n_runs":              len(times),
            })
            run_rows += [{"layout": layout, "query_id": qid, "run": i + 1,
                          "wall_time_ms": round(t, 1)} for i, t in enumerate(times)]
            console.print(
                f"    {qid}: {bytes_read/1e6:.1f} MB  {np.mean(times):.0f} ± {std:.0f} ms "
                f"(n={len(times)})"
            )

    if not all_rows:
        console.print("  No results. Run: python experiments/run_all.py --only layouts")
        return []

    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "rq1.csv"
    df.to_csv(csv_path, index=False)
    pd.DataFrame(run_rows).to_csv(RESULTS_DIR / "rq1_runs.csv", index=False)
    console.print(f"\n  CSV → {csv_path} (+ rq1_runs.csv: every timed run)")

    _plot_bytes_by_layout(all_rows)
    _plot_latency_bar(all_rows)
    _plot_tradeoff(all_rows)
    _plot_jis_ranking(graph)

    return all_rows


# ── Charts ────────────────────────────────────────────────────────────────────

def _mean_by(rows, group_key, metric):
    totals: dict[str, list] = {}
    for r in rows:
        totals.setdefault(r[group_key], []).append(r[metric])
    return {k: float(np.mean(v)) for k, v in totals.items()}


def _rows_to_records(rows: list[dict], layouts: list[str], metric: str,
                     scale: float = 1.0) -> list[dict]:
    label_map = dict(zip(LAYOUT_NAMES, LAYOUT_LABELS))
    return [
        {"Layout": label_map[r["layout"]], "value": r[metric] * scale}
        for r in rows if r["layout"] in layouts
    ]


def _plot_bytes_by_layout(rows: list[dict]) -> None:
    """Bar chart: mean bytes read per layout across all queries."""
    records = _rows_to_records(rows, LAYOUT_NAMES, "bytes_read", scale=1e-6)
    df = pd.DataFrame(records)
    means = df.groupby("Layout", sort=False)["value"].mean().reset_index()
    label_order = [LAYOUT_LABELS[LAYOUT_NAMES.index(n)]
                   for n in LAYOUT_NAMES if LAYOUT_LABELS[LAYOUT_NAMES.index(n)] in means["Layout"].values]
    palette = dict(zip(LAYOUT_LABELS, LAYOUT_COLORS))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=means, x="Layout", y="value", order=label_order,
                palette=palette, edgecolor="black", linewidth=0.5, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f MB", padding=2, fontsize=9)

    ax.set_ylabel("Mean Bytes Read (MB)\n[from Delta min/max statistics — real skipping]")
    ax.set_xlabel("")
    ax.set_title("RQ1 — Mean Bytes Scanned per Query by Layout\n(lower is better)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    sns.despine()
    plt.tight_layout()
    out = RESULTS_DIR / "rq1_bytes_scanned.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq1_bytes_scanned.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {out}")


def _plot_latency_bar(rows: list[dict]) -> None:
    """Mean per-query latency per layout; error bar = pooled run-to-run std."""
    df = pd.DataFrame(rows)
    present = [n for n in LAYOUT_NAMES if n in set(df["layout"])]
    means, errs = [], []
    for n in present:
        sub = df[df["layout"] == n]
        means.append(sub["wall_time_ms"].mean() / 1e3)
        errs.append(np.sqrt((sub["wall_time_std_ms"] ** 2).mean()) / 1e3)
    labels = [LAYOUT_LABELS[LAYOUT_NAMES.index(n)] for n in present]
    colors = [LAYOUT_COLORS[LAYOUT_NAMES.index(n)] for n in present]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=errs, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{m:.2f} ± {e:.2f} s" for m, e in zip(means, errs)],
                 padding=3, fontsize=9)

    n_runs = int(df["n_runs"].max()) if "n_runs" in df else 1
    ax.set_ylabel(f"Mean Query Latency (seconds)\n[error bar: pooled std over {n_runs} runs]")
    ax.set_xlabel("")
    ax.set_title("RQ1 — Mean Query Latency by Layout\n(lower is better)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    sns.despine()
    plt.tight_layout()
    out = RESULTS_DIR / "rq1_query_latency.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq1_query_latency.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {out}")


def _plot_tradeoff(rows: list[dict]) -> None:
    """Per-query scatter: B2 bytes vs AJLO bytes. Points below diagonal → AJLO wins."""

    from matplotlib.patches import Patch

    b2_bytes   = {r["query_id"]: r["bytes_read"] for r in rows if r["layout"] == "b2"}
    ajlo_bytes = {r["query_id"]: r["bytes_read"] for r in rows if r["layout"] == "ajlo"}

    qids = [q for q in b2_bytes if q in ajlo_bytes]
    if not qids:
        return

    df = pd.DataFrame({
        "b2":       [b2_bytes[q] / 1e6   for q in qids],
        "ajlo":     [ajlo_bytes[q] / 1e6 for q in qids],
        "query_id": qids,
    })

    max_val = max(df["b2"].max(), df["ajlo"].max(), 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=df, x="b2", y="ajlo",
                    color="#4c72b0", s=80, edgecolor="black", linewidth=0.5,
                    zorder=3, ax=ax)

    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.4)
    ax.fill_between([0, max_val], [0, 0], [0, max_val], alpha=0.05, color="#2ca02c")
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], alpha=0.05, color="#ff7f0e")

    for _, row in df.iterrows():
        ax.annotate(row["query_id"], (row["b2"], row["ajlo"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    legend_patches = [
        Patch(color="#2ca02c", alpha=0.2, label="AJLO wins (below diagonal)"),
        Patch(color="#ff7f0e", alpha=0.2, label="B2 wins (above diagonal)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)

    ax.set_xlabel("B2 (ALC) bytes read (MB)")
    ax.set_ylabel("AJLO bytes read (MB)")
    ax.set_title("RQ1 — B2 vs AJLO: Per-Query Bytes Scanned\n"
                 "Below diagonal = AJLO scans less; Above diagonal = B2 scans less")
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    sns.despine()
    plt.tight_layout()
    out = RESULTS_DIR / "rq1_tradeoff.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq1_tradeoff.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {out}")


def _plot_jis_ranking(graph: JoinGraph) -> None:

    ranked = sorted(graph.edges, key=lambda e: e.jis, reverse=True)[:10]
    df = pd.DataFrame({
        "edge":  [f"{e.table_i[:3]}.{e.col_i[:8]}\n↔\n{e.table_j[:3]}.{e.col_j[:8]}"
                  for e in ranked],
        "jis":   [e.jis for e in ranked],
        "top3":  [i < 3 for i in range(len(ranked))],
    })
    palette = {True: "#2ca02c", False: "#aec6cf"}

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="edge", y="jis", hue="top3",
                palette=palette, edgecolor="black", linewidth=0.5,
                dodge=False, legend=False, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)

    ax.set_ylabel("JIS Score (Algorithm 2)")
    ax.set_xlabel("")
    ax.set_title("RQ1 — JIS-Ranked Join Edges (Algorithm 2 output)\n"
                 "Green = top-3 edges AJLO co-clustered; Grey = lower priority")
    ax.set_ylim(0, 1.05)
    sns.despine()
    plt.tight_layout()
    out = RESULTS_DIR / "rq1_jis_ranked_edges.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    out = RESULTS_DIR/ "eps" / "rq1_jis_ranked_edges.eps"
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {out}")


if __name__ == "__main__":
    run()
