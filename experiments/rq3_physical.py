"""RQ3 (physical): execute AJLO's drift-triggered re-clustering on real Delta tables.

rq3_drift.py shows *when* AJLO decides to reorganise and what the cost model
expects. This step executes exactly those decisions on Delta tables and
measures what they actually cost and gain:

  1. Build every table with the layout in force before the structural drift
     (plan after W4 — regime A/B keys).
  2. Benchmark the regime-D queries + two regime-A queries ("before").
  3. For each reorg op AJLO emits at the first drift window with ops (W5):
       ALTER TABLE delta.`t` CLUSTER BY (new_key);  OPTIMIZE delta.`t` FULL
     recording wall time and bytes actually rewritten (from the Delta log).
  4. Benchmark again ("after").

Output:
  results/rq3_physical.csv     per (phase, query): bytes, files, skip %, latency mean/std
  results/rq3_reorg_ops.csv    per op: table, from→to key, rewrite seconds, bytes rewritten,
                               cost-model predicted rewrite bytes
  results/rq3_physical.png     bytes scanned per query, before vs after
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.markup import escape

from ajlo.cost_model import rewrite_cost
from experiments.baselines import TABLES, _load_raw, _write_clustered, _write_delta
from experiments.config import LAYOUTS_DIR, RESULTS_DIR, spark_path
from experiments.delta_stats import bytes_read_with_predicate
from experiments.experiment_queries import get_rq3_queries
from experiments.query_logs import get_cardinalities
from experiments.rq3_drift import _key_of, simulate_drift
from experiments.spark_utils import get_spark_session, register_views, time_query

console = Console()

RQ3_LAYOUT = "rq3"


# ── Delta log helpers ─────────────────────────────────────────────────────────

def _latest_version(table_path: Path) -> int:
    log_dir = table_path / "_delta_log"
    versions = [int(p.stem) for p in log_dir.glob("*.json") if p.stem.isdigit()]
    return max(versions) if versions else -1


def _bytes_added_after(table_path: Path, version: int) -> int:
    """Total size of data files added by commits after `version` (bytes rewritten)."""
    total = 0
    for log_file in (table_path / "_delta_log").glob("*.json"):
        if not log_file.stem.isdigit() or int(log_file.stem) <= version:
            continue
        for line in log_file.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                if "add" in entry:
                    total += int(entry["add"].get("size", 0))
    return total


# ── Steps ─────────────────────────────────────────────────────────────────────

def _build_initial(spark, out: Path, plan) -> None:
    console.print(f"  Building pre-drift layout → {out}")
    for table in TABLES:
        strat = plan.get(table)
        df = _load_raw(spark, table)
        if strat.strategy_type == "NOOP":
            _write_delta(spark, df, out / table)
            console.print(f"    {table}: no clustering")
        else:
            _write_clustered(spark, df, out / table, cluster_col=_key_of(strat))
            console.print(f"    {table}: CLUSTER BY {_key_of(strat)}")


def _recluster(spark, table_path: Path, new_key: str) -> str:
    """Change the clustering key in place and rewrite all files. Returns method used."""
    p = spark_path(table_path)
    try:
        spark.sql(f"ALTER TABLE delta.`{p}` CLUSTER BY ({new_key})")
        spark.sql(f"OPTIMIZE delta.`{p}` FULL")
        return "ALTER CLUSTER BY + OPTIMIZE FULL"
    except Exception as exc:  # older Delta without OPTIMIZE FULL / ALTER CLUSTER BY
        console.print(f"    [yellow]ALTER/OPTIMIZE FULL unavailable ({escape(str(exc)[:120])}); "
                      f"rewriting with CTAS[/yellow]")
        df = spark.read.format("delta").load(p).localCheckpoint()
        _write_clustered(spark, df, table_path, cluster_col=new_key)
        return "CTAS CLUSTER BY + OPTIMIZE"


def _benchmark(spark, layout_path: Path, queries: list, phase: str) -> list[dict]:
    register_views(spark, layout_path)
    rows = []
    for qid, sql, fspec in queries:
        bytes_read = files_read = files_total = 0
        for table, (col, lo, hi) in fspec.items():
            b, fr, ft = bytes_read_with_predicate(layout_path / table, col, lo, hi)
            bytes_read += b
            files_read += fr
            files_total += ft
        times = time_query(spark, sql)
        skip = (files_total - files_read) / files_total * 100.0 if files_total else 0.0
        rows.append({
            "phase": phase, "query_id": qid,
            "bytes_read": bytes_read, "files_read": files_read, "files_total": files_total,
            "skip_pct": round(skip, 1),
            "wall_time_ms": round(float(np.mean(times)), 1),
            "wall_time_std_ms": round(float(np.std(times, ddof=1)) if len(times) > 1 else 0.0, 1),
            "n_runs": len(times),
        })
        console.print(f"    {phase:6s} {qid}: {bytes_read/1e6:9.1f} MB  skip={skip:5.1f}%  "
                      f"{np.mean(times):7.0f} ± {rows[-1]['wall_time_std_ms']:.0f} ms")
    return rows


def run(spark=None, sf: int = 1) -> list[dict]:
    console.print(f"\n=== RQ3 (physical): executing drift-triggered re-clustering (SF={sf}) ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if spark is None:
        spark = get_spark_session("rq3-physical")

    steps = simulate_drift(sf)
    drift_idx = next((i for i, s in enumerate(steps) if s["ops"] and not s["initial"]), None)
    if drift_idx is None:
        console.print("  No drift window produced reorg ops — nothing to execute.")
        return []
    before_plan = steps[drift_idx - 1]["plan"]
    drift = steps[drift_idx]
    console.print(f"  Drift window {drift['window']} (regime {drift['regime']}): "
                  f"{len(drift['ops'])} ops")

    out = LAYOUTS_DIR / RQ3_LAYOUT
    _build_initial(spark, out, before_plan)

    queries = get_rq3_queries(sf)
    rows = _benchmark(spark, out, queries, "before")

    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    op_rows = []
    for op in drift["ops"]:
        path = out / op.table
        old_key, new_key = _key_of(op.from_strategy), _key_of(op.to_strategy)
        v0 = _latest_version(path)
        t0 = time.perf_counter()
        method = _recluster(spark, path, new_key)
        secs = time.perf_counter() - t0
        rewritten = _bytes_added_after(path, v0)
        predicted = rewrite_cost(table_sizes.get(op.table, 0.0),
                                 op.from_strategy.strategy_type, op.to_strategy.strategy_type)
        op_rows.append({
            "window": drift["window"], "table": op.table,
            "from_key": old_key, "to_key": new_key, "method": method,
            "rewrite_seconds": round(secs, 1),
            "bytes_rewritten": rewritten,
            "predicted_rewrite_bytes": int(predicted),
        })
        console.print(f"    reorg {op.table}: {old_key} → {new_key}  {secs:.1f}s  "
                      f"{rewritten/1e9:.2f} GB rewritten (model: {predicted/1e9:.2f} GB)")

    rows += _benchmark(spark, out, queries, "after")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "rq3_physical.csv", index=False)
    pd.DataFrame(op_rows).to_csv(RESULTS_DIR / "rq3_reorg_ops.csv", index=False)
    console.print(f"  CSV → {RESULTS_DIR / 'rq3_physical.csv'}, {RESULTS_DIR / 'rq3_reorg_ops.csv'}")
    _plot(rows, op_rows)
    return rows


def _plot(rows: list[dict], op_rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    qids = list(dict.fromkeys(df["query_id"]))
    before = df[df.phase == "before"].set_index("query_id").loc[qids]
    after = df[df.phase == "after"].set_index("query_id").loc[qids]
    x = np.arange(len(qids))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, before["bytes_read"] / 1e9, width, label="Before (pre-drift layout)",
           color="#9e9e9e", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, after["bytes_read"] / 1e9, width, label="After AJLO re-clustering",
           color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n({'regime D' if q.startswith('D') else 'regime A'})" for q in qids])
    ax.set_ylabel("Bytes Read (GB)\n[Delta min/max statistics]")
    total_gb = sum(r["bytes_rewritten"] for r in op_rows) / 1e9
    ops_txt = ", ".join(f"{r['table']}→{r['to_key']}" for r in op_rows)
    ax.set_title("RQ3 — Physical Re-clustering After Structural Drift\n"
                 f"({ops_txt}; {total_gb:.1f} GB rewritten)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "rq3_physical.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    plt.savefig(RESULTS_DIR / "eps" / "rq3_physical.eps", format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {out}")


if __name__ == "__main__":
    run()
