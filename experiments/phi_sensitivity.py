"""φ_scan sensitivity: do AJLO's layout decisions depend on the assumed skip fraction?

The cost model assumes φ_scan = 0.30 for co-clustering (0.20 for co-partitioning).
Measured skipping at scale differs (see rq2.csv). This sweep re-runs
Algorithms 1→2→3 for φ_CO_CLUSTER ∈ {0.1 … 0.8} (φ_CO_PARTITION scaled by the
same ratio) on every workload regime and records each table's chosen layout.

A table is "stable" at φ if its (strategy, key) equals the decision at the
default φ = 0.30.

Output:
  results/phi_sensitivity.csv   regime, phi, table, strategy, key, same_as_default
  results/phi_sensitivity.png   tables changed vs default, per regime × φ
"""
from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from ajlo import cost_model
from ajlo.algorithms.jis import compute_jis
from ajlo.algorithms.join_graph import extract_join_graph
from ajlo.algorithms.layout_planner import recommend_layout
from ajlo.models import LayoutPlan
from experiments.config import RESULTS_DIR
from experiments.query_logs import generate_query_log, get_cardinalities

console = Console()

PHI_VALUES = [round(0.1 * i, 1) for i in range(1, 9)]
DEFAULT_PHI = cost_model.PHI_SCAN["CO_CLUSTER"]
REGIMES = ["A", "B", "C", "D"]


def _plan_for(graph, table_sizes: dict, phi: float) -> LayoutPlan:
    """Run Algorithm 3 with φ temporarily overridden (same budgets as the AJLO layout)."""
    original = dict(cost_model.PHI_SCAN)
    ratio = original["CO_PARTITION"] / original["CO_CLUSTER"]
    cost_model.PHI_SCAN.update({"CO_CLUSTER": phi, "CO_PARTITION": phi * ratio})
    try:
        total = sum(table_sizes.values())
        return recommend_layout(graph, LayoutPlan(), rewrite_budget=total,
                                storage_budget=total * 0.05, table_sizes=table_sizes)
    finally:
        cost_model.PHI_SCAN.clear()
        cost_model.PHI_SCAN.update(original)


def _decision(plan: LayoutPlan, table: str) -> tuple[str, str]:
    s = plan.get(table)
    key = s.cluster_cols[0] if s.cluster_cols else (s.partition_col or "—")
    return s.strategy_type, key


def _measured_phi() -> float | None:
    """Mean AJLO file-skip fraction on the join-key queries, if rq2.csv exists."""
    path = RESULTS_DIR / "rq2.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    sel = df[(df["layout"] == "ajlo") & (df["query_id"].isin([f"Q0{i}" for i in range(1, 8)]))]
    return float(sel["skip_pct"].mean()) / 100.0 if len(sel) else None


def run(sf: int = 1) -> list[dict]:
    console.print(f"\n=== φ_scan sensitivity (SF={sf}) ===")
    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    start = datetime(2024, 1, 1)
    window = (start, start + timedelta(days=7))

    rows = []
    for regime in REGIMES:
        log = generate_query_log(regime, n_queries=500, start_time=start, seed=0, sf=sf)
        graph = compute_jis(extract_join_graph(log, window))
        default = _plan_for(graph, table_sizes, DEFAULT_PHI)
        tables = sorted(graph.tables())
        for phi in PHI_VALUES:
            plan = _plan_for(graph, table_sizes, phi)
            changed = 0
            for t in tables:
                stype, key = _decision(plan, t)
                same = (stype, key) == _decision(default, t)
                changed += not same
                rows.append({"regime": regime, "phi": phi, "table": t,
                             "strategy": stype, "key": key, "same_as_default": same})
            console.print(f"  regime {regime}  φ={phi:.1f}: {changed}/{len(tables)} tables "
                          f"differ from φ={DEFAULT_PHI}")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "phi_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"  CSV → {csv_path}")
    _plot(df)
    return rows


def _plot(df: pd.DataFrame) -> None:
    changed = (~df["same_as_default"]).groupby([df["regime"], df["phi"]]).sum().unstack("phi")
    n_tables = df.groupby("regime")["table"].nunique()

    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(changed.values, cmap="Reds", vmin=0, vmax=max(1, int(n_tables.max())),
                   aspect="auto")
    for i, regime in enumerate(changed.index):
        for j, phi in enumerate(changed.columns):
            ax.text(j, i, int(changed.loc[regime, phi]), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(changed.columns)))
    ax.set_xticklabels([f"{p:.1f}" for p in changed.columns])
    ax.set_yticks(range(len(changed.index)))
    ax.set_yticklabels([f"Regime {r}" for r in changed.index])
    ax.set_xlabel("Assumed φ_scan (CO_CLUSTER)")
    default_idx = list(changed.columns).index(DEFAULT_PHI)
    ax.axvline(default_idx, color="black", linestyle="--", linewidth=1)
    measured = _measured_phi()
    title = (f"φ_scan Sensitivity — tables whose layout differs from φ={DEFAULT_PHI} "
             f"(dashed)")
    if measured is not None:
        title += f"\nmeasured AJLO skip fraction on join-key queries: {measured:.2f}"
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Tables changed")
    plt.tight_layout()
    out = RESULTS_DIR / "phi_sensitivity.png"
    plt.savefig(out, dpi=1000)
    plt.rcParams['ps.fonttype'] = 42
    plt.savefig(RESULTS_DIR / "eps" / "phi_sensitivity.eps", format="eps", bbox_inches="tight")
    plt.close()
    console.print(f"  Chart → {out}")


if __name__ == "__main__":
    run()
