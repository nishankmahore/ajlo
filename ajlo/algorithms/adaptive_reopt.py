"""Algorithm 5: Adaptive Re-optimization (Section 5, Algorithm 5).

Composes Algorithms 4 (drift detection) → 3 (layout planning).
Only triggers re-optimization when drift exceeds threshold.
Complexity: O(|Q_new|·p + |E_new|·log|E_new|).
"""
from ajlo.models import JoinGraph, LayoutPlan, ReorgOp
from ajlo.algorithms.drift_detector import detect_drift
from ajlo.algorithms.layout_planner import recommend_layout


def _layout_signature(s) -> tuple:
    """Physical identity of a layout: strategy type plus the keys it uses."""
    return (s.strategy_type, s.partition_col, tuple(s.cluster_cols))


def adaptive_reoptimize(
    graph_old: JoinGraph,
    graph_new: JoinGraph,
    current_plan: LayoutPlan,
    rewrite_budget: float,
    storage_budget: float,
    delta: float = 0.10,
    table_sizes: dict[str, float] | None = None,
    skew_stats: dict[str, tuple[float, float]] | None = None,
    H: int = 4,
    r: float = 0.10,
) -> tuple[LayoutPlan, list[ReorgOp]]:
    """Algorithm 5: Return updated plan and prioritised reorganisation ops.

    Returns (current_plan, []) if no significant drift is detected.
    """
    drift_detected, drift_score = detect_drift(graph_old, graph_new, delta)

    if not drift_detected:
        return current_plan, []

    new_plan = recommend_layout(
        graph_new,
        current_plan,
        rewrite_budget,
        storage_budget,
        table_sizes=table_sizes,
        skew_stats=skew_stats,
        H=H,
        r=r,
    )

    # Build reorg ops for changed tables only
    all_tables = set(new_plan.strategies) | set(current_plan.strategies)
    ops: list[ReorgOp] = []

    for table in all_tables:
        old_s = current_plan.get(table)
        new_s = new_plan.get(table)
        # A change of key (e.g. CLUSTER BY l_orderkey → l_partkey) is a physical
        # rewrite just like a change of strategy type.
        if _layout_signature(old_s) != _layout_signature(new_s):
            # Priority = max JIS of edges involving this table
            priority = max(
                (e.jis for e in graph_new.edges
                 if e.table_i == table or e.table_j == table),
                default=0.0,
            )
            ops.append(ReorgOp(
                table=table,
                from_strategy=old_s,
                to_strategy=new_s,
                priority=priority,
            ))

    ops.sort(key=lambda op: op.priority, reverse=True)
    return new_plan, ops
