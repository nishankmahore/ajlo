"""Algorithm 3: Layout Recommendation — Dynamic Greedy Approximation.

Section 7, Algorithm 3. Budget-constrained greedy NPV maximization.

The static version (sort edges once by JIS, iterate) is wrong — it ignores
conflicts when two high-JIS edges share a table. Dynamic re-prioritization
ensures the priority queue reflects actual residual NPV given the plan-so-far.

Approximation guarantee: (1 − 1/e) under submodularity (Sviridenko [23]).
Complexity: O(|E|² · |C| · log|E|) worst case (star schema); O(|E|·|C|·log|E|) typical.
"""
import heapq
from dataclasses import dataclass

from ajlo.models import JoinEdge, JoinGraph, LayoutPlan, LayoutStrategy
from ajlo.cost_model import (
    expected_scan_bytes,
    scan_reduction,
    rewrite_cost,
    storage_cost,
    skew_penalty,
    npv,
    BYTES_PER_ROW,
    SCAN_OVERHEAD,
)

DEFAULT_H = 4
DEFAULT_R = 0.10
DEFAULT_BYTES_PER_ROW = BYTES_PER_ROW
_JIS_MIN = 0.0


@dataclass
class _Candidate:
    strategy_type: str
    net_npv: float


def _table_bytes(table: str, table_sizes: dict[str, float]) -> float:
    return table_sizes.get(table, 1_000_000_000.0)  # default 1 GB


def _evaluate_candidates(
    edge: JoinEdge,
    plan: LayoutPlan,
    table_sizes: dict[str, float],
    skew_stats: dict[str, tuple[float, float]],
    H: int,
    r: float,
) -> _Candidate:
    """Evaluate CO_CLUSTER, CO_PARTITION, NOOP for one edge. Return best."""
    exp_scan = expected_scan_bytes(
        edge.frequency, edge.data_volume, DEFAULT_BYTES_PER_ROW, SCAN_OVERHEAD
    )
    # Skew: take the worse side of the join — either key can cause hot partitions
    max_f_i, avg_f_i = skew_stats.get(f"{edge.table_i}.{edge.col_i}", (1.0, 1.0))
    max_f_j, avg_f_j = skew_stats.get(f"{edge.table_j}.{edge.col_j}", (1.0, 1.0))
    skew = max(skew_penalty(max_f_i, avg_f_i), skew_penalty(max_f_j, avg_f_j))

    best_type = "NOOP"
    best_npv_val = _JIS_MIN

    for stype in ("CO_CLUSTER", "CO_PARTITION"):
        cur_i = plan.get(edge.table_i).strategy_type
        cur_j = plan.get(edge.table_j).strategy_type

        rc = (
            rewrite_cost(_table_bytes(edge.table_i, table_sizes), cur_i, stype)
            + rewrite_cost(_table_bytes(edge.table_j, table_sizes), cur_j, stype)
        )
        sc = (
            storage_cost(_table_bytes(edge.table_i, table_sizes), stype)
            + storage_cost(_table_bytes(edge.table_j, table_sizes), stype)
        )
        sr = scan_reduction(exp_scan, stype, skew)
        candidate_npv = npv(sr, rc, sc, H, r)

        if candidate_npv > best_npv_val:
            best_npv_val = candidate_npv
            best_type = stype

    return _Candidate(strategy_type=best_type, net_npv=best_npv_val)


def recommend_layout(
    graph: JoinGraph,
    current_plan: LayoutPlan,
    rewrite_budget: float,
    storage_budget: float,
    table_sizes: dict[str, float] | None = None,
    skew_stats: dict[str, tuple[float, float]] | None = None,
    H: int = DEFAULT_H,
    r: float = DEFAULT_R,
) -> LayoutPlan:
    """Algorithm 3: Return a new LayoutPlan maximising NPV under budget.

    Args:
        graph: JIS-annotated join graph (output of Algorithm 2).
        current_plan: Existing layout (used to compute migration costs).
        rewrite_budget: Max bytes worth of rewrite operations allowed.
        storage_budget: Max bytes of additional storage allowed.
        table_sizes: {table_name: size_in_bytes}. Defaults to 1 GB each.
        skew_stats: {table.col: (max_key_freq, avg_key_freq)}.
        H: Planning horizon in windows.
        r: Per-window discount rate.
    """
    if table_sizes is None:
        table_sizes = {}
    if skew_stats is None:
        skew_stats = {}

    new_plan = current_plan.copy()
    budget_rem = rewrite_budget
    storage_rem = storage_budget

    # Build initial priority queue: (-net_npv, idx)
    cache: dict[int, _Candidate] = {}
    pq: list[tuple[float, int]] = []

    for idx, edge in enumerate(graph.edges):
        cand = _evaluate_candidates(edge, new_plan, table_sizes, skew_stats, H, r)
        cache[idx] = cand
        heapq.heappush(pq, (-cand.net_npv, idx))

    claimed: dict[str, tuple[str, str]] = {}  # table → (strategy, key) set this round
    seen: set[int] = set()    # fully processed entries
    requeued: set[int] = set()  # stale entries re-queued once; avoid infinite re-queue

    while pq:
        neg_val, idx = heapq.heappop(pq)

        if idx in seen:
            continue

        # Stale entry — re-evaluate before using
        cand = _evaluate_candidates(
            graph.edges[idx], new_plan, table_sizes, skew_stats, H, r
        )
        if abs(cand.net_npv - (-neg_val)) > 1.0 and idx not in requeued:
            heapq.heappush(pq, (-cand.net_npv, idx))
            cache[idx] = cand
            requeued.add(idx)
            continue

        seen.add(idx)

        if cand.net_npv <= _JIS_MIN:
            continue

        edge = graph.edges[idx]
        stype = cand.strategy_type

        cur_i = new_plan.get(edge.table_i).strategy_type
        cur_j = new_plan.get(edge.table_j).strategy_type

        rc = (
            rewrite_cost(_table_bytes(edge.table_i, table_sizes), cur_i, stype)
            + rewrite_cost(_table_bytes(edge.table_j, table_sizes), cur_j, stype)
        )
        sc = (
            storage_cost(_table_bytes(edge.table_i, table_sizes), stype)
            + storage_cost(_table_bytes(edge.table_j, table_sizes), stype)
        )

        if rc > budget_rem or sc > storage_rem:
            continue  # skip, don't break — smaller changes may still fit

        # Apply layout change to both tables
        join_col_i = edge.col_i
        join_col_j = edge.col_j

        # A table keeps the key chosen by the higher-NPV edge that claimed it
        # earlier in this round; a later edge may only reuse it with the same key.
        # Without this, low-value edges overwrite high-value assignments.
        if any(
            t in claimed and claimed[t] != (stype, c)
            for t, c in ((edge.table_i, join_col_i), (edge.table_j, join_col_j))
        ):
            continue
        claimed[edge.table_i] = (stype, join_col_i)
        claimed[edge.table_j] = (stype, join_col_j)

        new_plan.strategies[edge.table_i] = LayoutStrategy(
            table=edge.table_i,
            strategy_type=stype,
            partition_col=join_col_i if stype == "CO_PARTITION" else None,
            cluster_cols=[join_col_i] if stype == "CO_CLUSTER" else [],
        )
        new_plan.strategies[edge.table_j] = LayoutStrategy(
            table=edge.table_j,
            strategy_type=stype,
            partition_col=join_col_j if stype == "CO_PARTITION" else None,
            cluster_cols=[join_col_j] if stype == "CO_CLUSTER" else [],
        )

        budget_rem -= rc
        storage_rem -= sc

        # Propagate: re-prioritize edges incident on the just-changed tables
        affected = {edge.table_i, edge.table_j}
        for nidx, nedge in enumerate(graph.edges):
            if nidx in seen:
                continue
            if nedge.table_i in affected or nedge.table_j in affected:
                ncand = _evaluate_candidates(
                    nedge, new_plan, table_sizes, skew_stats, H, r
                )
                cache[nidx] = ncand
                heapq.heappush(pq, (-ncand.net_npv, nidx))

    return new_plan
