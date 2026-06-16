"""Algorithm 2: JIS Computation (Section 6.2, Algorithm 2).

JIS(i,j) = F̂(i,j) × D̂(i,j) × Ĉ(i,j)

D uses log-robust normalization to handle Pareto-distributed data volumes.
F and C use standard min-max normalization.

Complexity: O(|E|).
"""
import math

from ajlo.models import JoinEdge, JoinGraph

def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        # All values identical: dimension carries no signal, give full weight to all.
        return [1.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def _log_robust_minmax(values: list[float]) -> list[float]:
    """Log10-scale then min-max. Prevents large D values collapsing others."""
    logged = [math.log10(v + 1) for v in values]
    return _minmax(logged)


def compute_jis(graph: JoinGraph) -> JoinGraph:
    """Algorithm 2: Annotate every edge in the graph with its JIS score.

    Returns a NEW JoinGraph; input is not mutated.
    If the graph has 0 or 1 edges, JIS = 0.0 (normalization undefined).
    """
    if len(graph.edges) < 2:
        return JoinGraph(
            edges=[
                JoinEdge(
                    table_i=e.table_i, col_i=e.col_i,
                    table_j=e.table_j, col_j=e.col_j,
                    frequency=e.frequency, data_volume=e.data_volume,
                    exec_cost=e.exec_cost, jis=0.0,
                )
                for e in graph.edges
            ],
            window_start=graph.window_start,
            window_end=graph.window_end,
        )

    f_vals = [float(e.frequency) for e in graph.edges]
    d_vals = [float(e.data_volume) for e in graph.edges]
    c_vals = [float(e.exec_cost) for e in graph.edges]

    f_norm = _minmax(f_vals)
    d_norm = _log_robust_minmax(d_vals)
    c_norm = _minmax(c_vals)

    new_edges = [
        JoinEdge(
            table_i=e.table_i, col_i=e.col_i,
            table_j=e.table_j, col_j=e.col_j,
            frequency=e.frequency, data_volume=e.data_volume,
            exec_cost=e.exec_cost,
            jis=fn * dn * cn,
        )
        for e, fn, dn, cn in zip(graph.edges, f_norm, d_norm, c_norm)
    ]

    return JoinGraph(
        edges=new_edges,
        window_start=graph.window_start,
        window_end=graph.window_end,
    )
