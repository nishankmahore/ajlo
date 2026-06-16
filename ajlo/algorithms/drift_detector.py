"""Algorithm 4: Workload Drift Detection (Section 6.4, Algorithm 4).

Drift(G_k, G_{k+1}) = mean normalized absolute JIS change across all edges.
Complexity: O(|E_union|).
"""
from ajlo.models import JoinGraph

_EPSILON = 1e-9


def detect_drift(
    graph_old: JoinGraph,
    graph_new: JoinGraph,
    delta: float = 0.10,
) -> tuple[bool, float]:
    """Algorithm 4: Detect workload drift between two consecutive windows.

    Args:
        graph_old: Join graph from window k (JIS already computed).
        graph_new: Join graph from window k+1 (JIS already computed).
        delta: Drift threshold. Typical range: [0.05, 0.40].

    Returns:
        (drift_detected, drift_score)

    Raises:
        ValueError: If either graph has edges with jis=0 that were never scored
            (i.e. compute_jis() was not called). A non-empty graph where every
            edge has jis==0.0 is the telltale sign.
    """
    for label, g in (("graph_old", graph_old), ("graph_new", graph_new)):
        if g.edges and all(e.jis == 0.0 for e in g.edges):
            raise ValueError(
                f"detect_drift: {label} has {len(g.edges)} edges but all jis=0.0. "
                "Call compute_jis() before detect_drift()."
            )

    old_jis: dict[str, float] = {e.key(): e.jis for e in graph_old.edges}
    new_jis: dict[str, float] = {e.key(): e.jis for e in graph_new.edges}
    all_keys = set(old_jis) | set(new_jis)

    if not all_keys:
        return False, 0.0

    drift_sum = 0.0
    for key in all_keys:
        jis_old = old_jis.get(key, 0.0)
        jis_new = new_jis.get(key, 0.0)
        denom = max(jis_old, jis_new, _EPSILON)
        drift_sum += abs(jis_old - jis_new) / denom

    drift_score = drift_sum / len(all_keys)
    return drift_score > delta, drift_score
