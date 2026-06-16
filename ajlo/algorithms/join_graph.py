"""Algorithm 1: Join Graph Extraction (Section 5.2, Algorithm 1).

Parses SQL text for equijoin predicates. Sufficient for TPC-H's
well-structured queries. Production use would operate on analyzed
logical plans (see Section 5.2 implementation notes).
"""
import re
from collections import defaultdict
from datetime import datetime

from ajlo.models import JoinEdge, JoinGraph

# Matches: alias.col = alias.col  (equijoin predicate)
_EQUIJOIN_RE = re.compile(
    r'\b(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\b',
    re.IGNORECASE,
)

# Matches: FROM table [AS] alias  or  JOIN table [AS] alias
_ALIAS_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+(\w+)(?:\s+AS\s+|\s+)(\w+)',
    re.IGNORECASE,
)


def _build_alias_map(sql: str) -> dict[str, str]:
    """Return {alias: real_table_name} from SQL text."""
    mapping: dict[str, str] = {}
    for m in _ALIAS_RE.finditer(sql):
        table = m.group(1).lower()
        alias = m.group(2).lower()
        # Skip SQL keywords that follow FROM/JOIN
        if alias in {'where', 'on', 'set', 'inner', 'outer', 'left', 'right',
                     'cross', 'full', 'natural', 'group', 'order', 'having',
                     'limit', 'union', 'intersect', 'except', 'with'}:
            alias = table
        mapping[alias] = table
        mapping[table] = table
    return mapping


def _extract_predicates(sql: str) -> list[tuple[str, str, str, str]]:
    """Extract canonical (table_i, col_i, table_j, col_j) equijoin tuples."""
    alias_map = _build_alias_map(sql)
    predicates: set[tuple[str, str, str, str]] = set()

    for m in _EQUIJOIN_RE.finditer(sql):
        a_alias = m.group(1).lower()
        a_col = m.group(2).lower()
        b_alias = m.group(3).lower()
        b_col = m.group(4).lower()

        if a_alias == b_alias:
            continue  # filter predicate, not a join

        a_table = alias_map.get(a_alias, a_alias)
        b_table = alias_map.get(b_alias, b_alias)

        if a_table == b_table:
            continue  # self-join on same table, skip

        # Canonicalize order so (A,B) and (B,A) are the same edge
        if (a_table, a_col) > (b_table, b_col):
            a_table, a_col, b_table, b_col = b_table, b_col, a_table, a_col

        predicates.add((a_table, a_col, b_table, b_col))

    return list(predicates)


def extract_join_graph(
    query_log: list[dict],
    window: tuple[datetime, datetime],
) -> JoinGraph:
    """Algorithm 1: Build a weighted join graph from a query log window.

    Each query_log entry must contain:
      sql            : str   — SQL text
      execution_cost : float — wall-clock time in ms
      timestamp      : datetime
      cardinalities  : dict[table → int]  — optional row count estimates

    Complexity: O(|Q| · p) where p = max join predicates per query.
    """
    window_start, window_end = window

    freq: dict[str, int] = defaultdict(int)
    data_vol: dict[str, float] = defaultdict(float)
    exec_cost: dict[str, float] = defaultdict(float)
    edge_meta: dict[str, tuple[str, str, str, str]] = {}

    for q in query_log:
        ts: datetime = q.get("timestamp", window_start)
        if not (window_start <= ts <= window_end):
            continue

        sql: str = q.get("sql", "")
        cost: float = float(q.get("execution_cost", 0.0))
        cards: dict[str, float] = q.get("cardinalities", {})

        predicates = _extract_predicates(sql)
        cost_per_pred = cost / len(predicates) if predicates else 0.0
        for ti, ci, tj, cj in predicates:
            key = f"{ti}.{ci}={tj}.{cj}"
            freq[key] += 1
            data_vol[key] += cards.get(ti, 1.0) * cards.get(tj, 1.0)
            exec_cost[key] += cost_per_pred
            edge_meta[key] = (ti, ci, tj, cj)

    edges = []
    for key, (ti, ci, tj, cj) in edge_meta.items():
        f = freq[key]
        edges.append(JoinEdge(
            table_i=ti, col_i=ci,
            table_j=tj, col_j=cj,
            frequency=f,
            data_volume=data_vol[key] / f,
            exec_cost=exec_cost[key] / f,
        ))

    return JoinGraph(edges=edges, window_start=window_start, window_end=window_end)
