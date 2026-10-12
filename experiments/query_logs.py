"""Synthetic query log generator — input to AJLO's Algorithms 1→2→3.

Produces a list of query log entries (dicts) representing historical query
executions. AJLO mines this log to build the join graph, compute JIS scores,
and recommend a coordinated layout plan.

This is NOT the benchmark query set. See experiment_queries.py for that.

Generates query log entries for the three workload regimes
(A, B, C) from Section 9.3, adapted to the full TPC-H schema.

Regime A: orders ↔ lineitem dominated (high volume, frequent)
Regime B: orders ↔ customer dominated
Regime C: mixed — all join patterns roughly equal
Regime D: part-key joins dominate (lineitem ↔ part / partsupp on partkey) —
          a join predicate that is rare in A/B/C, used by RQ3 to force a
          genuine re-clustering decision rather than a frequency shift.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Literal

# ── TPC-H SF=1 approximate row counts (used as cardinality estimates) ─────────
CARDINALITIES = {
    "orders":   1_500_000,
    "lineitem": 6_001_215,
    "customer":   150_000,
    "part":       200_000,
    "supplier":    10_000,
    "partsupp":   800_000,
    "nation":          25,
    "region":           5,
}

_SCALE_EXEMPT = {"nation", "region"}  # fixed size regardless of SF


def get_cardinalities(sf: int = 1) -> dict[str, int]:
    """Return TPC-H row counts scaled to the given scale factor."""
    return {
        t: (n if t in _SCALE_EXEMPT else n * sf)
        for t, n in CARDINALITIES.items()
    }

# ── TPC-H join-heavy query templates (simplified, predicate-preserving) ───────
# Each template is a (query_id, sql, regime_weights) tuple.
# regime_weights = (A_prob, B_prob, C_prob[, D_prob]); a missing D weight
# means the template is background noise in regime D (_D_BACKGROUND).

_QUERY_TEMPLATES: list[tuple[str, str, tuple[float, float, float]]] = [
    # Q1 — lineitem scan, no join
    ("Q1",
     "SELECT l_returnflag, l_linestatus, SUM(l_quantity), COUNT(*) "
     "FROM lineitem WHERE l_shipdate <= '1998-09-01' GROUP BY l_returnflag, l_linestatus",
     (0.05, 0.05, 0.10)),

    # Q3 — orders ↔ lineitem ↔ customer
    ("Q3",
     "SELECT o.o_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue, "
     "o.o_orderdate, o.o_shippriority "
     "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
     "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
     "WHERE c.c_mktsegment = 'BUILDING' AND o.o_orderdate < '1995-03-15' "
     "GROUP BY o.o_orderkey, o.o_orderdate, o.o_shippriority "
     "ORDER BY revenue DESC LIMIT 10",
     (0.10, 0.15, 0.12)),

    # Q4 — orders ↔ lineitem (order priority check)
    ("Q4",
     "SELECT o.o_orderpriority, COUNT(*) AS order_count "
     "FROM orders o JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
     "WHERE o.o_orderdate >= '1993-07-01' AND o.o_orderdate < '1993-10-01' "
     "AND l.l_commitdate < l.l_receiptdate "
     "GROUP BY o.o_orderpriority ORDER BY o.o_orderpriority",
     (0.15, 0.05, 0.08)),

    # Q5 — orders ↔ lineitem ↔ customer ↔ supplier ↔ nation ↔ region
    ("Q5",
     "SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
     "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
     "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
     "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
     "JOIN nation n ON c.c_nationkey = n.n_nationkey "
     "AND s.s_nationkey = n.n_nationkey "
     "JOIN region r ON n.n_regionkey = r.r_regionkey "
     "WHERE r.r_name = 'ASIA' AND o.o_orderdate >= '1994-01-01' "
     "GROUP BY n.n_name ORDER BY revenue DESC",
     (0.08, 0.10, 0.12)),

    # Q6 — lineitem only (no join)
    ("Q6",
     "SELECT SUM(l_extendedprice * l_discount) AS revenue "
     "FROM lineitem "
     "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' "
     "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24",
     (0.05, 0.03, 0.05)),

    # Q7 — orders ↔ lineitem ↔ customer ↔ supplier (volume shipping)
    ("Q7",
     "SELECT s.s_suppkey, c.c_custkey, "
     "SUM(l.l_extendedprice * (1 - l.l_discount)) AS volume "
     "FROM supplier s JOIN lineitem l ON s.s_suppkey = l.l_suppkey "
     "JOIN orders o ON o.o_orderkey = l.l_orderkey "
     "JOIN customer c ON c.c_custkey = o.o_custkey "
     "JOIN nation n1 ON s.s_nationkey = n1.n_nationkey "
     "JOIN nation n2 ON c.c_nationkey = n2.n_nationkey "
     "WHERE ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') "
     "OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE')) "
     "GROUP BY s.s_suppkey, c.c_custkey LIMIT 100",
     (0.08, 0.08, 0.10)),

    # Q10 — orders ↔ lineitem ↔ customer ↔ nation (returned items)
    ("Q10",
     "SELECT c.c_custkey, c.c_name, "
     "SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
     "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
     "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
     "JOIN nation n ON c.c_nationkey = n.n_nationkey "
     "WHERE o.o_orderdate >= '1993-10-01' AND l.l_returnflag = 'R' "
     "GROUP BY c.c_custkey, c.c_name ORDER BY revenue DESC LIMIT 20",
     (0.08, 0.20, 0.12)),

    # Q12 — orders ↔ lineitem (shipping modes)
    ("Q12",
     "SELECT l.l_shipmode, "
     "SUM(CASE WHEN o.o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high, "
     "SUM(CASE WHEN o.o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low "
     "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
     "WHERE l.l_shipmode IN ('MAIL','SHIP') "
     "AND l.l_commitdate < l.l_receiptdate "
     "GROUP BY l.l_shipmode ORDER BY l.l_shipmode",
     (0.15, 0.05, 0.08)),

    # Q18 — customer ↔ orders ↔ lineitem (large volume customers)
    ("Q18",
     "SELECT c.c_name, o.o_orderkey, o.o_orderdate, o.o_totalprice, "
     "SUM(l.l_quantity) AS qty "
     "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
     "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
     "GROUP BY c.c_name, o.o_orderkey, o.o_orderdate, o.o_totalprice "
     "HAVING SUM(l.l_quantity) > 300 "
     "ORDER BY o.o_totalprice DESC, o.o_orderdate LIMIT 100",
     (0.08, 0.15, 0.12)),

    # Q20 — part ↔ partsupp ↔ supplier ↔ nation
    ("Q20",
     "SELECT s.s_suppkey, s.s_name "
     "FROM supplier s JOIN nation n ON s.s_nationkey = n.n_nationkey "
     "JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey "
     "JOIN part p ON p.p_partkey = ps.ps_partkey "
     "WHERE n.n_name = 'CANADA' AND p.p_name LIKE 'forest%' "
     "ORDER BY s.s_name LIMIT 50",
     (0.08, 0.05, 0.06)),

    # Q21 — supplier ↔ lineitem ↔ orders ↔ nation (suppliers who kept orders waiting)
    ("Q21",
     "SELECT s.s_name, COUNT(*) AS numwait "
     "FROM supplier s JOIN lineitem l1 ON s.s_suppkey = l1.l_suppkey "
     "JOIN orders o ON o.o_orderkey = l1.l_orderkey "
     "JOIN nation n ON s.s_nationkey = n.n_nationkey "
     "WHERE o.o_orderstatus = 'F' AND n.n_name = 'SAUDI ARABIA' "
     "AND l1.l_receiptdate > l1.l_commitdate "
     "GROUP BY s.s_name ORDER BY numwait DESC, s.s_name LIMIT 100",
     (0.06, 0.05, 0.05)),

    # ── Regime D only: part-key joins (zero weight in A/B/C) ─────────────────
    # Q9 — part ↔ lineitem ↔ partsupp ↔ supplier ↔ nation (product profit; the
    #      TPC-H orders join for the year breakdown is dropped so regime D is a
    #      structural shift to part-key joins, not a mix with orderkey joins)
    ("Q9",
     "SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount) "
     "- ps.ps_supplycost * l.l_quantity) AS profit "
     "FROM part p JOIN lineitem l ON p.p_partkey = l.l_partkey "
     "JOIN partsupp ps ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey "
     "JOIN supplier s ON s.s_suppkey = l.l_suppkey "
     "JOIN nation n ON s.s_nationkey = n.n_nationkey "
     "WHERE p.p_name LIKE '%green%' GROUP BY n.n_name",
     (0.0, 0.0, 0.0, 0.20)),

    # Q14 — lineitem ↔ part (promotion effect)
    ("Q14",
     "SELECT SUM(CASE WHEN p.p_type LIKE 'PROMO%' "
     "THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END) AS promo "
     "FROM lineitem l JOIN part p ON l.l_partkey = p.p_partkey "
     "WHERE l.l_shipdate >= '1995-09-01' AND l.l_shipdate < '1995-10-01'",
     (0.0, 0.0, 0.0, 0.25)),

    # Q17 — lineitem ↔ part (small-quantity revenue)
    ("Q17",
     "SELECT SUM(l.l_extendedprice) / 7.0 AS avg_yearly "
     "FROM lineitem l JOIN part p ON p.p_partkey = l.l_partkey "
     "WHERE p.p_brand = 'Brand#23' AND p.p_container = 'MED BOX' AND l.l_quantity < 5",
     (0.0, 0.0, 0.0, 0.20)),

    # Q19 — lineitem ↔ part (discounted revenue)
    ("Q19",
     "SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
     "FROM lineitem l JOIN part p ON p.p_partkey = l.l_partkey "
     "WHERE p.p_brand = 'Brand#12' AND l.l_quantity BETWEEN 1 AND 11 "
     "AND l.l_shipmode IN ('AIR', 'REG AIR')",
     (0.0, 0.0, 0.0, 0.20)),
]

_D_BACKGROUND = 0.005 # weight in regime D for templates without an explicit D weight

Regime = Literal["A", "B", "C", "D"]

_REGIME_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _exec_cost(query_id: str, rng: random.Random) -> float:
    """Simulate realistic execution costs (ms) with noise."""
    base = {
        "Q1": 4000, "Q3": 15000, "Q4": 12000, "Q5": 20000,
        "Q6": 3000, "Q7": 18000, "Q10": 14000, "Q12": 10000,
        "Q18": 16000, "Q20": 8000, "Q21": 17000,
        "Q9": 22000, "Q14": 9000, "Q17": 14000, "Q19": 11000,
    }.get(query_id, 10000)
    return base * rng.uniform(0.7, 1.3)


def generate_query_log(
    regime: Regime,
    n_queries: int = 200,
    start_time: datetime | None = None,
    seed: int = 42,
    sf: int = 1,
) -> list[dict]:
    """Generate a synthetic query log for a given workload regime.

    Row counts recorded with each query are scaled to SF so the cost model's
    scan estimates match the SF-scaled table sizes used for rewrite costs.

    Returns a list of dicts compatible with ajlo.algorithms.join_graph.extract_join_graph.
    """
    rng = random.Random(seed)
    if start_time is None:
        start_time = datetime(2024, 1, 1)

    idx = _REGIME_IDX[regime]
    cards = get_cardinalities(sf)
    weights = [t[2][idx] if idx < len(t[2]) else _D_BACKGROUND for t in _QUERY_TEMPLATES]
    total = sum(weights)
    probs = [w / total for w in weights]

    log = []
    for i in range(n_queries):
        # Weighted random query selection
        r = rng.random()
        cumulative = 0.0
        selected = _QUERY_TEMPLATES[-1]
        for tmpl, prob in zip(_QUERY_TEMPLATES, probs):
            cumulative += prob
            if r < cumulative:
                selected = tmpl
                break

        qid, sql, _ = selected
        ts = start_time + timedelta(hours=i * (24 * 7 / n_queries))

        log.append({
            "query_id":       qid,
            "sql":            sql,
            "execution_cost": _exec_cost(qid, rng),
            "timestamp":      ts,
            "cardinalities":  dict(cards),
        })

    return log


def generate_drift_log(
    transitions: list[tuple[Regime, int]],
    start_time: datetime | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate a multi-regime log with drift transitions.

    transitions = [(regime, n_queries), ...] executed in sequence.
    Example: [("A", 100), ("B", 100), ("C", 100)]
    """
    if start_time is None:
        start_time = datetime(2024, 1, 1)
    combined = []
    current_time = start_time
    for i, (regime, n) in enumerate(transitions):
        log = generate_query_log(regime, n, start_time=current_time, seed=seed + i)
        combined.extend(log)
        if log:
            current_time = log[-1]["timestamp"] + timedelta(hours=1)
    return combined
