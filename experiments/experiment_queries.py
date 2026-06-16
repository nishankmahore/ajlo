"""Benchmark query set for measuring layout performance after AJLO's decision.

70 % of queries filter on AJLO's cluster key (o_orderkey / l_orderkey).
30 % filter on B2's cluster key (o_orderdate / l_shipdate).

Each entry: (query_id, sql, filter_spec)
  filter_spec: {table_name: (filter_col, pred_min, pred_max)}
    → passed to delta_stats.py to compute real bytes read via Delta min/max stats
"""
from __future__ import annotations

ALL_EXPERIMENT_QUERIES = [
    (
        "Q01",
        "SELECT o.o_orderkey, o.o_totalprice, "
        "COUNT(l.l_linenumber) AS n_items, "
        "SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
        "WHERE o.o_orderkey BETWEEN 150000 AND 300000 "
        "GROUP BY o.o_orderkey, o.o_totalprice ORDER BY revenue DESC LIMIT 50",
        {
            "orders":   ("o_orderkey", 150000, 300000),
            "lineitem": ("l_orderkey", 150000, 300000),
        },
    ),
    (
        "Q02",
        "SELECT o.o_orderstatus, SUM(l.l_quantity) AS total_qty, "
        "SUM(l.l_extendedprice) AS total_price "
        "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
        "WHERE o.o_orderkey BETWEEN 500000 AND 650000 "
        "GROUP BY o.o_orderstatus",
        {
            "orders":   ("o_orderkey", 500000, 650000),
            "lineitem": ("l_orderkey", 500000, 650000),
        },
    ),
    (
        "Q03",
        "SELECT COUNT(*), AVG(l.l_discount), SUM(l.l_extendedprice) "
        "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
        "WHERE o.o_orderkey BETWEEN 750000 AND 900000 "
        "AND l.l_quantity > 10",
        {
            "orders":   ("o_orderkey", 750000, 900000),
            "lineitem": ("l_orderkey", 750000, 900000),
        },
    ),
    (
        "Q04",
        "SELECT l.l_shipmode, COUNT(*) AS cnt, SUM(l.l_extendedprice) AS rev "
        "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
        "WHERE o.o_orderkey BETWEEN 300000 AND 450000 "
        "GROUP BY l.l_shipmode ORDER BY rev DESC",
        {
            "orders":   ("o_orderkey", 300000, 450000),
            "lineitem": ("l_orderkey", 300000, 450000),
        },
    ),
    (
        "Q05",
        "SELECT o.o_orderpriority, COUNT(*) AS cnt, "
        "SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
        "WHERE o.o_orderkey BETWEEN 50000 AND 200000 "
        "GROUP BY o.o_orderpriority ORDER BY revenue DESC",
        {
            "orders":   ("o_orderkey", 50000, 200000),
            "lineitem": ("l_orderkey", 50000, 200000),
        },
    ),
    (
        "Q06",
        "SELECT l.l_partkey, SUM(l.l_quantity) AS total_qty, "
        "SUM(l.l_extendedprice) AS total_price "
        "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
        "WHERE o.o_orderkey BETWEEN 900000 AND 1050000 "
        "AND l.l_discount < 0.05 "
        "GROUP BY l.l_partkey ORDER BY total_price DESC LIMIT 20",
        {
            "orders":   ("o_orderkey", 900000, 1050000),
            "lineitem": ("l_orderkey", 900000, 1050000),
        },
    ),
    (
        "Q07",
        "SELECT o.o_custkey, COUNT(DISTINCT o.o_orderkey) AS n_orders, "
        "SUM(l.l_extendedprice * (1 - l.l_discount)) AS lifetime_value "
        "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
        "WHERE o.o_orderkey BETWEEN 600000 AND 750000 "
        "GROUP BY o.o_custkey ORDER BY lifetime_value DESC LIMIT 25",
        {
            "orders":   ("o_orderkey", 600000, 750000),
            "lineitem": ("l_orderkey", 600000, 750000),
        },
    ),
    (
        "Q08",
        "SELECT l_returnflag, l_linestatus, "
        "SUM(l_quantity) AS sum_qty, SUM(l_extendedprice) AS sum_price, "
        "COUNT(*) AS count_order "
        "FROM lineitem "
        "WHERE l_shipdate BETWEEN '1994-01-01' AND '1994-12-31' "
        "GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus",
        {"lineitem": ("l_shipdate", "1994-01-01", "1994-12-31")},
    ),
    (
        "Q09",
        "SELECT SUM(l_extendedprice * l_discount) AS revenue "
        "FROM lineitem "
        "WHERE l_shipdate BETWEEN '1993-07-01' AND '1993-10-01' "
        "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24",
        {"lineitem": ("l_shipdate", "1993-07-01", "1993-10-01")},
    ),
    (
        "Q10",
        "SELECT COUNT(*), AVG(o_totalprice), SUM(o_totalprice) "
        "FROM orders "
        "WHERE o_orderdate BETWEEN '1995-01-01' AND '1995-06-30'",
        {"orders": ("o_orderdate", "1995-01-01", "1995-06-30")},
    ),
]

# Documents what column each layout actually clusters on.
LAYOUT_CLUSTER_KEYS = {
    "b2": {
        "orders":   "o_orderdate",
        "lineitem": "l_shipdate",
        "customer": "c_mktsegment",
    },
    "ajlo": {
        "orders":   "o_orderkey",
        "lineitem": "l_orderkey",
        "customer": "c_custkey",
    },
}
