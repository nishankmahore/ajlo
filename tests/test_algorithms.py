"""Unit tests for all 5 AJLO algorithms. No Spark required."""
from datetime import datetime, timedelta

import pytest

from ajlo.models import JoinEdge, JoinGraph, LayoutPlan, LayoutStrategy
from ajlo.algorithms.join_graph import extract_join_graph, _extract_predicates
from ajlo.algorithms.jis import compute_jis
from ajlo.algorithms.layout_planner import recommend_layout
from ajlo.algorithms.drift_detector import detect_drift
from ajlo.algorithms.adaptive_reopt import adaptive_reoptimize
from ajlo.cost_model import npv, scan_reduction, rewrite_cost, skew_penalty


# ── Fixtures ─────────────────────────────────────────────────────────────────

NOW = datetime(2024, 1, 1, 0, 0, 0)
WINDOW = (NOW, NOW + timedelta(days=7))


def _make_query(sql: str, cost: float = 1000.0, cards: dict | None = None) -> dict:
    return {
        "sql": sql,
        "execution_cost": cost,
        "timestamp": NOW + timedelta(hours=1),
        "cardinalities": cards or {},
    }


def _make_graph(edges: list[JoinEdge]) -> JoinGraph:
    return JoinGraph(edges=edges, window_start=NOW, window_end=NOW + timedelta(days=7))


# ── Algorithm 1: Join Graph Extraction ───────────────────────────────────────

class TestExtractPredicates:
    def test_simple_equijoin(self):
        sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
        preds = _extract_predicates(sql)
        assert len(preds) == 1
        ti, ci, tj, cj = preds[0]
        assert {ti, tj} == {"orders", "customers"}
        assert ci == cj == "customer_id"

    def test_filter_predicate_excluded(self):
        # "o.status = 'OPEN'" — same-table predicate, should not appear
        sql = "SELECT * FROM orders o WHERE o.status = o.priority"
        preds = _extract_predicates(sql)
        assert preds == []

    def test_multi_join(self):
        sql = """
            SELECT * FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN lineitem l ON o.order_id = l.order_id
        """
        preds = _extract_predicates(sql)
        assert len(preds) == 2

    def test_canonical_order(self):
        sql_ab = "SELECT * FROM a JOIN b ON a.id = b.id"
        sql_ba = "SELECT * FROM b JOIN a ON b.id = a.id"
        assert _extract_predicates(sql_ab) == _extract_predicates(sql_ba)


class TestExtractJoinGraph:
    def test_frequency_counts(self):
        sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
        log = [_make_query(sql) for _ in range(5)]
        g = extract_join_graph(log, WINDOW)
        assert len(g.edges) == 1
        assert g.edges[0].frequency == 5

    def test_mean_data_volume(self):
        sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
        log = [
            _make_query(sql, cards={"orders": 1000, "customers": 100}),
            _make_query(sql, cards={"orders": 2000, "customers": 200}),
        ]
        g = extract_join_graph(log, WINDOW)
        assert g.edges[0].data_volume == pytest.approx(250_000.0)  # mean of (100k, 400k)

    def test_window_filtering(self):
        in_window = _make_query("SELECT * FROM a JOIN b ON a.id = b.id")
        out_of_window = {**in_window, "timestamp": NOW - timedelta(days=10)}
        g = extract_join_graph([in_window, out_of_window], WINDOW)
        assert g.edges[0].frequency == 1  # only in-window query counted

    def test_empty_log(self):
        g = extract_join_graph([], WINDOW)
        assert g.edges == []


# ── Algorithm 2: JIS Computation ─────────────────────────────────────────────

class TestComputeJIS:
    def test_jis_in_unit_interval(self):
        edges = [
            JoinEdge("orders", "customers", "customer_id", "customer_id",
                     frequency=10, data_volume=1e9, exec_cost=5000),
            JoinEdge("orders", "lineitem", "order_id", "order_id",
                     frequency=5, data_volume=1e6, exec_cost=1000),
            JoinEdge("lineitem", "part", "part_id", "part_id",
                     frequency=2, data_volume=1e4, exec_cost=200),
        ]
        g = compute_jis(_make_graph(edges))
        for e in g.edges:
            assert 0.0 <= e.jis <= 1.0

    def test_multiplicative_suppresses_small_tables(self):
        """High-frequency tiny join should score lower than balanced large-volume join.

        With only 2 edges, min-max collapses one dimension to 0 for each edge,
        making both JIS = 0.  We need >=3 edges to see the multiplicative effect.
        """
        edges = [
            JoinEdge("a", "b", "id", "id", frequency=100, data_volume=10, exec_cost=10),
            JoinEdge("c", "d", "id", "id", frequency=50, data_volume=1e9, exec_cost=5000),
            JoinEdge("e", "f", "id", "id", frequency=5,  data_volume=5e8, exec_cost=2500),
        ]
        g = compute_jis(_make_graph(edges))
        jis_map = {e.key(): e.jis for e in g.edges}
        # tiny-table high-frequency join (a.b) should score lower than large balanced join (c.d)
        assert jis_map["a.id=b.id"] < jis_map["c.id=d.id"]

    def test_single_edge_returns_zero(self):
        edges = [JoinEdge("a", "b", "id", "id", frequency=5, data_volume=1e6, exec_cost=500)]
        g = compute_jis(_make_graph(edges))
        assert g.edges[0].jis == 0.0

    def test_input_not_mutated(self):
        edges = [
            JoinEdge("a", "b", "id", "id", frequency=10, data_volume=1e6, exec_cost=1000),
            JoinEdge("c", "d", "id", "id", frequency=1, data_volume=1e3, exec_cost=100),
        ]
        original_jis = [e.jis for e in edges]
        compute_jis(_make_graph(edges))
        assert [e.jis for e in edges] == original_jis


# ── Algorithm 3: Layout Planner ───────────────────────────────────────────────

class TestRecommendLayout:
    def _graph(self) -> JoinGraph:
        edges = [
            JoinEdge("orders", "customers", "customer_id", "customer_id",
                     frequency=50, data_volume=1e8, exec_cost=10000, jis=0.8),
            JoinEdge("orders", "lineitem", "order_id", "order_id",
                     frequency=20, data_volume=5e7, exec_cost=5000, jis=0.4),
        ]
        return _make_graph(edges)

    def test_recommends_co_cluster_for_high_jis_edge(self):
        g = self._graph()
        plan = recommend_layout(
            g, LayoutPlan(),
            rewrite_budget=1e12, storage_budget=1e12,
            table_sizes={"orders": 1e9, "customers": 1e8, "lineitem": 5e9},
        )
        # Top edge should get CO_CLUSTER
        orders_strat = plan.get("orders")
        assert orders_strat.strategy_type in ("CO_CLUSTER", "CO_PARTITION")

    def test_budget_zero_returns_noop(self):
        g = self._graph()
        plan = recommend_layout(g, LayoutPlan(), rewrite_budget=0, storage_budget=0)
        for table in ("orders", "customers", "lineitem"):
            assert plan.get(table).strategy_type == "NOOP"

    def test_returns_new_plan_not_mutated(self):
        g = self._graph()
        original = LayoutPlan()
        new_plan = recommend_layout(
            g, original, rewrite_budget=1e12, storage_budget=1e12,
            table_sizes={"orders": 1e9, "customers": 1e8, "lineitem": 5e9},
        )
        # original should not be mutated
        assert original.strategies == {}


# ── Algorithm 4: Drift Detection ─────────────────────────────────────────────

class TestDetectDrift:
    def _graph_with_jis(self, jis_map: dict[str, float]) -> JoinGraph:
        edges = []
        for key, jis in jis_map.items():
            ti_ci, tj_cj = key.split("=")
            ti, ci = ti_ci.split(".")
            tj, cj = tj_cj.split(".")
            edges.append(JoinEdge(ti, tj, ci, cj, frequency=10,
                                  data_volume=1e6, exec_cost=1000, jis=jis))
        return _make_graph(edges)

    def test_no_drift_when_jis_stable(self):
        g1 = self._graph_with_jis({"orders.cid=customers.cid": 0.8,
                                    "orders.oid=lineitem.oid": 0.4})
        g2 = self._graph_with_jis({"orders.cid=customers.cid": 0.82,
                                    "orders.oid=lineitem.oid": 0.38})
        detected, score = detect_drift(g1, g2, delta=0.10)
        assert not detected
        assert score < 0.10

    def test_drift_detected_when_workload_shifts(self):
        g1 = self._graph_with_jis({"orders.cid=customers.cid": 0.9,
                                    "orders.oid=lineitem.oid": 0.1})
        g2 = self._graph_with_jis({"orders.cid=customers.cid": 0.1,
                                    "orders.oid=lineitem.oid": 0.9})
        detected, score = detect_drift(g1, g2, delta=0.10)
        assert detected

    def test_new_edge_triggers_drift(self):
        g1 = self._graph_with_jis({"orders.cid=customers.cid": 0.5})
        g2 = self._graph_with_jis({"orders.cid=customers.cid": 0.5,
                                    "lineitem.pid=part.pid": 0.9})
        detected, _ = detect_drift(g1, g2, delta=0.05)
        assert detected

    def test_empty_graphs_no_drift(self):
        g1 = _make_graph([])
        g2 = _make_graph([])
        detected, score = detect_drift(g1, g2)
        assert not detected
        assert score == 0.0


# ── Algorithm 5: Adaptive Re-optimization ────────────────────────────────────

class TestAdaptiveReoptimize:
    def _make_graphs(self) -> tuple[JoinGraph, JoinGraph]:
        old_edges = [
            JoinEdge("orders", "customers", "customer_id", "customer_id",
                     frequency=50, data_volume=1e8, exec_cost=10000, jis=0.9),
        ]
        new_edges = [
            JoinEdge("orders", "customers", "customer_id", "customer_id",
                     frequency=5, data_volume=1e4, exec_cost=100, jis=0.05),
            JoinEdge("orders", "lineitem", "order_id", "order_id",
                     frequency=80, data_volume=2e8, exec_cost=20000, jis=0.95),
        ]
        return _make_graph(old_edges), _make_graph(new_edges)

    def test_no_reorg_when_no_drift(self):
        old_edges = [JoinEdge("a", "b", "id", "id", frequency=10,
                               data_volume=1e6, exec_cost=1000, jis=0.5)]
        g_same = _make_graph(old_edges)
        plan, ops = adaptive_reoptimize(
            g_same, g_same, LayoutPlan(), 1e12, 1e12, delta=0.10
        )
        assert ops == []

    def test_reorg_ops_returned_on_drift(self):
        g_old, g_new = self._make_graphs()
        _, ops = adaptive_reoptimize(
            g_old, g_new, LayoutPlan(), 1e12, 1e12, delta=0.05,
            table_sizes={"orders": 1e9, "customers": 1e8, "lineitem": 5e9},
        )
        assert len(ops) > 0

    def test_ops_sorted_by_priority_descending(self):
        g_old, g_new = self._make_graphs()
        _, ops = adaptive_reoptimize(
            g_old, g_new, LayoutPlan(), 1e12, 1e12, delta=0.05,
            table_sizes={"orders": 1e9, "customers": 1e8, "lineitem": 5e9},
        )
        priorities = [op.priority for op in ops]
        assert priorities == sorted(priorities, reverse=True)


# ── Cost Model ────────────────────────────────────────────────────────────────

class TestCostModel:
    def test_npv_positive_for_beneficial_layout(self):
        sr = 1e10  # 10 GB saved per window
        rc = 1e9   # 1 GB rewrite cost
        sc = 0.02e9
        val = npv(sr, rc, sc, H=4, r=0.10)
        assert val > 0

    def test_npv_negative_when_rewrite_exceeds_benefit(self):
        sr = 1e6   # 1 MB saved per window
        rc = 1e10  # 10 GB rewrite
        val = npv(sr, rc, 0, H=4, r=0.10)
        assert val < 0

    def test_scan_reduction_co_cluster_gt_co_partition(self):
        sr_cc = scan_reduction(1e9, "CO_CLUSTER")
        sr_cp = scan_reduction(1e9, "CO_PARTITION")
        assert sr_cc > sr_cp

    def test_scan_reduction_noop_is_zero(self):
        assert scan_reduction(1e9, "NOOP") == 0.0

    def test_migration_penalty_applied(self):
        # Switching CO_CLUSTER → CO_PARTITION should cost 3× table size
        normal = rewrite_cost(1e9, "NOOP", "CO_CLUSTER")
        migration = rewrite_cost(1e9, "CO_CLUSTER", "CO_PARTITION")
        assert migration == pytest.approx(normal * 3.0)

    def test_skew_penalty_clamps_to_one(self):
        # Extreme skew: max_freq / avg_freq = 1000
        assert skew_penalty(1000.0, 1.0, kappa=0.1) == 1.0

    def test_skew_penalty_zero_when_no_skew(self):
        # max_freq == avg_freq → raw = kappa - 1 < 0 → clamped to 0
        assert skew_penalty(1.0, 1.0, kappa=0.1) == 0.0
