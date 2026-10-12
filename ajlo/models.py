"""Core data classes mapping to the paper's notation (Section 4.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class JoinEdge:
    """Represents one edge in the join graph G = (V, E, W).

    Maps to weight vector W(i,j) = (F, D, C) from Section 6.1.
    """
    table_i: str
    table_j: str
    col_i: str
    col_j: str
    frequency: int = 0          # F(i,j) — query count containing this join
    data_volume: float = 0.0    # D(i,j) — mean rows read |T_i| + |T_j| at query time
    exec_cost: float = 0.0      # C(i,j) — mean execution cost (ms)
    jis: float = 0.0            # Populated by Algorithm 2

    def key(self) -> str:
        """Canonical order-independent edge key."""
        a = f"{self.table_i}.{self.col_i}"
        b = f"{self.table_j}.{self.col_j}"
        return f"{min(a, b)}={max(a, b)}"


@dataclass
class JoinGraph:
    """Weighted join graph G = (V, E, W) over a query window."""
    edges: list[JoinEdge] = field(default_factory=list)
    window_start: datetime = field(default_factory=datetime.utcnow)
    window_end: datetime = field(default_factory=datetime.utcnow)

    def tables(self) -> set[str]:
        tables: set[str] = set()
        for e in self.edges:
            tables.add(e.table_i)
            tables.add(e.table_j)
        return tables

    def edges_for_table(self, table: str) -> list[JoinEdge]:
        return [e for e in self.edges if e.table_i == table or e.table_j == table]

    def edge_by_key(self) -> dict[str, JoinEdge]:
        return {e.key(): e for e in self.edges}


@dataclass
class LayoutStrategy:
    """Physical layout decision for a single table — λ(t) from Section 4.1.

    On Delta Lake: K_t = ∅, B_t = ∅ always (no Hive bucketing).
    CLUSTER BY and PARTITIONED BY are mutually exclusive on Delta Lake.
    """
    table: str
    strategy_type: str = "NOOP"         # "CO_CLUSTER" | "CO_PARTITION" | "NOOP"
    partition_col: Optional[str] = None  # P_t
    cluster_cols: list[str] = field(default_factory=list)  # C_t


@dataclass
class LayoutPlan:
    """Layout plan L: T → Λ mapping each table to its strategy."""
    strategies: dict[str, LayoutStrategy] = field(default_factory=dict)

    def get(self, table: str) -> LayoutStrategy:
        return self.strategies.get(table, LayoutStrategy(table=table, strategy_type="NOOP"))

    def copy(self) -> LayoutPlan:
        return LayoutPlan({
            k: LayoutStrategy(
                table=v.table,
                strategy_type=v.strategy_type,
                partition_col=v.partition_col,
                cluster_cols=list(v.cluster_cols),
            )
            for k, v in self.strategies.items()
        })


@dataclass
class ReorgOp:
    """A single table reorganization operation produced by Algorithm 5."""
    table: str
    from_strategy: LayoutStrategy
    to_strategy: LayoutStrategy
    priority: float = 0.0


@dataclass
class QueryMetrics:
    """Execution metrics captured after running a SQL query via Spark."""
    sql: str = ""
    bytes_read: int = 0
    files_read: int = 0
    files_total: int = 0        # Total files in table (for skip ratio)
    shuffle_bytes: int = 0
    wall_time_ms: float = 0.0
    result_rows: int = 0

    @property
    def files_skipped(self) -> int:
        return max(0, self.files_total - self.files_read)

    @property
    def skip_ratio(self) -> float:
        if self.files_total == 0:
            return 0.0
        return self.files_skipped / self.files_total
