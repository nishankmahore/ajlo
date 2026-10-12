"""PySpark session factory and query metrics collection.

This is the only file in the project that imports PySpark.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from experiments.config import DATA_DIR, IS_DATABRICKS, spark_path


def get_spark_session(app_name: str = "ajlo", driver_memory: str = "8g"):
    """Create or retrieve a local PySpark + Delta Lake session.

    On Databricks, return the cluster's existing session instead (Delta is built in).
    """
    if IS_DATABRICKS:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()

    import os
    import sys
    # Remove system Spark env vars that conflict with the venv PySpark install
    for key in ["PYSPARK_SUBMIT_ARGS", "SPARK_HOME", "PYTHONSTARTUP"]:
        os.environ.pop(key, None)
    # Force driver AND worker to use this venv's Python (not system Python 3.9)
    venv_python = sys.executable
    os.environ["PYSPARK_PYTHON"] = venv_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Enables column-level statistics for data skipping
        .config("spark.databricks.delta.properties.defaults.enableColumnMapping", "true")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Collect per-task input metrics
        .config("spark.sql.statistics.histogram.enabled", "true")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


@dataclass
class QueryMetrics:
    layout: str = ""
    query_id: str = ""
    sql: str = ""
    bytes_read: int = 0
    files_read: int = 0
    files_total: int = 0
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


def _collect_plan_metrics(executed_plan) -> dict:
    """Walk executed plan tree and sum all SQLMetric values by name.

    Uses Python's for-loop over the Scala collections (py4j exposes the Java
    Iterable interface), which is more reliable than calling .iterator() /
    .hasNext() / .next() manually.
    """
    totals: dict[str, int] = {}

    def _walk(node):
        try:
            for entry in node.metrics():
                name: str = entry._1()
                val: int = entry._2().value()
                totals[name] = totals.get(name, 0) + val
        except Exception:
            pass
        try:
            for child in node.children():
                _walk(child)
        except Exception:
            pass

    _walk(executed_plan)
    return totals


def table_size_bytes(layout_dir: Path) -> int:
    """Sum of all Parquet file sizes under a layout directory (bytes on disk)."""
    return sum(
        f.stat().st_size
        for f in layout_dir.rglob("*.parquet")
        if not f.name.startswith("_")
    )


def tables_touched_by_query(sql: str, table_names: list[str]) -> list[str]:
    """Return table names referenced in the SQL (case-insensitive search)."""
    sql_lower = sql.lower()
    return [t for t in table_names if t.lower() in sql_lower]


def run_query(
    spark,
    sql: str,
    layout: str = "",
    query_id: str = "",
    files_total: int = 0,
    layout_dir: Path | None = None,
) -> QueryMetrics:
    """Execute a SQL query and return measured execution metrics.

    bytes_read: Sum of Parquet file sizes on disk for tables touched by the query.
                This is the actual storage the query must read from — a real
                measurement of how OPTIMIZE and clustering affect storage footprint.
    wall_time_ms: Real wall-clock execution time (primary latency metric).
    """
    # Compute bytes_read from actual disk sizes for touched tables
    bytes_read = 0
    if layout_dir is not None and layout_dir.exists():
        all_tables = [p.name for p in layout_dir.iterdir() if p.is_dir()]
        touched = tables_touched_by_query(sql, all_tables)
        for t in touched:
            bytes_read += table_size_bytes(layout_dir / t)

    t0 = time.perf_counter()
    df = spark.sql(sql)
    rows = df.count()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    try:
        plan_metrics = _collect_plan_metrics(df.queryExecution.executedPlan)
        # "data size" is the SQLMetric name on ShuffleExchangeExec nodes;
        # "shuffle bytes written" is a task-level metric and never appears here.
        shuffle_bytes = plan_metrics.get("data size", 0)
    except Exception:
        shuffle_bytes = 0

    return QueryMetrics(
        layout=layout,
        query_id=query_id,
        sql=sql[:120],
        bytes_read=int(bytes_read),
        files_read=0,
        files_total=files_total,
        shuffle_bytes=int(shuffle_bytes),
        wall_time_ms=elapsed_ms,
        result_rows=rows,
    )


def time_query(spark, sql: str, repeats: int | None = None,
               warmup: int | None = None) -> list[float]:
    """Run sql `warmup` times untimed, then `repeats` timed runs; return ms per run."""
    from experiments.config import QUERY_REPEATS, QUERY_WARMUP
    repeats = QUERY_REPEATS if repeats is None else repeats
    warmup = QUERY_WARMUP if warmup is None else warmup
    for _ in range(warmup):
        spark.sql(sql).count()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        spark.sql(sql).count()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def count_parquet_files(path: str | Path) -> int:
    """Count the number of Parquet data files at a path (not metadata)."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*.parquet") if not f.name.startswith("_"))


def register_views(spark, layout_dir: Path, suffix: str = "") -> dict[str, int]:
    """Register all Delta tables in a layout directory as temp views.

    Returns {table_name: total_file_count} for skip-ratio computation.
    """
    file_counts: dict[str, int] = {}
    for table_path in sorted(layout_dir.iterdir()):
        if not table_path.is_dir():
            continue
        table_name = table_path.name
        view_name = f"{table_name}{suffix}"
        # All layouts are written as Delta — use Delta reader
        df = spark.read.format("delta").load(spark_path(table_path))
        df.createOrReplaceTempView(view_name)
        file_counts[table_name] = count_parquet_files(table_path)
    return file_counts
