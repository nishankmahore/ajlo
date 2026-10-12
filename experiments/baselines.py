"""Baseline layout implementations (Section 8.4).

Each baseline is a function that writes re-organised Delta tables
from the raw Parquet source into a layout-specific output directory.

B0:   Raw Parquet, no layout directives
B1:   Per-table partition on most-filtered column (no cross-table coordination)
B2:   Liquid Clustering on frequently-filtered columns (ALC proxy)
B3:   Per-table independent optimum — each table clusters on the column (filter
      or join) most often predicated on it in the same query log AJLO mines,
      with no cross-table coordination (join partners need not share a key)
AJLO: Coordinated co-clustering derived by running Algorithms 1→2→3 on the
      TPC-H workload query log.  The library makes every layout decision here.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.progress import track

console = Console()

from experiments.config import IS_DATABRICKS, OPTIMIZE_MAX_FILE_SIZE, RAW_DIR, spark_path, try_set_conf
from experiments.config import LAYOUTS_DIR as BASE_DIR


def _link(path: Path) -> str:
    return f"[link=file://{path}]{path}[/link]"

# ── Per-table layout hints (manually specified, as a DBA would) ───────────────

# Most frequently filtered columns per table (for B1/B2)
FILTER_COLS = {
    "orders":   "o_orderdate",
    "lineitem": "l_shipdate",
    "customer": "c_mktsegment",
    "part":     "p_type",
    "supplier": "s_nationkey",
    "partsupp": "ps_partkey",
    "nation":   "n_regionkey",
    "region":   "r_regionkey",
}


TABLES = list(FILTER_COLS.keys())


# Target file size for OPTIMIZE — small enough to produce multiple files on SF=1 data
# so Delta's min/max statistics have something to skip against.
# spark.databricks.delta.optimize.maxFileSize is the OSS Delta Spark conf key.
# Override with AJLO_OPTIMIZE_MAX_FILE_MB (see experiments/config.py).


# Databricks-only write optimizations that OSS Delta does not apply. Disabled on every
# layout table so all baselines get identical, unassisted file layouts.
# (OSS Delta rejects some of these keys, so they are only set on Databricks.)
_NO_AUTO_OPT_PROPS = (
    ", 'delta.autoOptimize.optimizeWrite' = 'false'"
    ", 'delta.autoOptimize.autoCompact' = 'false'"
    if IS_DATABRICKS else ""
)


def _load_raw(spark, table: str):
    """Load a raw Parquet table."""
    path = RAW_DIR / table
    return spark.read.parquet(spark_path(path))


def _write_clustered(spark, df, out_path: Path, cluster_col: str) -> None:
    """Write a Delta table with CLUSTER BY cluster_col, then OPTIMIZE.

    Uses SQL CTAS (CREATE OR REPLACE TABLE ... CLUSTER BY) rather than the
    DataFrameWriter.clusterBy() API.  The DataFrameWriter path does not write
    the clustering specification into the Delta log metadata for path-based
    tables in OSS Delta, so clusterBy ends up as "[]" and OPTIMIZE skips the
    Hilbert-curve pass entirely.  The SQL DDL path correctly sets the feature
    flag and clusteringColumns entry in the log before OPTIMIZE runs.
    """
    _tmp = "_tmp_cluster_write"
    df.createOrReplaceTempView(_tmp)
    # Databricks sizes liquid-clustered files from delta.targetFileSize, not the conf below.
    target_size_prop = (f", 'delta.targetFileSize' = '{OPTIMIZE_MAX_FILE_SIZE}'"
                        if IS_DATABRICKS else "")

    spark.sql(f"""
        CREATE OR REPLACE TABLE delta.`{spark_path(out_path)}`
        USING DELTA
        TBLPROPERTIES ('delta.enableDeletionVectors' = 'false'{target_size_prop}{_NO_AUTO_OPT_PROPS})
        CLUSTER BY ({cluster_col})
        AS SELECT * FROM {_tmp}
    """)
    spark.catalog.dropTempView(_tmp)

    # Set max file size and run OPTIMIZE to physically reorganise into sorted files
    try_set_conf(spark, "spark.databricks.delta.optimize.maxFileSize",
                   str(OPTIMIZE_MAX_FILE_SIZE))
    console.print(f"  Running OPTIMIZE on {out_path} with max file size "
                  f"{OPTIMIZE_MAX_FILE_SIZE // (1024 * 1024)} MB...")
    spark.sql(f"OPTIMIZE delta.`{spark_path(out_path)}`")


def _write_delta(spark, df, out_path: Path, partition_col: str | None = None) -> None:
    """Write Delta table with no clustering (B0), optionally Hive-partitioned (B1).

    SQL CTAS so the same TBLPROPERTIES as the clustered layouts can be applied.
    """
    _tmp = "_tmp_plain_write"
    df.createOrReplaceTempView(_tmp)
    partition_clause = f"PARTITIONED BY ({partition_col})" if partition_col else ""
    spark.sql(f"""
        CREATE OR REPLACE TABLE delta.`{spark_path(out_path)}`
        USING DELTA
        {partition_clause}
        TBLPROPERTIES ('delta.enableDeletionVectors' = 'false'{_NO_AUTO_OPT_PROPS})
        AS SELECT * FROM {_tmp}
    """)
    spark.catalog.dropTempView(_tmp)


def build_b0(spark) -> Path:
    """B0: Copy raw Parquet → Delta with no layout changes."""
    out = BASE_DIR / "b0"
    console.print("Building B0 (no optimization)...")
    for table in track(TABLES, description="B0 tables"):
        df = _load_raw(spark, table)
        _write_delta(spark, df, out / table)
        console.print(f"  {table}: {df.count():,} rows")
    return out


def build_b1(spark) -> Path:
    """B1: Partition each table on its most-filtered column.

    Date columns are bucketed to year-month strings to avoid creating
    thousands of tiny per-day partitions on a 1 GB table.
    High-cardinality numeric keys (ps_partkey) use modulo 100 bucketing.
    """
    out = BASE_DIR / "b1"
    console.print("Building B1 (per-table partitioning)...")

    DATE_TABLES = {
        "orders":   ("o_orderdate_ym",  "DATE_FORMAT(CAST(o_orderdate  AS DATE), 'yyyy-MM')"),
        "lineitem": ("l_shipdate_ym",   "DATE_FORMAT(CAST(l_shipdate   AS DATE), 'yyyy-MM')"),
    }

    for table in track(TABLES, description="B1 tables"):
        df = _load_raw(spark, table)

        if table in DATE_TABLES:
            new_col, expr = DATE_TABLES[table]
            df = df.selectExpr("*", f"{expr} AS {new_col}")
            partition_col = new_col
        elif table == "partsupp":
            from pyspark.sql.functions import col as scol
            df = df.withColumn("ps_part_bucket", (scol("ps_partkey") % 100).cast("int"))
            partition_col = "ps_part_bucket"
        else:
            partition_col = FILTER_COLS[table]

        _write_delta(spark, df, out / table, partition_col=partition_col)
        console.print(f"  {table}: partitioned by {partition_col}")
    return out



def build_b2(spark) -> Path:
    """B2: ALC proxy — each table sorted by its per-table frequently-filtered column.

    Mimics what Automatic Liquid Clustering (ALC) would do: cluster each table
    independently on the column with the highest single-table filter frequency.
    Has NO knowledge of join relationships between tables.

    Key contrast with AJLO:
      B2:   orders sorted on o_orderdate  |  lineitem sorted on l_shipdate
      AJLO: orders sorted on o_orderkey   |  lineitem sorted on l_orderkey
    → B2 wins on date-range queries; AJLO wins on join-key-range queries.
    """
    out = BASE_DIR / "b2"
    console.print("Building B2 (ALC proxy — per-table filter-column sort)...")
    for table in track(TABLES, description="B2 tables"):
        col = FILTER_COLS[table]
        df = _load_raw(spark, table)
        _write_clustered(spark, df, out / table, cluster_col=col)
        console.print(f"  {table}: sorted by {col}  ← per-table filter column (ALC)")
    return out



# TPC-H column prefix → table ("ps_" must be tested before "p_").
_COL_PREFIX = [("ps_", "partsupp"), ("l_", "lineitem"), ("o_", "orders"), ("p_", "part"),
               ("c_", "customer"), ("s_", "supplier"), ("n_", "nation"), ("r_", "region")]
_COL_RE = re.compile(r"\b(?:\w+\.)?((?:ps|l|o|p|c|s|n|r)_\w+)\b", re.IGNORECASE)
_PRED_END_RE = re.compile(r"\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b", re.IGNORECASE)


def _predicate_columns(sql: str) -> set[str]:
    """Columns referenced in join (ON) and filter (WHERE) predicates."""
    start = re.search(r"\bFROM\b", sql, re.IGNORECASE)
    if not start:
        return set()
    region = sql[start.start():]
    end = _PRED_END_RE.search(region)
    region = region[:end.start()] if end else region
    return {m.group(1).lower() for m in _COL_RE.finditer(region)}


def _table_of(col: str) -> str | None:
    return next((t for prefix, t in _COL_PREFIX if col.startswith(prefix)), None)


def derive_b3_layout(sf: int = 1) -> dict[str, str]:
    """B3 keys: per-table independent optimum from the same log AJLO uses.

    Each table maximises its own benefit (query frequency × its own size; the
    size is constant per table, so the argmax is the most frequently
    predicated column). Join partners are not required to share a key.
    Ties break alphabetically for determinism.
    """
    from experiments.query_logs import generate_query_log

    start = datetime(2024, 1, 1)
    log = generate_query_log("C", n_queries=500, start_time=start, seed=0, sf=sf)
    counts: dict[str, Counter] = {t: Counter() for t in TABLES}
    for q in log:
        for col in _predicate_columns(q["sql"]):
            table = _table_of(col)
            if table in counts:
                counts[table][col] += 1

    keys = {}
    for table in TABLES:
        ranked = sorted(counts[table].items(), key=lambda kv: (-kv[1], kv[0]))
        keys[table] = ranked[0][0] if ranked else FILTER_COLS[table]
    return keys


def _derive_ajlo_layout(spark, sf: int = 1) -> dict[str, str]:
    """Run Algorithms 1→2→3 on the TPC-H query log to derive clustering keys.

    This is the only place the AJLO library makes real layout decisions.
    Returns {table_name: column_to_cluster_on}.
    """
    from ajlo.algorithms.join_graph import extract_join_graph
    from ajlo.algorithms.jis import compute_jis
    from ajlo.algorithms.layout_planner import recommend_layout
    from ajlo.models import LayoutPlan
    from experiments.query_logs import generate_query_log, get_cardinalities

    console.print("  [AJLO] Running Algorithm 1: extracting join graph from query log...")
    start = datetime(2024, 1, 1)
    window = (start, start + timedelta(days=7))

    # Generate a realistic mixed workload (regime C = all join patterns)
    log = generate_query_log("C", n_queries=500, start_time=start, seed=0, sf=sf)
    graph = extract_join_graph(log, window)
    console.print(f"  [AJLO] Algorithm 1: found {len(graph.edges)} join relationships "
                  f"across {len(graph.tables())} tables")

    console.print("  [AJLO] Running Algorithm 2: computing JIS scores...")
    graph = compute_jis(graph)
    for e in sorted(graph.edges, key=lambda x: x.jis, reverse=True)[:5]:
        console.print(f"    JIS={e.jis:.3f}  {e.table_i}.{e.col_i} ↔ "
                      f"{e.table_j}.{e.col_j}  (freq={e.frequency})")

    console.print("  [AJLO] Running Algorithm 3: recommending coordinated layout...")
    table_sizes = {t: n * 200 for t, n in get_cardinalities(sf).items()}
    plan = recommend_layout(
        graph,
        LayoutPlan(),
        rewrite_budget=sum(table_sizes.values()),   # allow full rewrite
        storage_budget=sum(table_sizes.values()) * 0.05,
        table_sizes=table_sizes,
    )

    # Extract recommended cluster column per table
    cluster_keys: dict[str, str] = {}
    for table in track(TABLES, description="Extracting cluster keys"):
        strat = plan.get(table)
        if strat.strategy_type == "CO_CLUSTER" and strat.cluster_cols:
            cluster_keys[table] = strat.cluster_cols[0]
        elif strat.strategy_type == "CO_PARTITION" and strat.partition_col:
            cluster_keys[table] = strat.partition_col
        else:
            # Fall back to most-filtered column for tables not in join graph
            cluster_keys[table] = FILTER_COLS[table]

    console.print("  [AJLO] Algorithm 3 recommended layout:")
    for table, col in cluster_keys.items():
        strat = plan.get(table).strategy_type
        console.print(f"    {table}: {strat} → CLUSTER BY {col}")

    return cluster_keys


def build_b3(spark, sf: int = 1) -> Path:
    """B3: per-table independent optimal clustering (no cross-table coordination)."""
    out = BASE_DIR / "b3"
    keys = derive_b3_layout(sf)
    console.print("Building B3 (per-table independent optimum)...")
    for table in track(TABLES, description="B3 tables"):
        df = _load_raw(spark, table)
        _write_clustered(spark, df, out / table, cluster_col=keys[table])
        console.print(f"  {table}: sorted by {keys[table]}  ← own most-predicated column")
    return out


def build_ajlo(spark, sf: int = 1) -> Path:
    """AJLO: Coordinated Liquid Clustering derived by Algorithms 1→2→3.

    The cluster column for every table comes from the AJLO library's
    join-graph analysis of the TPC-H query log — not from a hardcoded list.
    """
    out = BASE_DIR / "ajlo"
    console.print("Building AJLO (library-recommended coordinated co-clustering)...")

    ajlo_keys = _derive_ajlo_layout(spark, sf=sf)

    for table in track(TABLES, description="AJLO tables"):
        col = ajlo_keys[table]
        print(f"  {table}: clustering on {col}...")
        df = _load_raw(spark, table)
        _write_clustered(spark, df, out / table, cluster_col=col)
        console.print(f"  {table}: sorted by {col}  ← join-aware key from Algorithm 3")
    return out


BASELINES: dict[str, callable] = {
    "b0":   build_b0,
    "b1":   build_b1,
    "b2":   build_b2,
    "b3":   build_b3,
    "ajlo": build_ajlo,
}

_SF_AWARE = {"b3", "ajlo"}


def build_all(spark, sf: int = 1, only: list[str] | None = None) -> dict[str, Path]:
    """Build baseline and AJLO layouts (all, or the names in `only`). Returns {name: path}."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in BASELINES.items():
        if only and name not in only:
            continue
        results[name] = fn(spark, sf=sf) if name in _SF_AWARE else fn(spark)
    return results





@click.group()
def cli():
    """Build TPC-H baseline layouts for the AJLO experiment."""


@cli.command("build")
@click.argument("baseline", type=click.Choice(["b0", "b1", "b2", "b3", "ajlo", "all"]))
def cmd_build(baseline: str):
    """Build one or all baselines.

    BASELINE is one of: b0, b1, b2, ajlo, all
    """
    from experiments.spark_utils import get_spark_session
    spark = get_spark_session(app_name="ajlo-baselines")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if baseline == "all":
        results = build_all(spark)
        for name, path in results.items():
            console.print(f"[green]{name}[/green] → {_link(path)}")
    else:
        fn = BASELINES[baseline]
        path = fn(spark, sf=1) if baseline in _SF_AWARE else fn(spark)
        console.print(f"[green]{baseline}[/green] → {_link(path)}")


if __name__ == "__main__":
    cli()
