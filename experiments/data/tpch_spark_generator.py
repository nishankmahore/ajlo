"""Distributed TPC-H generator — Spark-native, for large scale factors (SF ≥ 10).

The pure-Python generator (tpch_generator.py) builds every row on the driver,
which is impractical at SF=100 (600M lineitem rows). This module produces the
same 8 tables with the same column names, types and value distributions, but
every row is computed on the executors with spark.range + column expressions.

Values are not byte-identical to tpch_generator.py (different RNG), only the
schema and distributions match.

Usage (on a cluster):
    from experiments.data.tpch_spark_generator import generate
    generate(spark, sf=100)
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from rich.console import Console

from experiments.config import RAW_DIR, spark_path
from experiments.data.tpch_generator import (
    LINE_STATUS, MKTSEGMENT, NATIONS, ORDER_PRIORITY, ORDER_STATUS,
    PART_CONTAINER, PART_TYPE_SYLLABLES, REGIONS, RETURN_FLAG, SHIP_MODE,
)

console = Console()

_BASE_DATE = "1992-01-01"


def _randint(lo: int, hi: int, seed: int) -> Column:
    """Uniform integer in [lo, hi) — mirrors numpy rng.integers(lo, hi)."""
    return (F.floor(F.rand(seed) * (hi - lo)) + lo).cast("long")


def _uniform(lo: float, hi: float, seed: int) -> Column:
    return F.round(F.rand(seed) * (hi - lo) + lo, 2)


def _choice(values: list, seed: int) -> Column:
    arr = F.array(*[F.lit(v) for v in values])
    return F.element_at(arr, (F.floor(F.rand(seed) * len(values)) + 1).cast("int"))


def _phone(seed: int) -> Column:
    return F.concat_ws("-", _randint(10, 99, seed), _randint(100, 999, seed + 1),
                       _randint(1000, 9999, seed + 2))


def _date_plus(base: Column, lo: int, hi: int, seed: int) -> Column:
    return F.date_add(base, _randint(lo, hi, seed).cast("int"))


def _keys(spark, n: int, name: str, partitions: int) -> DataFrame:
    return spark.range(1, n + 1, numPartitions=partitions).withColumnRenamed("id", name)


def _write(df: DataFrame, path: Path, name: str) -> None:
    console.print(f"  Writing {name} → {path}")
    df.write.mode("overwrite").parquet(spark_path(path))


# ── Tables ────────────────────────────────────────────────────────────────────

def _region(spark) -> DataFrame:
    rows = [(r, n, f"region {n.lower()}") for r, n in REGIONS]
    return spark.createDataFrame(rows, "r_regionkey long, r_name string, r_comment string")


def _nation(spark) -> DataFrame:
    rows = [(nk, nm, rk, f"nation {nm.lower()}") for nk, nm, rk in NATIONS]
    return spark.createDataFrame(
        rows, "n_nationkey long, n_name string, n_regionkey long, n_comment string")


def _supplier(spark, n: int, parts: int) -> DataFrame:
    return _keys(spark, n, "s_suppkey", parts).select(
        "s_suppkey",
        F.format_string("Supplier#%09d", "s_suppkey").alias("s_name"),
        _randint(0, 25, 101).alias("s_nationkey"),
        _uniform(-999.99, 9999.99, 102).alias("s_acctbal"),
        _phone(103).alias("s_phone"),
    )


def _part(spark, n: int, parts: int) -> DataFrame:
    p_type = F.concat_ws(" ", *[_choice(s, 210 + i) for i, s in enumerate(PART_TYPE_SYLLABLES)])
    return _keys(spark, n, "p_partkey", parts).select(
        "p_partkey",
        F.concat(F.lit("part "), F.col("p_partkey").cast("string")).alias("p_name"),
        F.concat(F.lit("Manufacturer#"), _randint(1, 6, 201)).alias("p_mfgr"),
        F.concat(F.lit("Brand#"), _randint(1, 5, 202), _randint(1, 5, 203)).alias("p_brand"),
        p_type.alias("p_type"),
        _randint(1, 51, 204).alias("p_size"),
        _choice(PART_CONTAINER, 205).alias("p_container"),
        F.round(F.lit(900.0) + (F.col("p_partkey") / 10.0) % 200.0, 2).alias("p_retailprice"),
    )


def _partsupp(spark, part_n: int, supplier_n: int, parts: int) -> DataFrame:
    # 4 distinct suppliers per part — TPC-H spec formula (Clause 4.2.3).
    s = supplier_n
    pk = F.col("ps_partkey")
    suppkey = ((pk + F.col("i") * (F.lit(s // 4) + F.floor((pk - 1) / s))) % s + 1).cast("long")
    return (
        _keys(spark, part_n, "ps_partkey", parts)
        .withColumn("i", F.explode(F.sequence(F.lit(0), F.lit(3))))
        .select(
            "ps_partkey",
            suppkey.alias("ps_suppkey"),
            _randint(1, 9999, 301).alias("ps_availqty"),
            _uniform(1.0, 1000.0, 302).alias("ps_supplycost"),
        )
    )


def _customer(spark, n: int, parts: int) -> DataFrame:
    return _keys(spark, n, "c_custkey", parts).select(
        "c_custkey",
        F.format_string("Customer#%09d", "c_custkey").alias("c_name"),
        _randint(0, 25, 401).alias("c_nationkey"),
        _uniform(-999.99, 9999.99, 402).alias("c_acctbal"),
        _choice(MKTSEGMENT, 403).alias("c_mktsegment"),
        _phone(404).alias("c_phone"),
    )


def _orders(spark, n: int, customer_n: int, parts: int) -> DataFrame:
    order_date = _date_plus(F.to_date(F.lit(_BASE_DATE)), 0, 2557, 502)
    return _keys(spark, n, "o_orderkey", parts).select(
        "o_orderkey",
        _randint(1, customer_n + 1, 501).alias("o_custkey"),
        _choice(ORDER_STATUS, 503).alias("o_orderstatus"),
        _uniform(100.0, 500000.0, 504).alias("o_totalprice"),
        order_date.cast("string").alias("o_orderdate"),
        _choice(ORDER_PRIORITY, 505).alias("o_orderpriority"),
        F.format_string("Clerk#%09d", _randint(1, 1001, 506)).alias("o_clerk"),
        F.lit(0).cast("long").alias("o_shippriority"),
    )


def _lineitem(orders: DataFrame, part_n: int, supplier_n: int) -> DataFrame:
    """Derive lineitem from the *persisted* orders table (deterministic dates)."""
    order_date = F.to_date("o_orderdate")
    exploded = (
        orders.select("o_orderkey", "o_orderdate")
        .withColumn("n_lines", _randint(1, 8, 601))
        .withColumn("l_linenumber", F.explode(F.sequence(F.lit(1), F.col("n_lines").cast("int"))))
        .withColumn("_ship", _date_plus(order_date, 1, 122, 602))
    )
    return exploded.select(
        F.col("o_orderkey").alias("l_orderkey"),
        _randint(1, part_n + 1, 603).alias("l_partkey"),
        _randint(1, supplier_n + 1, 604).alias("l_suppkey"),
        F.col("l_linenumber").cast("long").alias("l_linenumber"),
        _randint(1, 51, 605).alias("l_quantity"),
        _uniform(100.0, 100000.0, 606).alias("l_extendedprice"),
        F.round(_randint(0, 11, 607) / 100.0, 2).alias("l_discount"),
        F.round(_randint(0, 9, 608) / 100.0, 2).alias("l_tax"),
        _choice(RETURN_FLAG, 609).alias("l_returnflag"),
        _choice(LINE_STATUS, 610).alias("l_linestatus"),
        F.col("_ship").cast("string").alias("l_shipdate"),
        _date_plus(order_date, 30, 91, 611).cast("string").alias("l_commitdate"),
        _date_plus(F.col("_ship"), 1, 31, 612).cast("string").alias("l_receiptdate"),
        _choice(SHIP_MODE, 613).alias("l_shipmode"),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def generate(spark, sf: int = 1, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    supplier_n = 10_000    * sf
    part_n     = 200_000   * sf
    customer_n = 150_000   * sf
    orders_n   = 1_500_000 * sf

    # ~1 partition per 750k orders → ~128 MB lineitem files at SF=100.
    big = max(8, sf * 2)
    mid = max(4, sf // 2)

    console.print(f"Generating TPC-H SF={sf} (distributed) → {out_dir}")
    _write(_region(spark), out_dir / "region", "region")
    _write(_nation(spark), out_dir / "nation", "nation")
    _write(_supplier(spark, supplier_n, max(1, sf // 10)), out_dir / "supplier", "supplier")
    _write(_part(spark, part_n, mid), out_dir / "part", "part")
    _write(_partsupp(spark, part_n, supplier_n, mid), out_dir / "partsupp", "partsupp")
    _write(_customer(spark, customer_n, mid), out_dir / "customer", "customer")
    _write(_orders(spark, orders_n, customer_n, big), out_dir / "orders", "orders")

    orders = spark.read.parquet(spark_path(out_dir / "orders"))
    _write(_lineitem(orders, part_n, supplier_n), out_dir / "lineitem", "lineitem")

    for t in ["region", "nation", "supplier", "part", "partsupp", "customer", "orders", "lineitem"]:
        n = spark.read.parquet(spark_path(out_dir / t)).count()
        console.print(f"  {t:9s}: {n:,} rows")
    console.print("Done. All tables written.")
    return out_dir
