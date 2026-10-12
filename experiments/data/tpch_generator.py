"""TPC-H data generator — pure Python / NumPy, no dbgen binary required.

Generates all 8 TPC-H tables and writes them as Parquet files via PySpark.
Row counts follow the official TPC-H specification, scaled by SF.

Usage:
    python experiments/data/tpch_generator.py [--sf N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, track

console = Console()

# Orders processed per Spark write at high SF (keeps Python heap bounded).
_CHUNK_SIZE = 500_000

from experiments.config import RAW_DIR, spark_path

# ── Value pools (TPC-H spec) ──────────────────────────────────────────────────
REGIONS = [
    (0, "AFRICA"),    (1, "AMERICA"),   (2, "ASIA"),
    (3, "EUROPE"),    (4, "MIDDLE EAST"),
]
NATIONS = [
    (0, "ALGERIA", 0),   (1, "ARGENTINA", 1),  (2, "BRAZIL", 1),
    (3, "CANADA", 1),    (4, "EGYPT", 4),       (5, "ETHIOPIA", 0),
    (6, "FRANCE", 3),    (7, "GERMANY", 3),     (8, "INDIA", 2),
    (9, "INDONESIA", 2), (10, "IRAN", 4),       (11, "IRAQ", 4),
    (12, "JAPAN", 2),    (13, "JORDAN", 4),     (14, "KENYA", 0),
    (15, "MOROCCO", 0),  (16, "MOZAMBIQUE", 0), (17, "PERU", 1),
    (18, "CHINA", 2),    (19, "ROMANIA", 3),    (20, "SAUDI ARABIA", 4),
    (21, "VIETNAM", 2),  (22, "RUSSIA", 3),     (23, "UNITED KINGDOM", 3),
    (24, "UNITED STATES", 1),
]
ORDER_STATUS   = ["F", "O", "P"]
ORDER_PRIORITY = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
MKTSEGMENT     = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
SHIP_MODE      = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"]
RETURN_FLAG    = ["N", "R", "A"]
LINE_STATUS    = ["O", "F"]
PART_TYPE_SYLLABLES = [
    ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"],
    ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"],
    ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"],
]
PART_CONTAINER = ["SM BOX", "LG BOX", "MED BAG", "SM PACK", "LG PACK",
                  "MED DRUM", "LG DRUM", "WRAP CASE", "JUMBO DRUM", "WRAP PAK"]


def _link(path) -> str:
    return f"[link=file://{path}]{path}[/link]"


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── Table generators ──────────────────────────────────────────────────────────

def gen_region() -> list[dict]:
    return [{"r_regionkey": r, "r_name": n, "r_comment": f"region {n.lower()}"}
            for r, n in REGIONS]


def gen_nation() -> list[dict]:
    return [{"n_nationkey": nk, "n_name": nm, "n_regionkey": rk,
             "n_comment": f"nation {nm.lower()}"}
            for nk, nm, rk in NATIONS]


def gen_supplier(rng: np.random.Generator, supplier_n: int, nation_n: int = 25) -> list[dict]:
    rows = []
    for sk in range(1, supplier_n + 1):
        rows.append({
            "s_suppkey":   sk,
            "s_name":      f"Supplier#{sk:09d}",
            "s_nationkey": int(rng.integers(0, nation_n)),
            "s_acctbal":   round(float(rng.uniform(-999.99, 9999.99)), 2),
            "s_phone":     f"{rng.integers(10,99)}-{rng.integers(100,999)}-{rng.integers(1000,9999)}",
        })
    return rows


def gen_part(rng: np.random.Generator, part_n: int) -> list[dict]:
    rows = []
    for pk in range(1, part_n + 1):
        syllables = [rng.choice(s) for s in PART_TYPE_SYLLABLES]
        rows.append({
            "p_partkey":     pk,
            "p_name":        f"part {pk}",
            "p_mfgr":        f"Manufacturer#{rng.integers(1, 6)}",
            "p_brand":       f"Brand#{rng.integers(1,5)}{rng.integers(1,5)}",
            "p_type":        " ".join(syllables),
            "p_size":        int(rng.integers(1, 51)),
            "p_container":   str(rng.choice(PART_CONTAINER)),
            "p_retailprice": round(900.0 + (pk / 10.0) % 200.0, 2),
        })
    return rows


def gen_partsupp(rng: np.random.Generator, part_n: int, supplier_n: int) -> list[dict]:
    rows = []
    for pk in range(1, part_n + 1):
        suppkeys = rng.choice(supplier_n, size=4, replace=False) + 1
        for sk in suppkeys:
            rows.append({
                "ps_partkey":    pk,
                "ps_suppkey":    int(sk),
                "ps_availqty":   int(rng.integers(1, 9999)),
                "ps_supplycost": round(float(rng.uniform(1.0, 1000.0)), 2),
            })
    return rows


def gen_customer(rng: np.random.Generator, customer_n: int, nation_n: int = 25) -> list[dict]:
    rows = []
    for ck in range(1, customer_n + 1):
        rows.append({
            "c_custkey":    ck,
            "c_name":       f"Customer#{ck:09d}",
            "c_nationkey":  int(rng.integers(0, nation_n)),
            "c_acctbal":    round(float(rng.uniform(-999.99, 9999.99)), 2),
            "c_mktsegment": str(rng.choice(MKTSEGMENT)),
            "c_phone":      f"{rng.integers(10,99)}-{rng.integers(100,999)}-{rng.integers(1000,9999)}",
        })
    return rows


def gen_orders(rng: np.random.Generator, orders_n: int, customer_n: int) -> list[dict]:
    rows = []
    base_date = np.datetime64("1992-01-01")
    for ok in range(1, orders_n + 1):
        order_date = base_date + int(rng.integers(0, 2557))
        rows.append({
            "o_orderkey":      ok,
            "o_custkey":       int(rng.integers(1, customer_n + 1)),
            "o_orderstatus":   str(rng.choice(ORDER_STATUS)),
            "o_totalprice":    round(float(rng.uniform(100.0, 500000.0)), 2),
            "o_orderdate":     str(order_date),
            "o_orderpriority": str(rng.choice(ORDER_PRIORITY)),
            "o_clerk":         f"Clerk#{rng.integers(1, 1001):09d}",
            "o_shippriority":  0,
        })
    return rows


def gen_lineitem(rng: np.random.Generator, orders: list[dict],
                 part_n: int, supplier_n: int) -> list[dict]:
    rows = []
    for order in orders:
        ok = order["o_orderkey"]
        n_lines = int(rng.integers(1, 8))
        order_date = np.datetime64(order["o_orderdate"])
        part_keys = rng.integers(1, part_n + 1, size=n_lines)
        supp_keys = rng.integers(1, supplier_n + 1, size=n_lines)
        for ln, (pk, sk) in enumerate(zip(part_keys, supp_keys), start=1):
            ship_date = order_date + int(rng.integers(1, 122))
            rows.append({
                "l_orderkey":      ok,
                "l_partkey":       int(pk),
                "l_suppkey":       int(sk),
                "l_linenumber":    ln,
                "l_quantity":      int(rng.integers(1, 51)),
                "l_extendedprice": round(float(rng.uniform(100.0, 100000.0)), 2),
                "l_discount":      round(float(rng.choice([0.00,0.01,0.02,0.03,
                                                            0.04,0.05,0.06,0.07,
                                                            0.08,0.09,0.10])), 2),
                "l_tax":           round(float(rng.choice([0.00,0.01,0.02,0.03,
                                                            0.04,0.05,0.06,0.07,
                                                            0.08])), 2),
                "l_returnflag":    str(rng.choice(RETURN_FLAG)),
                "l_linestatus":    str(rng.choice(LINE_STATUS)),
                "l_shipdate":      str(ship_date),
                "l_commitdate":    str(order_date + int(rng.integers(30, 91))),
                "l_receiptdate":   str(ship_date + int(rng.integers(1, 31))),
                "l_shipmode":      str(rng.choice(SHIP_MODE)),
            })
    return rows


# ── Writers ───────────────────────────────────────────────────────────────────

def write_table(spark, data: list[dict], path: Path, name: str) -> None:
    """Write a list of dicts as a Parquet table via PySpark."""
    if not data:
        console.print(f"  [SKIP] {name}: no rows")
        return
    console.print(f"  Writing {name}: {len(data):,} rows → {_link(path)}")
    path.mkdir(parents=True, exist_ok=True)
    df = spark.createDataFrame(data)
    df.write.mode("overwrite").parquet(spark_path(path))


def _write_chunked_stream(spark, stream, path: Path, name: str) -> int:
    """Write a row stream in chunks, returning total row count."""
    path.mkdir(parents=True, exist_ok=True)
    total = 0
    batch: list[dict] = []
    first = True
    for row in stream:
        batch.append(row)
        if len(batch) >= _CHUNK_SIZE:
            mode = "overwrite" if first else "append"
            spark.createDataFrame(batch).write.mode(mode).parquet(spark_path(path))
            total += len(batch)
            batch = []
            first = False
    if batch:
        mode = "overwrite" if first else "append"
        spark.createDataFrame(batch).write.mode(mode).parquet(spark_path(path))
        total += len(batch)
    return total


def _stream_partsupp(rng, part_n, supplier_n):
    for pk in range(1, part_n + 1):
        suppkeys = rng.choice(supplier_n, size=4, replace=False) + 1
        for sk in suppkeys:
            yield {
                "ps_partkey":    pk,
                "ps_suppkey":    int(sk),
                "ps_availqty":   int(rng.integers(1, 9999)),
                "ps_supplycost": round(float(rng.uniform(1.0, 1000.0)), 2),
            }


def _write_orders_lineitem_chunked(spark, rng, orders_n, customer_n,
                                   part_n, supplier_n, out_dir):
    """Generate and write orders + lineitem together in chunks."""
    orders_path   = out_dir / "orders"
    lineitem_path = out_dir / "lineitem"
    orders_path.mkdir(parents=True, exist_ok=True)
    lineitem_path.mkdir(parents=True, exist_ok=True)

    base_date = np.datetime64("1992-01-01")
    order_batch: list[dict] = []
    lineitem_batch: list[dict] = []
    total_orders = 0
    total_lineitems = 0
    first_write = True

    def _flush():
        nonlocal total_orders, total_lineitems, first_write
        mode = "overwrite" if first_write else "append"
        spark.createDataFrame(order_batch).write.mode(mode).parquet(spark_path(orders_path))
        spark.createDataFrame(lineitem_batch).write.mode(mode).parquet(spark_path(lineitem_path))
        total_orders += len(order_batch)
        total_lineitems += len(lineitem_batch)
        order_batch.clear()
        lineitem_batch.clear()
        first_write = False

    with Progress(console=console) as progress:
        task = progress.add_task("  orders+lineitem", total=orders_n)
        for ok in range(1, orders_n + 1):
            order_date = base_date + int(rng.integers(0, 2557))
            order = {
                "o_orderkey":      ok,
                "o_custkey":       int(rng.integers(1, customer_n + 1)),
                "o_orderstatus":   str(rng.choice(ORDER_STATUS)),
                "o_totalprice":    round(float(rng.uniform(100.0, 500000.0)), 2),
                "o_orderdate":     str(order_date),
                "o_orderpriority": str(rng.choice(ORDER_PRIORITY)),
                "o_clerk":         f"Clerk#{rng.integers(1, 1001):09d}",
                "o_shippriority":  0,
            }
            order_batch.append(order)

            n_lines = int(rng.integers(1, 8))
            part_keys = rng.integers(1, part_n + 1, size=n_lines)
            supp_keys = rng.integers(1, supplier_n + 1, size=n_lines)
            for ln, (pk, sk) in enumerate(zip(part_keys, supp_keys), start=1):
                ship_date = order_date + int(rng.integers(1, 122))
                lineitem_batch.append({
                    "l_orderkey":      ok,
                    "l_partkey":       int(pk),
                    "l_suppkey":       int(sk),
                    "l_linenumber":    ln,
                    "l_quantity":      int(rng.integers(1, 51)),
                    "l_extendedprice": round(float(rng.uniform(100.0, 100000.0)), 2),
                    "l_discount":      round(float(rng.choice([0.00,0.01,0.02,0.03,
                                                                0.04,0.05,0.06,0.07,
                                                                0.08,0.09,0.10])), 2),
                    "l_tax":           round(float(rng.choice([0.00,0.01,0.02,0.03,
                                                                0.04,0.05,0.06,0.07,
                                                                0.08])), 2),
                    "l_returnflag":    str(rng.choice(RETURN_FLAG)),
                    "l_linestatus":    str(rng.choice(LINE_STATUS)),
                    "l_shipdate":      str(ship_date),
                    "l_commitdate":    str(order_date + int(rng.integers(30, 91))),
                    "l_receiptdate":   str(ship_date + int(rng.integers(1, 31))),
                    "l_shipmode":      str(rng.choice(SHIP_MODE)),
                })

            if len(order_batch) >= _CHUNK_SIZE:
                _flush()
            progress.advance(task)

    if order_batch:
        _flush()

    console.print(f"  orders: {total_orders:,} rows, lineitem: {total_lineitems:,} rows")


# ── Main entry point ──────────────────────────────────────────────────────────

def generate(out_dir: Path | None = None, spark=None, sf: int = 1) -> Path:
    """Generate all TPC-H tables at the given scale factor.

    Returns the output directory path.
    """
    from experiments.spark_utils import get_spark_session

    out_dir = out_dir or RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if spark is None:
        spark = get_spark_session("tpch-generator")

    nation_n   = 25
    supplier_n = 10_000   * sf
    part_n     = 200_000  * sf
    customer_n = 150_000  * sf
    orders_n   = 1_500_000 * sf

    console.print(f"Generating TPC-H SF={sf} → {_link(out_dir)}")
    rng = _rng(42)

    # Small / medium tables — generate in memory
    write_table(spark, gen_region(),  out_dir / "region",   "region")
    write_table(spark, gen_nation(),  out_dir / "nation",   "nation")
    write_table(spark, gen_supplier(rng, supplier_n, nation_n),
                out_dir / "supplier", "supplier")

    if part_n <= 1_000_000:
        write_table(spark, gen_part(rng, part_n),     out_dir / "part",     "part")
        write_table(spark, gen_customer(rng, customer_n, nation_n),
                    out_dir / "customer", "customer")
        write_table(spark, gen_partsupp(rng, part_n, supplier_n),
                    out_dir / "partsupp", "partsupp")
    else:
        # Chunked for large part_n (SF > 5)
        console.print("  Generating part (chunked) ...")
        def _part_stream():
            for pk in range(1, part_n + 1):
                syllables = [rng.choice(s) for s in PART_TYPE_SYLLABLES]
                yield {
                    "p_partkey":     pk,
                    "p_name":        f"part {pk}",
                    "p_mfgr":        f"Manufacturer#{rng.integers(1, 6)}",
                    "p_brand":       f"Brand#{rng.integers(1,5)}{rng.integers(1,5)}",
                    "p_type":        " ".join(syllables),
                    "p_size":        int(rng.integers(1, 51)),
                    "p_container":   str(rng.choice(PART_CONTAINER)),
                    "p_retailprice": round(900.0 + (pk / 10.0) % 200.0, 2),
                }
        n = _write_chunked_stream(spark, _part_stream(), out_dir / "part", "part")
        console.print(f"  part: {n:,} rows")

        console.print("  Generating customer (chunked) ...")
        def _customer_stream():
            for ck in range(1, customer_n + 1):
                yield {
                    "c_custkey":    ck,
                    "c_name":       f"Customer#{ck:09d}",
                    "c_nationkey":  int(rng.integers(0, nation_n)),
                    "c_acctbal":    round(float(rng.uniform(-999.99, 9999.99)), 2),
                    "c_mktsegment": str(rng.choice(MKTSEGMENT)),
                    "c_phone":      f"{rng.integers(10,99)}-{rng.integers(100,999)}-{rng.integers(1000,9999)}",
                }
        n = _write_chunked_stream(spark, _customer_stream(), out_dir / "customer", "customer")
        console.print(f"  customer: {n:,} rows")

        console.print("  Generating partsupp (chunked) ...")
        n = _write_chunked_stream(spark, _stream_partsupp(rng, part_n, supplier_n),
                                  out_dir / "partsupp", "partsupp")
        console.print(f"  partsupp: {n:,} rows")

    # Orders + lineitem always chunked at high SF, in-memory at SF=1
    if orders_n <= 1_500_000:
        console.print("  Generating orders ...")
        orders = gen_orders(rng, orders_n, customer_n)
        write_table(spark, orders, out_dir / "orders", "orders")
        console.print("  Generating lineitem ...")
        lineitems = gen_lineitem(rng, orders, part_n, supplier_n)
        write_table(spark, lineitems, out_dir / "lineitem", "lineitem")
    else:
        console.print("  Generating orders + lineitem (chunked) ...")
        _write_orders_lineitem_chunked(spark, rng, orders_n, customer_n,
                                       part_n, supplier_n, out_dir)

    console.print("Done. All tables written.")
    return out_dir


import click


@click.command()
@click.option("--sf", default=1, show_default=True, type=int,
              help="TPC-H scale factor (1 ≈ 1 GB, 30 ≈ 30 GB, 100 ≈ 100 GB).")
@click.option("--out-dir", default=None, type=click.Path(path_type=Path),
              help="Output directory (default: data/tpch/raw).")
def cli(sf: int, out_dir: Path | None) -> None:
    """Generate TPC-H Parquet data at the given scale factor."""
    generate(out_dir=out_dir, sf=sf)


if __name__ == "__main__":
    cli()
