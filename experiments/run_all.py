"""Master experiment runner.

Run everything:
    .venv/bin/python experiments/run_all.py run

Run one step:
    .venv/bin/python experiments/run_all.py run --only rq1,rq2

Skip data generation:
    .venv/bin/python experiments/run_all.py run --skip-data

Steps (in order):
    generate  — generate TPC-H SF=1 Parquet data
    layouts   — build B0, B1, B2, AJLO Delta Lake layouts
    rq1       — bytes scanned comparison (B0 / B2 / AJLO)
    rq2       — file skip ratio + shuffle bytes
    rq3       — workload drift detection (no Spark needed)
    rq5       — greedy approximation quality (no Spark needed)
"""
from __future__ import annotations

import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import track
from rich.rule import Rule

console = Console()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALL_STEPS = ["generate", "layouts", "rq1", "rq2", "rq3", "rq5"]
SPARK_STEPS = {"generate", "layouts", "rq1", "rq2"}


def _link(path) -> str:
    return f"[link=file://{path}]{path}[/link]"


def _print_header(title: str) -> None:
    console.print(Rule(title))


# ── Step functions ────────────────────────────────────────────────────────────

def step_generate(spark, sf: int = 1) -> None:
    _print_header(f"Step 1/6 — Generate TPC-H SF={sf} data")
    from experiments.data.tpch_generator import generate
    generate(spark=spark, sf=sf)


def step_layouts(spark, sf: int = 1) -> None:
    _print_header("Step 2/6 — Build layout baselines (B0, B1, B2, AJLO)")
    from experiments.baselines import build_all
    build_all(spark, sf=sf)


def step_rq1(spark, sf: int = 1) -> None:
    _print_header("Step 3/6 — RQ1: Bytes Scanned")
    from experiments.rq1_scan_reduction import run
    rows = run(spark, sf=sf)
    _summarise_rq1(rows)


def step_rq2(spark, sf: int = 1) -> None:
    _print_header("Step 4/6 — RQ2: File Skip Ratio + Shuffle Bytes")
    from experiments.rq2_file_skipping import run
    run(spark, sf=sf)


def step_rq3(sf: int = 1) -> None:
    _print_header("Step 5/6 — RQ3: Workload Drift (algorithm-only, no Spark)")
    from experiments.rq3_drift import run
    run(sf=sf)


def step_rq5() -> None:
    _print_header("Step 6/6 — RQ5: Greedy Approximation Quality (no Spark)")
    from experiments.rq5_approximation import run
    rows = run()
    _summarise_rq5(rows)


# ── Summary printers ──────────────────────────────────────────────────────────

def _summarise_rq1(rows: list[dict]) -> None:
    if not rows:
        return
    from collections import defaultdict
    import numpy as np
    totals: dict[str, list] = defaultdict(list)
    for r in rows:
        totals[r["layout"]].append(r["bytes_read"])
    console.print("\n  RQ1 Summary — Mean bytes scanned per query:")
    b0_mean = float(np.mean(totals.get("b0", [1])))
    for layout, vals in sorted(totals.items()):
        mean = float(np.mean(vals))
        pct  = (1 - mean / b0_mean) * 100 if b0_mean > 0 else 0
        console.print(f"    {layout:6s}: {mean/1e6:8.1f} MB  ({pct:+.1f}% vs B0)")


def _summarise_rq5(rows: list[dict]) -> None:
    import numpy as np
    ratios = [r["approx_ratio"] for r in rows]
    above  = sum(1 for r in rows if r["above_bound"])
    console.print(f"\n  RQ5 Summary — Greedy approximation ratio:")
    console.print(f"    Mean ratio:  {np.mean(ratios):.3f}")
    console.print(f"    Min ratio:   {min(ratios):.3f}")
    console.print(f"    Max ratio:   {max(ratios):.3f}")
    console.print(f"    Above (1-1/e)≈0.632: {above}/{len(rows)} configs")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """AJLO experiment runner."""


@cli.command()
@click.option(
    "--only", "only",
    default=None,
    metavar="STEP[,STEP...]",
    help=f"Run only these steps (comma-separated). Available: {', '.join(ALL_STEPS)}",
)
@click.option(
    "--skip-data", "skip_data",
    is_flag=True,
    default=False,
    help="Skip generate + layouts (assume data already exists).",
)
@click.option(
    "--sf", "sf",
    default=1,
    type=int,
    show_default=True,
    help="TPC-H scale factor (1 ≈ 1 GB, 30 ≈ 30 GB, 100 ≈ 100 GB).",
)
def run(only: str | None, skip_data: bool, sf: int) -> None:
    """Run experiment steps."""
    if only:
        steps = [s.strip() for s in only.split(",")]
    elif skip_data:
        steps = ["rq1", "rq2", "rq3", "rq5"]
    else:
        steps = ALL_STEPS

    invalid = [s for s in steps if s not in ALL_STEPS]
    if invalid:
        raise click.BadParameter(
            f"Unknown steps: {invalid}. Available: {ALL_STEPS}", param_hint="--only"
        )

    t_start = time.time()

    spark = None
    if any(s in SPARK_STEPS for s in steps):
        console.print("Starting Spark session (this takes ~30 seconds the first time)...")
        from experiments.spark_utils import get_spark_session
        spark = get_spark_session("ajlo-experiments")
        console.print("Spark ready.")

    dispatch = {
        "generate": lambda: step_generate(spark, sf=sf),
        "layouts":  lambda: step_layouts(spark, sf=sf),
        "rq1":      lambda: step_rq1(spark, sf=sf),
        "rq2":      lambda: step_rq2(spark, sf=sf),
        "rq3":      lambda: step_rq3(sf=sf),
        "rq5":      step_rq5,
    }

    for step in track(steps, description="Experiment steps"):
        t0 = time.time()
        dispatch[step]()
        console.print(f"  ✓ {step} completed in {time.time()-t0:.1f}s")

    elapsed = time.time() - t_start
    console.print(Rule())
    console.print(f"  All steps complete in {elapsed:.1f}s")
    console.print(f"  Results in: {_link(RESULTS_DIR)}")
    console.print(f"  CSV files:  {', '.join(p.name for p in RESULTS_DIR.glob('*.csv'))}")
    console.print(f"  Charts:     {', '.join(p.name for p in RESULTS_DIR.glob('*.png'))}")
    console.print(Rule())


if __name__ == "__main__":
    cli()
