"""Runtime configuration shared by all experiment modules.

Defaults reproduce the local (laptop) setup. On Databricks, the driver notebook
sets these environment variables BEFORE importing any experiment module:

    AJLO_DATA_DIR              root for raw + layout data (e.g. /Volumes/main/ajlo/data/sf100)
    AJLO_RESULTS_DIR           where CSVs and charts are written
    AJLO_LAYOUTS_DIR           layout tables (default: AJLO_DATA_DIR/layouts)
    AJLO_OPTIMIZE_MAX_FILE_MB  target file size for OPTIMIZE (default 2 MB locally)
    AJLO_QUERY_WARMUP          untimed warm-up runs per query (default 1)
    AJLO_QUERY_REPEATS         timed runs per query (default 5)

Paths are POSIX paths readable by Python (pathlib). On Databricks use a
Unity Catalog Volume (/Volumes/...) — the same string works for Spark and for
Python file access. DBFS FUSE paths (/dbfs/...) also work; spark_path()
translates them to dbfs:/ for Spark.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

DATA_DIR    = Path(os.environ.get("AJLO_DATA_DIR", _REPO_ROOT / "data" / "tpch"))
RAW_DIR     = DATA_DIR / "raw"
LAYOUTS_DIR = Path(os.environ.get("AJLO_LAYOUTS_DIR", DATA_DIR / "layouts"))
RESULTS_DIR = Path(os.environ.get("AJLO_RESULTS_DIR", _REPO_ROOT / "results"))

OPTIMIZE_MAX_FILE_SIZE = int(float(os.environ.get("AJLO_OPTIMIZE_MAX_FILE_MB", "2")) * 1024 * 1024)

# Timed queries: discarded warm-up runs, then measured repeats (mean ± std reported).
QUERY_WARMUP  = int(os.environ.get("AJLO_QUERY_WARMUP", "1"))
QUERY_REPEATS = int(os.environ.get("AJLO_QUERY_REPEATS", "5"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "eps").mkdir(parents=True, exist_ok=True)


def try_set_conf(spark, key: str, value: str) -> bool:
    """Set a Spark conf; return False where it is not settable (e.g. Databricks serverless)."""
    try:
        spark.conf.set(key, value)
        return True
    except Exception:
        return False


def is_serverless(spark) -> bool:
    """True for Spark Connect sessions (Databricks serverless): no JVM plan access."""
    return not hasattr(spark, "_jsparkSession")


def spark_path(path) -> str:
    """Convert a Python-visible path to the form Spark expects."""
    s = str(path)
    if s.startswith("/dbfs/"):
        return "dbfs:" + s[len("/dbfs"):]
    return s
