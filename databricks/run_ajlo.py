# Databricks notebook source
# MAGIC %md
# MAGIC # AJLO experiments on Databricks
# MAGIC
# MAGIC Runs the AJLO experiment suite (generate → layouts → RQ1/RQ2/RQ3/RQ5) on this cluster.
# MAGIC
# MAGIC 1. Set the widgets at the top (scale factor, Volume path, steps).
# MAGIC 2. **Run SF=1 first** as a smoke test, then SF=100.
# MAGIC
# MAGIC Data and results are written under `base_path/sf<SF>/`.

# COMMAND ----------

# MAGIC %pip install -q -r ./requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("sf", "1", "TPC-H scale factor")
dbutils.widgets.text("base_path", "/Volumes/workspace/ajlo/ajlo_data", "Base path (UC Volume)")
dbutils.widgets.text("steps", "generate,layouts,rq1,rq2,rq3,rq3phys,rq5,phi", "Steps (comma-separated)")
dbutils.widgets.text("optimize_file_mb", "128", "OPTIMIZE target file size (MB)")
dbutils.widgets.text("run_tag", "", "Run tag (suffix for layouts/results dirs)")
dbutils.widgets.text("layouts", "", "Layouts to build (blank = all; e.g. b2,ajlo)")
dbutils.widgets.text("results_tag", "", "Results dir suffix (blank = same as run_tag)")
dbutils.widgets.text("repeats", "5", "Timed runs per query")

SF        = int(dbutils.widgets.get("sf"))
BASE_PATH = dbutils.widgets.get("base_path").rstrip("/")
STEPS     = dbutils.widgets.get("steps").replace(" ", "")
FILE_MB   = dbutils.widgets.get("optimize_file_mb")
TAG       = dbutils.widgets.get("run_tag").strip()
SUFFIX    = f"_{TAG}" if TAG else ""
LAYOUTS   = dbutils.widgets.get("layouts").replace(" ", "")
RES_TAG   = dbutils.widgets.get("results_tag").strip()
RES_SUFFIX = f"_{RES_TAG}" if RES_TAG else SUFFIX
REPEATS   = dbutils.widgets.get("repeats").strip() or "5"

# COMMAND ----------

import os
import sys
from pathlib import Path

# Env vars MUST be set before any `experiments.*` import (config reads them at import time).
os.environ["AJLO_DATA_DIR"]             = f"{BASE_PATH}/sf{SF}/data"
os.environ["AJLO_LAYOUTS_DIR"]          = f"{BASE_PATH}/sf{SF}/data/layouts{SUFFIX}"
os.environ["AJLO_RESULTS_DIR"]          = f"{BASE_PATH}/sf{SF}/results{RES_SUFFIX}"
os.environ["AJLO_OPTIMIZE_MAX_FILE_MB"] = FILE_MB
os.environ["AJLO_QUERY_REPEATS"]        = REPEATS

# This notebook lives in <repo>/databricks/, so the repo root is one level up.
REPO_ROOT = str(Path(os.getcwd()).parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("Repo root :", REPO_ROOT)
print("Data dir  :", os.environ["AJLO_DATA_DIR"])
print("Layouts   :", os.environ["AJLO_LAYOUTS_DIR"])
print("Results   :", os.environ["AJLO_RESULTS_DIR"])
print("SF        :", SF, " steps:", STEPS, " file MB:", FILE_MB)

# COMMAND ----------

import json
from experiments.config import RESULTS_DIR, is_serverless, try_set_conf

# Turn off Databricks-only optimizations so results are closer to OSS Spark + Delta.
# AQE stays on because the local (OSS) runs use it too.
# Serverless only allows a few confs; whatever is rejected is recorded in environment.json.
# Write-side auto-optimizations are also disabled per table (see baselines._NO_AUTO_OPT_PROPS).
CONFS = {
    "spark.databricks.io.cache.enabled":                "false",  # SSD disk cache
    "spark.databricks.optimizer.dynamicFilePruning":    "false",  # DFP (not in OSS)
    "spark.databricks.delta.optimizeWrite.enabled":     "false",  # optimized writes
    "spark.databricks.delta.autoCompact.enabled":       "false",  # auto compaction
    "spark.databricks.photon.enabled":                  "false",  # cluster-level; usually rejected
    "spark.sql.adaptive.enabled":                       "true",
}
applied = {k: try_set_conf(spark, k, v) for k, v in CONFS.items()}
for k, ok in applied.items():
    print(f"{'set         ' if ok else 'NOT settable'}  {k}={CONFS[k]}")

env_info = {
    "dbr_version": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
    "serverless": is_serverless(spark),
    "sf": SF, "run_tag": TAG, "optimize_file_mb": FILE_MB,
    "confs_requested": CONFS, "confs_applied": applied,
}
(RESULTS_DIR / "environment.json").write_text(json.dumps(env_info, indent=2))
print("Wrote", RESULTS_DIR / "environment.json")

if is_serverless(spark):
    print("WARNING: serverless compute — executed-plan metrics are unavailable, so "
          "rq2 shuffle_bytes will be 0. Bytes/files skipped (Delta log) and wall time are still valid.")

# COMMAND ----------

# Sanity check: pure-Python unit tests (no Spark), < 10 s.
# Jobs cannot write __pycache__ into /Workspace: no bytecode, no assertion rewriting.
import pytest
sys.dont_write_bytecode = True
rc = pytest.main([f"{REPO_ROOT}/tests", "-q", "-p", "no:cacheprovider", "--assert=plain"])
assert rc == 0, "Unit tests failed — fix before running experiments"

# COMMAND ----------

from experiments.run_all import run

args = ["--only", STEPS, "--sf", str(SF)]
if LAYOUTS:
    args += ["--layouts", LAYOUTS]
run.main(args, standalone_mode=False)

# COMMAND ----------

import pandas as pd

results_dir = Path(os.environ["AJLO_RESULTS_DIR"])
for csv in sorted(results_dir.glob("*.csv")):
    print(f"=== {csv.name} ===")
    display(pd.read_csv(csv))
