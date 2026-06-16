# AJLO Experiment Suite

Experimental evidence for the paper **"Adaptive Join-Aware Physical Layout Optimization for Lakehouse Systems"**.

This repository implements all five algorithms from the paper and a full experiment harness covering RQ1–RQ5.

---

## Architecture

```
ajlo/                    ← Core library (pure Python, no Spark)
  models.py              ← JoinEdge, JoinGraph, LayoutPlan, LayoutStrategy
  cost_model.py          ← NPV, ScanReduction, RewriteCost, SkewPenalty
  algorithms/
    join_graph.py        ← Algorithm 1: Join Graph Extraction
    jis.py               ← Algorithm 2: JIS Computation
    layout_planner.py    ← Algorithm 3: Dynamic Greedy Layout Recommendation
    drift_detector.py    ← Algorithm 4: Workload Drift Detection
    adaptive_reopt.py    ← Algorithm 5: Adaptive Re-optimization

experiments/
  data/tpch_generator.py ← TPC-H SF=1 Parquet generation (no dbgen needed)
  spark_utils.py         ← PySpark session + query metrics collection
  baselines.py           ← B0–B5 Delta Lake layout builders
  workload.py            ← Synthetic TPC-H query log generator (3 regimes)
  rq1_scan_reduction.py  ← RQ1: bytes scanned per layout
  rq2_file_skipping.py   ← RQ2: file skip ratio + shuffle bytes
  rq3_drift.py           ← RQ3: drift detection across workload regimes
  rq4_vs_manual.py       ← RQ4: AJLO vs manual co-partitioning (B5)
  rq5_approximation.py   ← RQ5: greedy approximation quality vs optimal

results/                 ← Auto-generated CSVs and PNG charts
tests/
  test_algorithms.py     ← 29 unit tests (no Spark needed)
```

---

## Setup

**Requirements:** macOS (M1/M2/M3), 16 GB RAM, Java 11+ (for PySpark), Python 3.11+

### 1. Install Java

PySpark requires Java 11 or 17.

```bash
# Check if Java is installed
java -version

# Install via Homebrew if needed
brew install openjdk@17
export JAVA_HOME=$(brew --prefix openjdk@17)
```

### 2. Create environment and install dependencies

```bash
# From the project root
uv venv
uv pip install pyspark "delta-spark>=3.2" numpy networkx matplotlib pandas faker pytest
```

### 3. Run unit tests (no Spark needed)

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: **29 passed**

---

## Running Experiments

### Run everything (all RQs)

```bash
python experiments/run_all.py
```

### Run individual steps

```bash
# Generate TPC-H data only
python experiments/run_all.py --only generate

# Build layout baselines only (requires data)
python experiments/run_all.py --only layouts

# Skip data generation (if data already exists)
python experiments/run_all.py --skip-data

# Run only RQ3 and RQ5 (pure algorithm, no Spark — fast, < 1 min)
python experiments/run_all.py --only rq3,rq5

# Run a single experiment directly
python experiments/rq5_approximation.py
python experiments/rq3_drift.py
```

---

## Outputs

All results are written to `results/`:

| File | Content |
|------|---------|
| `rq1.csv` | Bytes read, wall time per query per layout |
| `rq1_bytes_scanned.png` | Bar chart: mean bytes scanned by layout |
| `rq1_query_latency.png` | Bar chart: mean query latency by layout |
| `rq2.csv` | File skip ratio, shuffle bytes per join-heavy query |
| `rq2_skip_ratio.png` | Bar chart: mean file skip ratio by layout |
| `rq2_shuffle_bytes.png` | Bar chart: mean shuffle bytes by layout |
| `rq3.csv` | Drift score per window, detection result, reorg ops |
| `rq3_drift_score.png` | Line chart: drift score over windows with threshold |
| `rq3_reorg_cost.png` | Bar chart: rewrite cost vs scan reduction per window |
| `rq4.csv` | Scan reduction %, NPV estimates: AJLO vs B5 |
| `rq4_scan_reduction_pct.png` | Grouped bars: scan reduction per query |
| `rq4_benefit_cost.png` | NPV benefit vs rewrite cost: AJLO vs B5 |
| `rq5.csv` | Approximation ratios across 9 graph configurations |
| `rq5_approx_ratio.png` | Bar chart: greedy/optimal ratio with (1-1/e) bound |
| `rq5_npv_comparison.png` | Grouped bars: optimal vs greedy NPV |

---

## Baselines

| ID | Description |
|----|-------------|
| B0 | Raw Delta, no layout directives |
| B1 | ~~Per-table partitioning on most-filtered column~~ |
| B3 | ~~Z-order on most-used columns (per-table, no coordination)~~ |
| B4 | Liquid Clustering on filter columns (ALC proxy) |
| B5 | Manual co-partitioning on highest-cardinality FK (DBA baseline) |
| AJLO | Coordinated co-clustering via Algorithm 3 |

> B2 (bucketing) excluded — open-source Delta Lake does not store Hive-style bucket metadata that satisfies Catalyst's `HashClusteredDistribution` requirement (paper Section 2.2).

---

## Known Constraints

- **Scale:** TPC-H SF=1 (~1 GB). SF=10 risks OOM on 16 GB M1 due to Spark JVM overhead (~4–6 GB).
- **Liquid Clustering does not eliminate shuffles.** AJLO's benefit comes entirely from reducing bytes entering the Exchange node via pre-shuffle data skipping — not from eliminating the shuffle itself.
- **φ_scan values** (0.3 co-cluster, 0.2 co-partition) are estimates calibrated from simulation. Actual measured values from RQ1 will appear in `rq1.csv`.


