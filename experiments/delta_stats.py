"""Read Delta Lake transaction log to compute real data-skipping metrics.

Delta stores min/max statistics per Parquet file in the transaction log.
This module parses those stats and evaluates predicates against them —
the same logic Delta's FileScan uses to skip files at query time.

No Spark needed. Pure Python.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_add_entries(delta_log_dir: Path) -> list[dict]:
    """Return all live 'add' file entries from the Delta transaction log."""
    log_dir = delta_log_dir / "_delta_log"
    if not log_dir.exists():
        return []

    # Collect all JSON log files (checkpoints not handled — fine for SF=1)
    active: dict[str, dict] = {}
    for log_file in sorted(log_dir.glob("*.json")):
        for line in log_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "add" in entry:
                path = entry["add"]["path"]
                active[path] = entry["add"]
            elif "remove" in entry:
                active.pop(entry["remove"]["path"], None)

    return list(active.values())


def _eval_predicate(stats: dict, col: str, pred_min: Any, pred_max: Any) -> bool:
    """Return True if a file MAY contain rows satisfying col BETWEEN pred_min AND pred_max.

    A file can be SKIPPED only if its max < pred_min OR its min > pred_max.
    Otherwise the file must be read (conservative — same as Delta's logic).
    """
    min_vals = stats.get("minValues", {})
    max_vals = stats.get("maxValues", {})

    if col not in min_vals or col not in max_vals:
        return True  # no stats → must read

    file_min = min_vals[col]
    file_max = max_vals[col]

    # Compare as strings for date/string columns, as numbers for numeric
    try:
        if isinstance(pred_min, (int, float)):
            file_min = type(pred_min)(file_min)
            file_max = type(pred_max)(file_max)
    except (ValueError, TypeError):
        pass

    # Skip if entire file is outside predicate range
    if file_max < pred_min or file_min > pred_max:
        return False  # can skip
    return True  # must read


def bytes_read_with_predicate(
    table_path: Path,
    filter_col: str,
    pred_min: Any,
    pred_max: Any,
) -> tuple[int, int, int]:
    """Compute bytes read for a BETWEEN predicate using Delta min/max stats.

    Returns (bytes_read, files_read, files_total).
    All bytes are actual Parquet file sizes on disk.
    """
    entries = _parse_add_entries(table_path)
    if not entries:
        return 0, 0, 0

    bytes_read = 0
    files_read = 0
    files_total = len(entries)

    for entry in entries:
        raw_stats = entry.get("stats")
        if isinstance(raw_stats, str):
            try:
                stats = json.loads(raw_stats)
            except json.JSONDecodeError:
                stats = {}
        elif isinstance(raw_stats, dict):
            stats = raw_stats
        else:
            stats = {}

        file_size = entry.get("size", 0)
        must_read = _eval_predicate(stats, filter_col, pred_min, pred_max)
        if must_read:
            bytes_read += file_size
            files_read += 1

    return bytes_read, files_read, files_total


def total_table_bytes(table_path: Path) -> tuple[int, int]:
    """Return (total_bytes, file_count) for a Delta table (all live files)."""
    entries = _parse_add_entries(table_path)
    total = sum(e.get("size", 0) for e in entries)
    return total, len(entries)


# ── Hive-style partition measurement ─────────────────────────────────────────

# Maps a query filter column to the Hive partition directory column.
# Date columns are bucketed to year-month; others map 1:1.
# Columns absent from this mapping (e.g. o_orderkey) have no matching
# partition column → Hive must perform a full scan for those predicates.
_HIVE_FILTER_TO_PART: dict[str, str] = {
    "o_orderdate":  "o_orderdate_ym",
    "l_shipdate":   "l_shipdate_ym",
    "c_mktsegment": "c_mktsegment",
    "p_type":       "p_type",
    "s_nationkey":  "s_nationkey",
    "n_regionkey":  "n_regionkey",
    "r_regionkey":  "r_regionkey",
}

# Filter columns whose values are full date strings but whose partition
# column stores year-month buckets ("yyyy-MM").
_YM_FILTER_COLS: frozenset[str] = frozenset({"o_orderdate", "l_shipdate"})


def _date_to_ym(date_str: str) -> str:
    """'1994-12-31' → '1994-12'"""
    return str(date_str)[:7]


def _hive_partition_matches(part_col: str, part_value: str,
                             filter_col: str, pred_min: Any, pred_max: Any) -> bool:
    """Return True if this Hive partition MAY contain rows matching the predicate."""
    mapped = _HIVE_FILTER_TO_PART.get(filter_col)
    if mapped != part_col:
        # Filter col doesn't correspond to the partition col → cannot prune.
        return True

    if filter_col in _YM_FILTER_COLS:
        min_ym = _date_to_ym(pred_min)
        max_ym = _date_to_ym(pred_max)
        return min_ym <= part_value <= max_ym

    # Direct value comparison (string or numeric)
    try:
        pv = type(pred_min)(part_value)
        return pred_min <= pv <= pred_max
    except (ValueError, TypeError):
        return str(pred_min) <= part_value <= str(pred_max)


def bytes_read_hive_partition(
    table_path: Path,
    filter_col: str,
    pred_min: Any,
    pred_max: Any,
) -> tuple[int, int, int]:
    """Compute bytes read for a Hive-style partitioned Parquet table.

    Applies directory-level pruning only — all Parquet files inside a
    matching partition directory are counted as read (no file-level stats).
    Returns (bytes_read, files_read, files_total).
    """
    part_dirs = [
        d for d in table_path.iterdir()
        if d.is_dir() and "=" in d.name and not d.name.startswith("_")
    ]

    if not part_dirs:
        # Unpartitioned Parquet — full scan of all files.
        all_files = list(table_path.rglob("*.parquet"))
        total = sum(f.stat().st_size for f in all_files)
        return total, len(all_files), len(all_files)

    part_col = part_dirs[0].name.split("=")[0]

    all_files: list[Path] = []
    read_files: list[Path] = []

    for part_dir in part_dirs:
        part_value = part_dir.name.split("=", 1)[1]
        files = list(part_dir.rglob("*.parquet"))
        all_files.extend(files)
        if _hive_partition_matches(part_col, part_value, filter_col, pred_min, pred_max):
            read_files.extend(files)

    bytes_read = sum(f.stat().st_size for f in read_files)
    return bytes_read, len(read_files), len(all_files)


def total_table_bytes_any(table_path: Path) -> tuple[int, int]:
    """Return (total_bytes, file_count) for either a Delta or Hive table."""
    if (table_path / "_delta_log").exists():
        return total_table_bytes(table_path)
    all_files = list(table_path.rglob("*.parquet"))
    total = sum(f.stat().st_size for f in all_files)
    return total, len(all_files)


def measure_bytes(
    table_path: Path,
    filter_col: str,
    pred_min: Any,
    pred_max: Any,
) -> tuple[int, int, int]:
    """Auto-detect Delta vs Hive layout and return (bytes_read, files_read, files_total)."""
    if (table_path / "_delta_log").exists():
        return bytes_read_with_predicate(table_path, filter_col, pred_min, pred_max)
    return bytes_read_hive_partition(table_path, filter_col, pred_min, pred_max)
