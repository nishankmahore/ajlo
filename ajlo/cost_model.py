"""Cost model functions — all values in bytes (Section 6.3).

JIS is a dimensionless ranking signal. It does NOT appear in the
optimization objective. The objective is defined entirely in bytes.
"""

PHI_SCAN = {
    "CO_CLUSTER": 0.30,   # Liquid Clustering file skipping
    "CO_PARTITION": 0.20,  # Partition pruning (→ 0 if all partitions qualify)
    "NOOP": 0.00,
}

BYTES_PER_ROW: float = 200.0    # Default estimate; overridden per table
SCAN_OVERHEAD: float = 1.10     # Column deserialization + predicate eval
MIGRATION_PENALTY: float = 3.0  # Multiplier when switching between incompatible layouts
CO_CLUSTER_STORAGE_OVERHEAD: float = 0.02  # ~2% extra for Hilbert-curve file organization


def skew_penalty(max_key_freq: float, avg_key_freq: float, kappa: float = 0.1) -> float:
    """SkewPenalty from Section 6.3. Returns value in [0, 1]."""
    if avg_key_freq <= 0:
        return 0.0
    raw = kappa * (max_key_freq / avg_key_freq - 1)
    return max(0.0, min(1.0, raw))


def expected_scan_bytes(
    frequency: float,
    data_volume: float,
    bytes_per_row: float = BYTES_PER_ROW,
    scan_overhead: float = SCAN_OVERHEAD,
) -> float:
    """ExpScan(i,j) = F × D × ρ × α_scan. Section 6.3."""
    return frequency * data_volume * bytes_per_row * scan_overhead


def scan_reduction(
    exp_scan: float,
    strategy_type: str,
    skew_pen: float = 0.0,
) -> float:
    """ScanReduction(i,j,L) per window. Section 6.3."""
    phi = PHI_SCAN.get(strategy_type, 0.0)
    return exp_scan * phi * (1.0 - skew_pen)


def rewrite_cost(
    table_size_bytes: float,
    from_strategy_type: str,
    to_strategy_type: str,
) -> float:
    """RewriteCost for transitioning one table to a new strategy. Section 6.3.

    Migration penalty (3×) applies when switching between CLUSTER BY
    and PARTITIONED BY — requires disabling current spec + full rebuild.
    """
    if to_strategy_type == "NOOP":
        return 0.0
    if from_strategy_type == to_strategy_type:
        return table_size_bytes  # re-clustering same type, no migration penalty
    incompatible = {
        ("CO_CLUSTER", "CO_PARTITION"),
        ("CO_PARTITION", "CO_CLUSTER"),
    }
    if (from_strategy_type, to_strategy_type) in incompatible:
        return table_size_bytes * MIGRATION_PENALTY
    return table_size_bytes


def storage_cost(table_size_bytes: float, strategy_type: str) -> float:
    """StorageCost overhead from applying a strategy. Section 6.3."""
    if strategy_type == "CO_CLUSTER":
        return table_size_bytes * CO_CLUSTER_STORAGE_OVERHEAD
    return 0.0


def npv(
    scan_reduction_per_window: float,
    rewrite_cost_bytes: float,
    storage_cost_bytes: float,
    H: int = 4,
    r: float = 0.10,
) -> float:
    """NPV(L) over horizon H at per-window discount rate r. Section 6.3.

    For H → ∞: NPV = scan_reduction / r − costs.
    H=4 = 4 weekly windows ≈ 1 month (default).
    """
    if r <= 0:
        total_reduction = scan_reduction_per_window * H
    else:
        annuity = (1.0 - (1.0 + r) ** (-H)) / r
        total_reduction = scan_reduction_per_window * annuity
    return total_reduction - rewrite_cost_bytes - storage_cost_bytes
