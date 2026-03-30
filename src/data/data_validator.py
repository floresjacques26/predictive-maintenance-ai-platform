"""Schema validation and statistical drift detection for sensor data.

Two distinct validation concerns
----------------------------------
1. **Schema validation**: Are the data types, ranges, and required fields correct?
   Run at data ingestion time to catch corrupt or malformed inputs early.

2. **Distribution drift detection**: Has the distribution of sensor values
   shifted relative to the training distribution?  A model trained on
   historical data may silently degrade if deployed on a machine whose
   sensors read outside the training distribution.

Drift detection method
----------------------
Population Stability Index (PSI):
    PSI = Σ_b (actual_fraction_b - expected_fraction_b) × log(actual / expected)

    PSI < 0.1 → no significant shift
    PSI 0.1–0.25 → moderate shift — monitor
    PSI > 0.25 → significant shift — consider retraining

PSI is commonly used in credit risk modelling for concept drift monitoring
and is appropriate here because it is:
  * Scale-free (works on normalised histograms)
  * Asymmetric (penalises large absolute deviations)
  * Fast to compute for online monitoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SensorSchema:
    """Expected ranges and types for sensor columns."""

    columns: list[str]
    target_column: str
    value_ranges: dict[str, tuple[float, float]]  # {col: (min_valid, max_valid)}
    max_nan_fraction: float = 0.05
    required_columns: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Outcome of schema validation."""

    passed: bool
    errors: list[str]
    warnings: list[str]

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation {status}"]
        if self.errors:
            lines += ["  Errors:"] + [f"    - {e}" for e in self.errors]
        if self.warnings:
            lines += ["  Warnings:"] + [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


DEFAULT_SCHEMA = SensorSchema(
    columns=["temperature", "vibration", "pressure", "rpm", "current"],
    target_column="failure_imminent",
    value_ranges={
        "temperature": (0.0, 500.0),
        "vibration": (0.0, 100.0),
        "pressure": (0.0, 50.0),
        "rpm": (0.0, 10_000.0),
        "current": (0.0, 100.0),
    },
    required_columns=["machine_id", "timestep"],
)


def validate_schema(df: pd.DataFrame, schema: Optional[SensorSchema] = None) -> ValidationResult:
    """Check a sensor DataFrame against the schema definition.

    Args:
        df: Raw sensor DataFrame.
        schema: Schema definition. Uses DEFAULT_SCHEMA if None.

    Returns:
        ValidationResult with pass/fail status and specific error messages.
    """
    if schema is None:
        schema = DEFAULT_SCHEMA

    errors: list[str] = []
    warnings: list[str] = []

    # Required columns
    all_required = set(schema.required_columns) | set(schema.columns)
    missing_cols = all_required - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    # NaN checks
    for col in schema.columns:
        if col not in df.columns:
            continue
        nan_frac = df[col].isnull().mean()
        if nan_frac > schema.max_nan_fraction:
            errors.append(
                f"Column '{col}' has {nan_frac:.1%} NaN values "
                f"(threshold: {schema.max_nan_fraction:.1%})"
            )
        elif nan_frac > 0:
            warnings.append(f"Column '{col}' has {nan_frac:.3%} NaN values (within tolerance).")

    # Value range checks
    for col, (lo, hi) in schema.value_ranges.items():
        if col not in df.columns:
            continue
        valid_mask = df[col].notna()
        vals = df.loc[valid_mask, col]
        out_of_range = ((vals < lo) | (vals > hi)).sum()
        if out_of_range > 0:
            fraction = out_of_range / len(vals)
            if fraction > 0.01:
                errors.append(
                    f"Column '{col}': {out_of_range:,} values ({fraction:.2%}) "
                    f"outside valid range [{lo}, {hi}]."
                )
            else:
                warnings.append(
                    f"Column '{col}': {out_of_range} outlier(s) outside [{lo}, {hi}] "
                    f"({fraction:.3%} — below 1% threshold)."
                )

    # Target column checks
    if schema.target_column in df.columns:
        unique_vals = set(df[schema.target_column].dropna().unique())
        non_binary = unique_vals - {0, 1, 0.0, 1.0}
        if non_binary:
            errors.append(
                f"Target column '{schema.target_column}' contains non-binary values: {non_binary}"
            )

    # Temporal monotonicity per machine
    if "machine_id" in df.columns and "timestep" in df.columns:
        non_monotone = 0
        for _, group in df.groupby("machine_id"):
            if not group["timestep"].is_monotonic_increasing:
                non_monotone += 1
        if non_monotone > 0:
            warnings.append(
                f"{non_monotone} machine(s) have non-monotone timestep sequences."
            )

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two univariate distributions.

    Args:
        reference: Reference (training) distribution.
        current: Current (production) distribution.
        n_bins: Number of equal-width histogram bins.
        eps: Smoothing factor to avoid log(0).

    Returns:
        PSI value. Interpretation:
          < 0.1    → stable
          0.1–0.25 → moderate drift
          > 0.25   → significant drift
    """
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max())
    if lo == hi:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_frac = (ref_hist + eps) / (len(reference) + eps * n_bins)
    cur_frac = (cur_hist + eps) / (len(current) + eps * n_bins)

    psi = float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))
    return psi


def detect_sensor_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    sensor_columns: list[str],
    psi_warning_threshold: float = 0.1,
    psi_critical_threshold: float = 0.25,
) -> dict[str, dict]:
    """Compute PSI for each sensor column and flag drifting channels.

    Args:
        reference_df: Training-time sensor data (reference distribution).
        current_df: Production/inference-time sensor data.
        sensor_columns: List of sensor column names.
        psi_warning_threshold: PSI above which a warning is issued.
        psi_critical_threshold: PSI above which drift is considered critical.

    Returns:
        Dict per sensor: {'psi', 'status', 'reference_mean', 'current_mean'}.
    """
    results: dict[str, dict] = {}

    for col in sensor_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        psi = compute_psi(ref_vals, cur_vals)

        if psi > psi_critical_threshold:
            status = "CRITICAL"
        elif psi > psi_warning_threshold:
            status = "WARNING"
        else:
            status = "OK"

        results[col] = {
            "psi": psi,
            "status": status,
            "reference_mean": float(ref_vals.mean()),
            "current_mean": float(cur_vals.mean()),
            "reference_std": float(ref_vals.std()),
            "current_std": float(cur_vals.std()),
        }

    return results
