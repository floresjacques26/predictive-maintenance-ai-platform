"""Detailed false-positive / false-negative analysis for predictive maintenance.

Standard metrics (F1, AUC) aggregate errors across the entire test set.
This module disaggregates errors to answer operational questions:

  * Which machine types are hardest to monitor?
  * At what degradation stage does the model fail most?
  * How close to failure do false negatives occur?
  * Does sensor dropout correlate with higher error rates?

Key function
------------
``build_window_metadata``
    Re-creates the metadata for every sliding window without altering the
    preprocessing pipeline.  Each window is characterised by the properties
    of its *last* timestep (degradation_progress, machine_type, etc.).

``error_analysis_by_group``
    Computes FN-rate and FP-rate for every level of a categorical variable.

``error_analysis_proximity``
    Buckets windows by timesteps-before-failure and measures per-bucket error.

``full_error_report``
    Calls all sub-analyses and returns a structured dict, optionally writing
    an HTML / text report.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Window metadata extraction
# ──────────────────────────────────────────────────────────────────────────────

def build_window_metadata(
    df: pd.DataFrame,
    machine_ids: np.ndarray,
    window_size: int,
    step_size: int,
    target_col: str = "failure_imminent",
) -> pd.DataFrame:
    """Reconstruct per-window metadata aligned with the windowed arrays.

    Mirrors the windowing logic in ``SensorDataPreprocessor._df_to_windows``
    but collects metadata instead of feature matrices.  The metadata for each
    window corresponds to the **last timestep** of that window — consistent
    with how labels are assigned.

    Args:
        df: Full sensor DataFrame (must contain machine_type, degradation_progress,
            timestep, failure_event columns if available).
        machine_ids: Array of machine IDs belonging to this split.
        window_size: Number of timesteps per window.
        step_size: Sliding window stride.
        target_col: Binary label column name.

    Returns:
        DataFrame with one row per window.  Columns:
          machine_id, machine_type, timestep, degradation_progress,
          failure_event, failure_imminent, timesteps_to_failure,
          degradation_stage
    """
    rows: list[dict] = []

    for machine_id in machine_ids:
        group = df[df["machine_id"] == machine_id].sort_values("timestep").reset_index(drop=True)
        n = len(group)

        # Pre-compute timesteps-to-failure (NaN if no failure event in group)
        failure_rows = group[group["failure_event"] == 1]["timestep"].values
        failure_t = int(failure_rows[0]) if len(failure_rows) > 0 else None

        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            last = group.iloc[end - 1]

            t = int(last["timestep"])
            tstf = (failure_t - t) if (failure_t is not None and failure_t >= t) else np.nan

            # Degradation stage from latent progress or from phase logic
            prog = float(last.get("degradation_progress", 0.0))
            if prog == 0.0:
                stage = "normal"
            elif prog < 1.0:
                stage = "degradation"
            else:
                stage = "post_failure_region"

            rows.append({
                "machine_id": int(last["machine_id"]),
                "machine_type": str(last.get("machine_type", "unknown")),
                "timestep": t,
                "degradation_progress": prog,
                "failure_event": int(last.get("failure_event", 0)),
                target_col: int(last.get(target_col, 0)),
                "timesteps_to_failure": tstf,
                "degradation_stage": stage,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Group-level error analysis
# ──────────────────────────────────────────────────────────────────────────────

def error_analysis_by_group(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
) -> dict[str, dict]:
    """Compute FN-rate and FP-rate for each level of ``group_col``.

    Args:
        metadata: Per-window metadata from ``build_window_metadata``.
        y_true: Ground-truth labels aligned with metadata rows.
        y_pred: Binary predicted labels (already thresholded).
        group_col: Column in metadata to group by.

    Returns:
        Dict mapping group value → {
            fn_rate, fp_rate, n_windows, n_positives, n_negatives,
            precision, recall
        }
    """
    assert len(metadata) == len(y_true) == len(y_pred), (
        f"Lengths must match: metadata={len(metadata)}, "
        f"y_true={len(y_true)}, y_pred={len(y_pred)}"
    )

    meta = metadata.copy()
    meta["y_true"] = y_true
    meta["y_pred"] = y_pred

    results: dict[str, dict] = {}
    for group_val, grp in meta.groupby(group_col):
        yt = grp["y_true"].values
        yp = grp["y_pred"].values

        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())

        n_pos = int(yt.sum())
        n_neg = int((yt == 0).sum())

        fn_rate = fn / n_pos if n_pos > 0 else 0.0      # = 1 - recall
        fp_rate = fp / n_neg if n_neg > 0 else 0.0      # = FPR
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[str(group_val)] = {
            "n_windows": len(grp),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fn_rate": float(fn_rate),
            "fp_rate": float(fp_rate),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Proximity-to-failure analysis
# ──────────────────────────────────────────────────────────────────────────────

def error_analysis_proximity(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: list[int] | None = None,
) -> dict[str, list]:
    """Error rates binned by timesteps-before-failure.

    Windows far from any failure get a synthetic bin ">horizon" meaning
    the label is negative and the model should predict 0.

    Args:
        metadata: Per-window metadata with ``timesteps_to_failure`` column.
        y_true: Ground-truth labels.
        y_pred: Binary predictions.
        bins: Right-open bin edges in timesteps.
              Default: [5, 10, 20, 30, 60, 100, ∞]

    Returns:
        Dict with lists keyed by bin label:
          bins, fn_rates, fp_rates, n_windows, n_positives
    """
    if bins is None:
        bins = [5, 10, 20, 30, 60, 100]

    meta = metadata.copy()
    meta["y_true"] = y_true
    meta["y_pred"] = y_pred

    bin_labels: list[str] = []
    fn_rates: list[float] = []
    fp_rates: list[float] = []
    n_windows_list: list[int] = []
    n_pos_list: list[int] = []

    # Bucket: windows with known timesteps_to_failure
    def _bucket(mask: pd.Series, label: str) -> None:
        sub = meta[mask]
        if len(sub) == 0:
            return
        yt = sub["y_true"].values
        yp = sub["y_pred"].values
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        n_pos = int(yt.sum())
        n_neg = int((yt == 0).sum())
        bin_labels.append(label)
        fn_rates.append(fn / n_pos if n_pos > 0 else 0.0)
        fp_rates.append(fp / n_neg if n_neg > 0 else 0.0)
        n_windows_list.append(len(sub))
        n_pos_list.append(n_pos)

    prev = 0
    for edge in bins:
        mask = (meta["timesteps_to_failure"] >= prev) & (meta["timesteps_to_failure"] < edge)
        _bucket(mask, f"{prev}–{edge-1}")
        prev = edge

    # Final bucket: timesteps_to_failure >= last bin edge
    mask = meta["timesteps_to_failure"] >= prev
    _bucket(mask, f"≥{prev}")

    # Far-from-failure (NaN timesteps_to_failure = already past failure or very early)
    mask_nan = meta["timesteps_to_failure"].isna()
    _bucket(mask_nan, "no failure in window")

    return {
        "bins": bin_labels,
        "fn_rates": fn_rates,
        "fp_rates": fp_rates,
        "n_windows": n_windows_list,
        "n_positives": n_pos_list,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full error report
# ──────────────────────────────────────────────────────────────────────────────

def full_error_report(
    df: pd.DataFrame,
    test_machine_ids: np.ndarray,
    window_size: int,
    step_size: int,
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, dict]:
    """Run all error analyses and return a nested result dict.

    Args:
        df: Full sensor DataFrame.
        test_machine_ids: Machine IDs in the test split.
        window_size: Window size used during preprocessing.
        step_size: Stride used during preprocessing.
        predictions: {model_name: proba_array}.
        y_true: Test set ground-truth labels.
        thresholds: {model_name: decision_threshold}.

    Returns:
        Nested dict: {model_name: {by_machine_type, by_degradation_stage,
                                   by_proximity, aggregate_stats}}
    """
    logger.info("Building window metadata for test split…")
    metadata = build_window_metadata(
        df, test_machine_ids, window_size, step_size
    )

    if len(metadata) != len(y_true):
        logger.warning(
            f"Metadata rows ({len(metadata)}) ≠ y_true length ({len(y_true)}). "
            "Skipping error analysis — verify that test_machine_ids match the "
            "preprocessor split used during training."
        )
        return {}

    report: dict[str, dict] = {}

    for name, proba in predictions.items():
        thresh = thresholds.get(name, 0.5)
        y_pred = (proba >= thresh).astype(int)

        by_type = error_analysis_by_group(metadata, y_true, y_pred, "machine_type")
        by_stage = error_analysis_by_group(metadata, y_true, y_pred, "degradation_stage")
        proximity = error_analysis_proximity(metadata, y_true, y_pred)

        # Aggregate FP/FN confidence analysis
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        report[name] = {
            "by_machine_type": by_type,
            "by_degradation_stage": by_stage,
            "by_proximity_to_failure": proximity,
            "fp_proba_stats": _describe(proba[fp_mask]),
            "fn_proba_stats": _describe(proba[fn_mask]),
            "n_fp": int(fp_mask.sum()),
            "n_fn": int(fn_mask.sum()),
            "fn_rate_overall": float(fn_mask.sum() / max(y_true.sum(), 1)),
            "fp_rate_overall": float(fp_mask.sum() / max((y_true == 0).sum(), 1)),
        }
        logger.info(
            f"  {name}: FN={report[name]['n_fn']} (rate={report[name]['fn_rate_overall']:.2%})"
            f"  FP={report[name]['n_fp']} (rate={report[name]['fp_rate_overall']:.2%})"
        )

    return report


def _describe(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Text summary
# ──────────────────────────────────────────────────────────────────────────────

def format_error_report(report: dict[str, dict]) -> str:
    """Format the error report dict as a human-readable text summary."""
    lines: list[str] = []

    for model_name, data in report.items():
        lines.append("=" * 70)
        lines.append(f"ERROR ANALYSIS — {model_name}")
        lines.append("=" * 70)
        lines.append(
            f"  Overall FN: {data['n_fn']} ({data['fn_rate_overall']:.2%} miss rate)  "
            f"FP: {data['n_fp']} ({data['fp_rate_overall']:.2%} false alarm rate)"
        )

        # By machine type
        lines.append("\n  By machine type:")
        lines.append(f"    {'Type':<30} {'FN-rate':>8} {'FP-rate':>8} {'F1':>8} {'N':>6}")
        lines.append(f"    {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for t, v in sorted(data["by_machine_type"].items(), key=lambda x: -x[1]["fn_rate"]):
            lines.append(
                f"    {t:<30} {v['fn_rate']:>8.2%} {v['fp_rate']:>8.2%} "
                f"{v['f1']:>8.4f} {v['n_windows']:>6}"
            )

        # By degradation stage
        lines.append("\n  By degradation stage:")
        lines.append(f"    {'Stage':<22} {'FN-rate':>8} {'FP-rate':>8} {'F1':>8} {'N':>6}")
        lines.append(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for s, v in data["by_degradation_stage"].items():
            lines.append(
                f"    {s:<22} {v['fn_rate']:>8.2%} {v['fp_rate']:>8.2%} "
                f"{v['f1']:>8.4f} {v['n_windows']:>6}"
            )

        # Proximity
        prox = data["by_proximity_to_failure"]
        lines.append("\n  By proximity to failure (timesteps before failure):")
        lines.append(f"    {'Bin':<22} {'FN-rate':>8} {'FP-rate':>8} {'N pos':>6} {'N total':>8}")
        lines.append(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
        for i, b in enumerate(prox["bins"]):
            lines.append(
                f"    {b:<22} {prox['fn_rates'][i]:>8.2%} {prox['fp_rates'][i]:>8.2%} "
                f"{prox['n_positives'][i]:>6} {prox['n_windows'][i]:>8}"
            )

        # FN/FP confidence
        fp_s = data["fp_proba_stats"]
        fn_s = data["fn_proba_stats"]
        lines.append(
            f"\n  FP scores: mean={fp_s['mean']:.3f} ± {fp_s['std']:.3f}  "
            f"[{fp_s['min']:.3f}, {fp_s['max']:.3f}]"
        )
        lines.append(
            f"  FN scores: mean={fn_s['mean']:.3f} ± {fn_s['std']:.3f}  "
            f"[{fn_s['min']:.3f}, {fn_s['max']:.3f}]"
        )
        lines.append(
            textwrap.dedent(f"""
  Insights:
    • FN scores close to the threshold → model is uncertain at decision boundary
    • FN scores very low (<0.1)         → model completely misses degradation pattern
    • FP rate highest in '{_max_key(data['by_machine_type'], 'fp_rate')}'
      → that machine type's normal signal resembles failure
    • FN rate highest in '{_max_key(data['by_machine_type'], 'fn_rate')}'
      → degradation signature is subtle for that machine type
    • Errors concentrated in '{_max_key(data['by_degradation_stage'], 'fn_rate')}' phase
      → early degradation is the hardest to detect reliably
""").rstrip()
        )

    return "\n".join(lines)


def _max_key(d: dict[str, dict], metric: str) -> str:
    if not d:
        return "N/A"
    return max(d, key=lambda k: d[k].get(metric, 0.0))
