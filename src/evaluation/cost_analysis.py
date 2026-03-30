"""Cost-sensitive threshold optimisation for predictive maintenance.

Business context
----------------
In industrial maintenance the cost of error types is asymmetric:

  False Negative (FN): missed failure → unplanned downtime, emergency repair,
                        safety incident, production loss.  Typically 10–100×
                        more costly than a false alarm.

  False Positive (FP): unnecessary maintenance stop → planned downtime,
                        labour cost, spare parts.  Predictable and manageable.

Standard F1-optimal thresholds treat FP and FN equally, which is wrong for
this domain.  A cost-optimal threshold minimises:

    Expected Cost(t) = FP(t) × C_fp + FN(t) × C_fn

where FP(t) and FN(t) are the counts at threshold t.

This module implements:
  1. Cost matrix definition
  2. Expected cost curve across all thresholds
  3. Cost-optimal threshold selection
  4. Comparison: F1-optimal vs cost-optimal decision
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostMatrix:
    """Business cost parameters for binary classification errors.

    Args:
        cost_fn: Cost of a False Negative (missed failure).
                 Represents unplanned downtime, emergency repair, safety risk.
        cost_fp: Cost of a False Positive (unnecessary maintenance alert).
                 Represents planned downtime and labour cost.
        cost_tp: Benefit (negative cost) of a True Positive — early warning
                 that enabled proactive maintenance.  Defaults to 0.
        cost_tn: Benefit (negative cost) of correct no-alarm.  Defaults to 0.

    Example (industrial pump):
        FN: 8 h unplanned stop × €5,000/h + emergency parts = €50,000
        FP: 4 h planned stop × €1,000/h + labour = €5,000
        → cost_fn=50_000, cost_fp=5_000, ratio=10:1
    """

    cost_fn: float
    cost_fp: float
    cost_tp: float = 0.0
    cost_tn: float = 0.0

    @property
    def fn_fp_ratio(self) -> float:
        """How many times more expensive is a missed failure than a false alarm."""
        return self.cost_fn / max(self.cost_fp, 1e-9)


def compute_expected_cost_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_matrix: CostMatrix,
    n_steps: int = 200,
) -> dict[str, np.ndarray]:
    """Compute total expected cost for each threshold value.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        cost_matrix: Business cost parameters.
        n_steps: Number of threshold values to evaluate.

    Returns:
        Dict with arrays: 'thresholds', 'expected_cost', 'normalised_cost',
        'n_fp', 'n_fn', 'n_tp', 'n_tn'.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    n = len(y_true)
    n_pos = y_true.sum()
    n_neg = n - n_pos

    expected_costs = np.zeros(n_steps)
    n_fp_arr = np.zeros(n_steps, dtype=int)
    n_fn_arr = np.zeros(n_steps, dtype=int)
    n_tp_arr = np.zeros(n_steps, dtype=int)
    n_tn_arr = np.zeros(n_steps, dtype=int)

    for i, t in enumerate(thresholds):
        y_pred = (y_pred_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        cost = (
            fn * cost_matrix.cost_fn
            + fp * cost_matrix.cost_fp
            - tp * cost_matrix.cost_tp
            - tn * cost_matrix.cost_tn
        )
        expected_costs[i] = cost
        n_fp_arr[i] = fp
        n_fn_arr[i] = fn
        n_tp_arr[i] = tp
        n_tn_arr[i] = tn

    # Normalise by worst-case cost (predict all as negative → all failures missed)
    worst_case = n_pos * cost_matrix.cost_fn
    normalised = expected_costs / max(worst_case, 1.0)

    return {
        "thresholds": thresholds,
        "expected_cost": expected_costs,
        "normalised_cost": normalised,
        "n_fp": n_fp_arr,
        "n_fn": n_fn_arr,
        "n_tp": n_tp_arr,
        "n_tn": n_tn_arr,
    }


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_matrix: CostMatrix,
    n_steps: int = 200,
) -> tuple[float, float]:
    """Find the threshold that minimises expected total cost.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        cost_matrix: Business cost parameters.
        n_steps: Resolution of threshold grid.

    Returns:
        (optimal_threshold, minimum_expected_cost)
    """
    curve = compute_expected_cost_curve(y_true, y_pred_proba, cost_matrix, n_steps)
    best_idx = int(np.argmin(curve["expected_cost"]))
    return (
        float(curve["thresholds"][best_idx]),
        float(curve["expected_cost"][best_idx]),
    )


def cost_threshold_comparison(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_matrix: CostMatrix,
    f1_threshold: float,
) -> dict[str, dict]:
    """Compare F1-optimal vs cost-optimal decision threshold.

    Returns a structured comparison showing how much cost the business
    saves by using the cost-optimal threshold instead of F1-optimal.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        cost_matrix: Business cost parameters.
        f1_threshold: Threshold selected by maximising F1.

    Returns:
        Dict comparing 'f1_threshold' and 'cost_optimal_threshold' decisions.
    """
    cost_thresh, min_cost = find_cost_optimal_threshold(y_true, y_pred_proba, cost_matrix)

    def _evaluate(threshold: float) -> dict:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        total_cost = fp * cost_matrix.cost_fp + fn * cost_matrix.cost_fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_cost": total_cost,
            "cost_fn_contribution": fn * cost_matrix.cost_fn,
            "cost_fp_contribution": fp * cost_matrix.cost_fp,
        }

    f1_eval = _evaluate(f1_threshold)
    cost_eval = _evaluate(cost_thresh)
    cost_saving = f1_eval["total_cost"] - cost_eval["total_cost"]

    return {
        "cost_matrix": {
            "cost_fn": cost_matrix.cost_fn,
            "cost_fp": cost_matrix.cost_fp,
            "fn_fp_ratio": cost_matrix.fn_fp_ratio,
        },
        "f1_threshold_decision": f1_eval,
        "cost_optimal_decision": cost_eval,
        "cost_saving_vs_f1": cost_saving,
        "cost_saving_pct": 100.0 * cost_saving / max(f1_eval["total_cost"], 1.0),
    }


def theoretical_cost_optimal_threshold(cost_fp: float, cost_fn: float, prevalence: float) -> float:
    """Compute the theoretically optimal threshold under cost-sensitive classification.

    Under the Bayes risk framework, the optimal threshold is:
        t* = C_fp / (C_fp + C_fn × (1 - prevalence) / prevalence)

    This assumes a perfectly calibrated model.  Use as a sanity check
    against empirically found thresholds.

    Args:
        cost_fp: Cost of a false positive.
        cost_fn: Cost of a false negative.
        prevalence: True positive rate in the population (P(y=1)).

    Returns:
        Theoretically optimal decision threshold.
    """
    if prevalence <= 0 or prevalence >= 1:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    numerator = cost_fp * prevalence
    denominator = cost_fp * prevalence + cost_fn * (1 - prevalence)
    return float(numerator / denominator)
