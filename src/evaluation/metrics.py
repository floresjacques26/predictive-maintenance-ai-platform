"""Comprehensive classification metrics for predictive maintenance evaluation.

Key considerations for this domain
------------------------------------
* **Class imbalance**: failures are rare → report PR-AUC, not just ROC-AUC.
* **Threshold sensitivity**: the optimal decision threshold varies by
  use-case (minimise missed failures vs minimise false alarms).
* **Business cost asymmetry**: FN (missed failure) >> FP (false alarm).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels {0, 1}.
        y_pred_proba: Predicted probability of class 1.
        threshold: Decision threshold.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc,
        specificity, balanced_accuracy, mcc (Matthews Correlation Coeff).
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (recall_score(y_true, y_pred, zero_division=0) + specificity) / 2

    # Matthews Correlation Coefficient
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": float(threshold),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return confusion matrix [[TN, FP], [FN, TP]] as numpy array."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
    beta: float = 1.0,
) -> tuple[float, float]:
    """Search over thresholds to maximise a given metric.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        metric: One of 'f1', 'fbeta', 'j_stat' (Youden's J).
        beta: Beta parameter for F-beta score (only used when metric='fbeta').

    Returns:
        (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "fbeta":
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            denom = (beta**2 * prec) + rec
            score = (1 + beta**2) * prec * rec / denom if denom > 0 else 0.0
        elif metric == "j_stat":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = sens + spec - 1
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        if score > best_score:
            best_score = score
            best_thresh = t

    return float(best_thresh), float(best_score)


def threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_steps: int = 100,
) -> dict[str, np.ndarray]:
    """Compute precision, recall and F1 across all thresholds.

    Returns:
        Dict with arrays 'thresholds', 'precision', 'recall', 'f1'.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return {
        "thresholds": thresholds,
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "f1": np.array(f1s),
    }


def get_roc_curve_data(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> dict[str, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def get_pr_curve_data(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> dict[str, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}
