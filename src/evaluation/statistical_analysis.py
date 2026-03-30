"""Statistical analysis utilities: bootstrap CI, error decomposition, calibration.

Why bootstrap?
--------------
Standard formula-based confidence intervals assume large-sample normality
and are inappropriate for complex metrics like AUC or MCC.  Bootstrap
resampling is non-parametric and makes no distributional assumptions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy

from src.evaluation.metrics import compute_classification_metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_samples: int = 1000,
    ci: float = 0.95,
    threshold: float = 0.5,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap CI for all classification metrics.

    For each of the ``n_samples`` bootstrap replicates we:
      1. Draw N samples with replacement from (y_true, y_pred_proba).
      2. Compute all metrics on the replicate.

    CI bounds are the (alpha/2, 1-alpha/2) percentiles of the distribution.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        n_samples: Number of bootstrap replicates.
        ci: Coverage probability (e.g. 0.95 for 95% CI).
        threshold: Decision threshold.
        random_seed: Seed for reproducibility.

    Returns:
        Dict mapping metric_name → {'mean', 'std', 'lower', 'upper'}.
    """
    rng = np.random.default_rng(random_seed)
    alpha = 1.0 - ci
    n = len(y_true)

    # Collect bootstrap distributions
    boot_metrics: dict[str, list[float]] = {}
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_pred_proba[idx]

        # Skip replicates with only one class (metrics would be undefined)
        if len(np.unique(y_b)) < 2:
            continue

        m = compute_classification_metrics(y_b, p_b, threshold=threshold)
        for key, val in m.items():
            if isinstance(val, float):
                boot_metrics.setdefault(key, []).append(val)

    result: dict[str, dict[str, float]] = {}
    for key, values in boot_metrics.items():
        arr = np.array(values)
        result[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "lower": float(np.percentile(arr, 100 * alpha / 2)),
            "upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return result


def prediction_distribution_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 50,
) -> dict[str, float | np.ndarray]:
    """Describe the distribution of predicted probabilities by true class.

    Computes per-class statistics and the KL divergence KL(pos || neg)
    to quantify how well the model separates the two score distributions.
    Higher KL divergence means better class separation.

    Args:
        y_true: Binary ground-truth labels.
        y_pred_proba: Predicted probability for class 1.
        n_bins: Number of histogram bins for KL divergence estimation.

    Returns:
        Dict with per-class statistics, class counts, and kl_divergence.
    """
    pos_preds = y_pred_proba[y_true == 1]
    neg_preds = y_pred_proba[y_true == 0]

    def _desc(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    # KL divergence via histogram density estimation
    # Add small epsilon to avoid log(0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    eps = 1e-10
    pos_hist, _ = np.histogram(pos_preds, bins=bins, density=True)
    neg_hist, _ = np.histogram(neg_preds, bins=bins, density=True)
    pos_hist = pos_hist + eps
    neg_hist = neg_hist + eps
    pos_hist /= pos_hist.sum()
    neg_hist /= neg_hist.sum()
    kl_pos_neg = float(scipy_entropy(pos_hist, neg_hist))  # KL(pos || neg)

    return {
        "positive_class": _desc(pos_preds),
        "negative_class": _desc(neg_preds),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "class_imbalance_ratio": float(len(neg_preds) / max(len(pos_preds), 1)),
        "kl_divergence_pos_neg": kl_pos_neg,
    }


def error_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Identify and characterise misclassified samples.

    Returns:
        Dict with boolean masks for FP and FN, plus their predicted probability
        distributions.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)

    return {
        "fp_mask": fp_mask,
        "fn_mask": fn_mask,
        "tp_mask": tp_mask,
        "tn_mask": tn_mask,
        "fp_proba": y_pred_proba[fp_mask],
        "fn_proba": y_pred_proba[fn_mask],
        "tp_proba": y_pred_proba[tp_mask],
        "tn_proba": y_pred_proba[tn_mask],
    }


def calibration_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute reliability diagram data (expected vs actual positive rates).

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        n_bins: Number of probability bins.

    Returns:
        Dict with 'bin_centers', 'mean_predicted', 'fraction_positive', 'counts'.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_indices == i
        counts[i] = mask.sum()
        if counts[i] > 0:
            mean_pred[i] = y_pred_proba[mask].mean()
            frac_pos[i] = y_true[mask].mean()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return {
        "bin_centers": bin_centers,
        "mean_predicted": mean_pred,
        "fraction_positive": frac_pos,
        "counts": counts,
    }
