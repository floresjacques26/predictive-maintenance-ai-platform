"""Model calibration: measurement, correction, and reporting.

Why calibration matters in maintenance
---------------------------------------
A model that outputs P(failure) = 0.8 should correspond to failures occurring
80% of the time in similar situations.  Without calibration, a maintenance
engineer cannot reason about the probability scale — they can only use
the score for ranking.

Calibration error is especially common in:
  * Sigmoid outputs with pos_weight (shifts the probability scale)
  * Imbalanced datasets (model biased toward majority class)
  * LSTM models (no explicit calibration layer)

Implemented methods
-------------------
* **Brier Score** — mean squared error of probability estimates; 0 = perfect
* **Expected Calibration Error (ECE)** — weighted mean |predicted - actual|
* **Platt Scaling** — logistic regression on raw model scores (linear)
* **Isotonic Regression** — non-parametric monotone fit (more flexible than Platt)
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Brier score: mean squared probability error.

    Range: [0, 1] — lower is better.
    Reference (no-skill) = prevalence × (1 - prevalence).

    Args:
        y_true: Binary ground-truth labels.
        y_pred_proba: Predicted probabilities for class 1.

    Returns:
        Brier score as float.
    """
    return float(brier_score_loss(y_true, y_pred_proba))


def compute_brier_skill_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Brier Skill Score (BSS) relative to a climatological baseline.

    BSS = 1 - (Brier / Brier_reference)
    BSS = 1 → perfect; BSS = 0 → no skill over base rate; BSS < 0 → worse than base rate.

    Args:
        y_true: Binary ground-truth labels.
        y_pred_proba: Predicted probabilities for class 1.

    Returns:
        Brier Skill Score ∈ (-∞, 1].
    """
    brier = brier_score_loss(y_true, y_pred_proba)
    prevalence = y_true.mean()
    brier_ref = prevalence * (1 - prevalence)  # always predicting the base rate
    if brier_ref == 0:
        return 0.0
    return float(1.0 - brier / brier_ref)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE): sample-weighted mean calibration gap.

    ECE = Σ_b (|B_b| / n) × |acc(B_b) - conf(B_b)|

    where B_b is the set of samples in bin b, acc is the fraction of positives,
    and conf is the mean predicted probability.

    Args:
        y_true: Binary ground-truth labels.
        y_pred_proba: Predicted probabilities.
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE ∈ [0, 1] — lower is better.
    """
    n = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        if mask.sum() == 0:
            continue
        conf = y_pred_proba[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)

    return float(ece)


class PlattScaler:
    """Post-hoc calibration via Platt scaling (logistic regression on scores).

    Platt scaling fits a logistic function: P(y=1 | s) = 1 / (1 + exp(A*s + B))
    where s is the raw model score.  Effective when miscalibration is roughly
    monotone (sigmoid-shaped).

    Usage::
        scaler = PlattScaler()
        scaler.fit(val_scores, val_labels)
        calibrated_proba = scaler.predict_proba(test_scores)

    Args:
        max_iter: Maximum LR solver iterations.
    """

    def __init__(self, max_iter: int = 1000) -> None:
        self._lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
        self._fitted = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        """Fit on held-out (validation) predictions — never on training scores."""
        self._lr.fit(scores.reshape(-1, 1), y_true)
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        if not self._fitted:
            raise RuntimeError("PlattScaler not fitted.")
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]

    @property
    def coef(self) -> float:
        return float(self._lr.coef_[0][0])

    @property
    def intercept(self) -> float:
        return float(self._lr.intercept_[0])


class IsotonicCalibrator:
    """Post-hoc calibration via isotonic regression (non-parametric).

    Isotonic regression fits a non-decreasing step function to the
    (score, label) pairs.  More flexible than Platt scaling but
    requires a larger calibration set to avoid overfitting.

    Recommended when: Platt scaling underperforms (non-sigmoid miscalibration).
    Risk: overfits on small calibration sets (< 1000 samples).

    Args:
        out_of_bounds: How to handle scores outside the calibration range.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self._ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=out_of_bounds)
        self._fitted = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._ir.fit(scores, y_true.astype(float))
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator not fitted.")
        return self._ir.predict(scores)


def compare_calibrators(
    y_true_val: np.ndarray,
    y_proba_val: np.ndarray,
    y_true_test: np.ndarray,
    y_proba_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Fit calibrators on val, evaluate on test set.

    Returns a dict comparing uncalibrated, Platt, and isotonic models
    across Brier Score, BSS, and ECE.

    Args:
        y_true_val / y_proba_val: Validation labels and raw model probabilities.
        y_true_test / y_proba_test: Test labels and raw model probabilities.

    Returns:
        {method: {brier_score, brier_skill_score, ece}}
    """
    platt = PlattScaler().fit(y_proba_val, y_true_val)
    isotonic = IsotonicCalibrator().fit(y_proba_val, y_true_val)

    methods: dict[str, np.ndarray] = {
        "uncalibrated": y_proba_test,
        "platt_scaling": platt.predict_proba(y_proba_test),
        "isotonic_regression": isotonic.predict_proba(y_proba_test),
    }

    results: dict[str, dict[str, float]] = {}
    for name, proba in methods.items():
        results[name] = {
            "brier_score": compute_brier_score(y_true_test, proba),
            "brier_skill_score": compute_brier_skill_score(y_true_test, proba),
            "ece": compute_expected_calibration_error(y_true_test, proba),
        }

    return results
