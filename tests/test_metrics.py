"""Tests for evaluation metrics and statistical analysis."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    find_optimal_threshold,
    get_pr_curve_data,
    get_roc_curve_data,
    threshold_analysis,
)
from src.evaluation.statistical_analysis import (
    bootstrap_confidence_intervals,
    calibration_analysis,
    error_analysis,
    prediction_distribution_analysis,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    return y_true, y_proba


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(0, 1, size=n)
    return y_true, y_proba


@pytest.fixture
def imbalanced_predictions():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.uniform(0, 1, n) < 0.1).astype(int)  # 10% positive
    y_proba = rng.uniform(0, 1, n)
    y_proba[y_true == 1] += 0.3  # slight signal
    y_proba = np.clip(y_proba, 0, 1)
    return y_true, y_proba


# ── Classification metrics ────────────────────────────────────────────────────

def test_perfect_predictions_give_perfect_metrics(perfect_predictions):
    y_true, y_proba = perfect_predictions
    metrics = compute_classification_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_metrics_keys_complete():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7])
    metrics = compute_classification_metrics(y_true, y_proba)
    required = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                "specificity", "balanced_accuracy", "mcc", "tp", "fp", "fn", "tn"}
    assert required.issubset(set(metrics.keys()))


def test_metrics_range(random_predictions):
    y_true, y_proba = random_predictions
    metrics = compute_classification_metrics(y_true, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of [0,1] range"


def test_confusion_matrix_sums_to_n():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.8, 0.9, 0.2, 0.7])
    cm = compute_confusion_matrix(y_true, y_proba, threshold=0.5)
    assert cm.sum() == len(y_true)


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7])
    cm = compute_confusion_matrix(y_true, y_proba)
    assert cm.shape == (2, 2)


def test_threshold_affects_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.3, 0.6, 0.7, 0.9])
    m_strict = compute_classification_metrics(y_true, y_proba, threshold=0.8)
    m_lax = compute_classification_metrics(y_true, y_proba, threshold=0.2)
    # Stricter threshold → lower recall (fewer predicted positives)
    assert m_strict["recall"] <= m_lax["recall"]


# ── Threshold search ─────────────────────────────────────────────────────────

def test_find_optimal_threshold_returns_valid_range(random_predictions):
    y_true, y_proba = random_predictions
    thresh, score = find_optimal_threshold(y_true, y_proba, metric="f1")
    assert 0.0 < thresh < 1.0
    assert 0.0 <= score <= 1.0


def test_optimal_threshold_improves_on_default(imbalanced_predictions):
    y_true, y_proba = imbalanced_predictions
    default_metrics = compute_classification_metrics(y_true, y_proba, threshold=0.5)
    opt_thresh, _ = find_optimal_threshold(y_true, y_proba, metric="f1")
    opt_metrics = compute_classification_metrics(y_true, y_proba, threshold=opt_thresh)
    assert opt_metrics["f1"] >= default_metrics["f1"] - 0.01  # allow tiny float error


def test_threshold_analysis_returns_arrays(random_predictions):
    y_true, y_proba = random_predictions
    result = threshold_analysis(y_true, y_proba, n_steps=50)
    assert len(result["thresholds"]) == 50
    assert len(result["precision"]) == 50
    assert len(result["recall"]) == 50
    assert len(result["f1"]) == 50


# ── ROC / PR curve data ──────────────────────────────────────────────────────

def test_roc_curve_data_keys(random_predictions):
    y_true, y_proba = random_predictions
    data = get_roc_curve_data(y_true, y_proba)
    assert {"fpr", "tpr", "thresholds"} == set(data.keys())


def test_pr_curve_data_keys(random_predictions):
    y_true, y_proba = random_predictions
    data = get_pr_curve_data(y_true, y_proba)
    assert {"precision", "recall", "thresholds"} == set(data.keys())


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_structure(random_predictions):
    y_true, y_proba = random_predictions
    ci = bootstrap_confidence_intervals(y_true, y_proba, n_samples=100, random_seed=42)
    for key, val in ci.items():
        assert {"mean", "std", "lower", "upper"} == set(val.keys())
        assert val["lower"] <= val["mean"] <= val["upper"]


def test_bootstrap_ci_coverage(random_predictions):
    y_true, y_proba = random_predictions
    ci = bootstrap_confidence_intervals(y_true, y_proba, n_samples=200, ci=0.95)
    # Point estimate should fall within CI
    for metric, bounds in ci.items():
        assert bounds["lower"] <= bounds["mean"] <= bounds["upper"]


# ── Statistical analysis ─────────────────────────────────────────────────────

def test_prediction_distribution_analysis(imbalanced_predictions):
    y_true, y_proba = imbalanced_predictions
    dist = prediction_distribution_analysis(y_true, y_proba)
    assert "positive_class" in dist
    assert "negative_class" in dist
    assert dist["n_positive"] + dist["n_negative"] == len(y_true)
    # Positive class should have higher mean predicted probability
    assert dist["positive_class"]["mean"] > dist["negative_class"]["mean"]


def test_error_analysis_masks_sum_to_n(perfect_predictions):
    y_true, y_proba = perfect_predictions
    errors = error_analysis(y_true, y_proba, threshold=0.5)
    total = (
        errors["tp_mask"].sum() + errors["tn_mask"].sum()
        + errors["fp_mask"].sum() + errors["fn_mask"].sum()
    )
    assert total == len(y_true)


def test_calibration_analysis_shape(random_predictions):
    y_true, y_proba = random_predictions
    cal = calibration_analysis(y_true, y_proba, n_bins=10)
    assert len(cal["bin_centers"]) == 10
    assert len(cal["fraction_positive"]) == 10
    assert len(cal["counts"]) == 10
    assert cal["counts"].sum() == len(y_true)
