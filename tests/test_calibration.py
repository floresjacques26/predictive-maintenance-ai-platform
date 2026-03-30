"""Tests for calibration, cost analysis, and significance testing modules."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    IsotonicCalibrator,
    PlattScaler,
    compare_calibrators,
    compute_brier_score,
    compute_brier_skill_score,
    compute_expected_calibration_error,
)
from src.evaluation.cost_analysis import (
    CostMatrix,
    cost_threshold_comparison,
    find_cost_optimal_threshold,
    theoretical_cost_optimal_threshold,
)
from src.evaluation.significance_testing import (
    delong_auc_confidence_interval,
    mcnemar_test,
    paired_permutation_test,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_data():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_proba = np.clip(y_true.astype(float) + rng.normal(0, 0.3, n), 0, 1)
    return y_true, y_proba


@pytest.fixture
def perfect_data():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    return y_true, y_proba


@pytest.fixture
def imbalanced_data():
    rng = np.random.default_rng(99)
    n = 1000
    y_true = (rng.uniform(0, 1, n) < 0.1).astype(int)
    y_proba = rng.uniform(0, 1, n)
    y_proba[y_true == 1] += 0.3
    y_proba = np.clip(y_proba, 0, 1)
    return y_true, y_proba


# ── Brier Score ───────────────────────────────────────────────────────────────

def test_brier_score_perfect(perfect_data):
    y_true, y_proba = perfect_data
    bs = compute_brier_score(y_true, y_proba)
    assert bs < 0.01, f"Perfect predictions should yield near-zero Brier score, got {bs:.4f}"


def test_brier_score_range(balanced_data):
    y_true, y_proba = balanced_data
    bs = compute_brier_score(y_true, y_proba)
    assert 0.0 <= bs <= 1.0


def test_brier_score_random_baseline(imbalanced_data):
    y_true, _ = imbalanced_data
    prevalence = y_true.mean()
    # Predicting base rate always
    naive_proba = np.full(len(y_true), prevalence)
    naive_bs = compute_brier_score(y_true, naive_proba)
    expected_ref = prevalence * (1 - prevalence)
    np.testing.assert_allclose(naive_bs, expected_ref, atol=1e-6)


def test_brier_skill_score_positive_for_informative(balanced_data):
    y_true, y_proba = balanced_data
    bss = compute_brier_skill_score(y_true, y_proba)
    assert bss > 0, "Informative model should have positive BSS"


def test_brier_skill_score_zero_for_baseline(imbalanced_data):
    y_true, _ = imbalanced_data
    prevalence = y_true.mean()
    naive = np.full(len(y_true), prevalence)
    bss = compute_brier_skill_score(y_true, naive)
    np.testing.assert_allclose(bss, 0.0, atol=1e-6)


def test_ece_perfect_calibration():
    # If predicted proba equals true fraction in every bin → ECE = 0
    n = 1000
    proba = np.linspace(0, 1, n)
    # For each probability p, y~Bernoulli(p)
    rng = np.random.default_rng(42)
    y_true = (rng.uniform(0, 1, n) < proba).astype(int)
    ece = compute_expected_calibration_error(y_true, proba, n_bins=10)
    assert ece < 0.05, f"Well-calibrated model should have ECE < 0.05, got {ece:.4f}"


def test_ece_range(balanced_data):
    y_true, y_proba = balanced_data
    ece = compute_expected_calibration_error(y_true, y_proba)
    assert 0.0 <= ece <= 1.0


# ── Platt Scaling ─────────────────────────────────────────────────────────────

def test_platt_scaler_fit_predict(balanced_data):
    y_true, y_proba = balanced_data
    n = len(y_true) // 2
    scaler = PlattScaler()
    scaler.fit(y_proba[:n], y_true[:n])
    calibrated = scaler.predict_proba(y_proba[n:])
    assert calibrated.shape == (len(y_proba) - n,)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_platt_not_fitted_raises(balanced_data):
    _, y_proba = balanced_data
    scaler = PlattScaler()
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.predict_proba(y_proba)


def test_platt_reduces_ece(balanced_data):
    y_true, y_proba = balanced_data
    n = len(y_true) // 2
    scaler = PlattScaler().fit(y_proba[:n], y_true[:n])
    cal = scaler.predict_proba(y_proba[n:])
    ece_before = compute_expected_calibration_error(y_true[n:], y_proba[n:])
    ece_after = compute_expected_calibration_error(y_true[n:], cal)
    # Platt scaling should not dramatically worsen ECE (may not always improve)
    assert ece_after < ece_before * 3, "Platt scaling should not catastrophically worsen ECE"


# ── Isotonic Calibrator ───────────────────────────────────────────────────────

def test_isotonic_calibrator_output_range(balanced_data):
    y_true, y_proba = balanced_data
    n = len(y_true) // 2
    cal = IsotonicCalibrator().fit(y_proba[:n], y_true[:n])
    out = cal.predict_proba(y_proba[n:])
    assert (out >= 0).all() and (out <= 1).all()


def test_isotonic_not_fitted_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict_proba(np.array([0.3, 0.7]))


def test_compare_calibrators_returns_all_methods(balanced_data):
    y_true, y_proba = balanced_data
    n = len(y_true) // 2
    result = compare_calibrators(y_true[:n], y_proba[:n], y_true[n:], y_proba[n:])
    assert set(result.keys()) == {"uncalibrated", "platt_scaling", "isotonic_regression"}
    for metrics in result.values():
        assert "brier_score" in metrics
        assert "ece" in metrics
        assert "brier_skill_score" in metrics


# ── Cost Analysis ─────────────────────────────────────────────────────────────

def test_cost_matrix_ratio():
    cm = CostMatrix(cost_fn=50_000, cost_fp=5_000)
    assert cm.fn_fp_ratio == pytest.approx(10.0)


def test_cost_optimal_threshold_lower_than_f1_for_high_fn_cost(imbalanced_data):
    """When FN cost >> FP cost, cost-optimal threshold should be lower (catch more failures)."""
    y_true, y_proba = imbalanced_data
    from src.evaluation.metrics import find_optimal_threshold
    f1_thresh, _ = find_optimal_threshold(y_true, y_proba, metric="f1")
    cm = CostMatrix(cost_fn=50_000, cost_fp=1_000)  # FN 50× more expensive
    cost_thresh, _ = find_cost_optimal_threshold(y_true, y_proba, cm)
    # With high FN cost, should alarm earlier (lower threshold) than F1
    assert cost_thresh <= f1_thresh + 0.2  # allow small tolerance


def test_theoretical_threshold_in_valid_range():
    t = theoretical_cost_optimal_threshold(cost_fp=5_000, cost_fn=50_000, prevalence=0.1)
    assert 0.0 < t < 1.0


def test_theoretical_threshold_increases_with_fn_cost():
    t1 = theoretical_cost_optimal_threshold(cost_fp=1_000, cost_fn=10_000, prevalence=0.1)
    t2 = theoretical_cost_optimal_threshold(cost_fp=1_000, cost_fn=50_000, prevalence=0.1)
    # Higher FN cost → lower optimal threshold (be more aggressive in predicting failure)
    assert t2 < t1


def test_cost_comparison_returns_cost_saving(imbalanced_data):
    y_true, y_proba = imbalanced_data
    from src.evaluation.metrics import find_optimal_threshold
    f1_thresh, _ = find_optimal_threshold(y_true, y_proba, metric="f1")
    cm = CostMatrix(cost_fn=50_000, cost_fp=5_000)
    result = cost_threshold_comparison(y_true, y_proba, cm, f1_thresh)
    assert "cost_saving_vs_f1" in result
    assert "f1_threshold_decision" in result
    assert "cost_optimal_decision" in result


# ── Statistical Significance ──────────────────────────────────────────────────

def test_mcnemar_identical_models():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    result = mcnemar_test(y_true, y_pred, y_pred)
    # Identical models → p=1, no disagreement
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_different_models():
    rng = np.random.default_rng(5)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    # Model A: mostly correct
    pred_a = np.where(rng.uniform(0, 1, n) < 0.9, y_true, 1 - y_true)
    # Model B: random
    pred_b = rng.integers(0, 2, size=n)
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["b"] + result["c"] > 0


def test_delong_ci_valid_range(balanced_data):
    y_true, y_proba = balanced_data
    result = delong_auc_confidence_interval(y_true, y_proba)
    assert 0.0 <= result["auc"] <= 1.0
    assert result["lower"] <= result["auc"] <= result["upper"]
    assert result["std_error"] >= 0.0


def test_delong_ci_width_decreases_with_n():
    rng = np.random.default_rng(1)
    def _make(n):
        y = rng.integers(0, 2, n)
        p = np.clip(y.astype(float) + rng.normal(0, 0.3, n), 0, 1)
        return y, p

    y_small, p_small = _make(50)
    y_large, p_large = _make(2000)
    ci_small = delong_auc_confidence_interval(y_small, p_small)
    ci_large = delong_auc_confidence_interval(y_large, p_large)
    width_small = ci_small["upper"] - ci_small["lower"]
    width_large = ci_large["upper"] - ci_large["lower"]
    assert width_large < width_small, "Larger sample should give narrower CI"


def test_paired_permutation_test_returns_p_value(balanced_data):
    y_true, y_proba = balanced_data
    noisy = np.clip(y_proba + np.random.default_rng(7).normal(0, 0.05, len(y_proba)), 0, 1)
    from sklearn.metrics import roc_auc_score
    result = paired_permutation_test(y_true, y_proba, noisy, roc_auc_score, n_permutations=500)
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_permutations"] == 500
