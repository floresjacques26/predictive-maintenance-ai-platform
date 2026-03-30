"""Tests for schema validation and PSI drift detection."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_validator import (
    DEFAULT_SCHEMA,
    SensorSchema,
    compute_psi,
    detect_sensor_drift,
    validate_schema,
)
from src.data.synthetic_generator import SyntheticSensorDataGenerator

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]


@pytest.fixture(scope="module")
def valid_df():
    gen = SyntheticSensorDataGenerator(n_machines=10, random_seed=0)
    return gen.generate()


# ── Schema validation ─────────────────────────────────────────────────────────

def test_valid_data_passes(valid_df):
    result = validate_schema(valid_df)
    assert result.passed, f"Valid data should pass. Errors: {result.errors}"


def test_missing_column_fails():
    df = pd.DataFrame({"temperature": [80.0], "vibration": [0.5]})
    result = validate_schema(df)
    assert not result.passed
    assert any("Missing required columns" in e for e in result.errors)


def test_nan_above_threshold_fails():
    gen = SyntheticSensorDataGenerator(n_machines=5, random_seed=0)
    df = gen.generate()
    # Inject many NaNs
    df.loc[df.index[:int(len(df) * 0.1)], "temperature"] = np.nan
    result = validate_schema(df)
    assert not result.passed


def test_out_of_range_values_fail():
    gen = SyntheticSensorDataGenerator(n_machines=5, random_seed=0)
    df = gen.generate()
    # Set 5% of temperature to negative (impossible)
    n_corrupt = int(len(df) * 0.05)
    df.loc[df.index[:n_corrupt], "temperature"] = -100.0
    result = validate_schema(df)
    assert not result.passed


def test_non_binary_target_fails():
    df = pd.DataFrame({col: [1.0] for col in SENSOR_COLS})
    df["machine_id"] = 0
    df["timestep"] = 0
    df["failure_imminent"] = 2  # invalid
    result = validate_schema(df, DEFAULT_SCHEMA)
    assert not result.passed
    assert any("non-binary" in e.lower() for e in result.errors)


def test_validation_result_str():
    result = validate_schema(pd.DataFrame())
    s = str(result)
    assert "FAILED" in s or "PASSED" in s


def test_custom_schema():
    schema = SensorSchema(
        columns=["temperature"],
        target_column="label",
        value_ranges={"temperature": (0, 200)},
    )
    df = pd.DataFrame({"temperature": [100.0, 150.0], "label": [0, 1]})
    result = validate_schema(df, schema)
    assert result.passed


# ── PSI ───────────────────────────────────────────────────────────────────────

def test_psi_identical_distributions():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 1000)
    psi = compute_psi(data, data)
    assert psi < 0.01, f"Identical distributions should have PSI ≈ 0, got {psi:.4f}"


def test_psi_very_different_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(5, 1, 1000)  # shifted by 5 sigma
    psi = compute_psi(ref, cur)
    assert psi > 0.25, f"Very different distributions should have PSI > 0.25, got {psi:.4f}"


def test_psi_returns_nonnegative():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 1, 500)
    cur = rng.uniform(0.2, 0.8, 500)
    psi = compute_psi(ref, cur)
    assert psi >= 0.0


def test_psi_symmetric_approximate():
    """PSI is not perfectly symmetric, but both directions should be non-negative."""
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0.5, 1, 500)
    psi_ab = compute_psi(a, b)
    psi_ba = compute_psi(b, a)
    assert psi_ab >= 0.0
    assert psi_ba >= 0.0


# ── Drift detection ───────────────────────────────────────────────────────────

def test_detect_drift_no_drift(valid_df):
    """Same distribution → all sensors should be OK."""
    results = detect_sensor_drift(valid_df, valid_df, SENSOR_COLS)
    for col, info in results.items():
        assert info["psi"] < 0.01, f"{col} PSI should be ~0 for identical data"
        assert info["status"] == "OK"


def test_detect_drift_with_shifted_data(valid_df):
    """Strongly shifted data should trigger WARNING or CRITICAL."""
    shifted = valid_df.copy()
    shifted["temperature"] = shifted["temperature"] * 2.0 + 50.0  # large shift
    results = detect_sensor_drift(valid_df, shifted, SENSOR_COLS)
    assert results["temperature"]["status"] in ("WARNING", "CRITICAL"), \
        f"Shifted temperature should trigger drift, status={results['temperature']['status']}"


def test_detect_drift_returns_all_sensors(valid_df):
    results = detect_sensor_drift(valid_df, valid_df, SENSOR_COLS)
    assert set(results.keys()) == set(SENSOR_COLS)
    for col in SENSOR_COLS:
        assert "psi" in results[col]
        assert "status" in results[col]
        assert "reference_mean" in results[col]
        assert "current_mean" in results[col]
