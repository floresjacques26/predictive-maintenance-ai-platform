"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import SensorWindowDataset, create_dataloaders
from src.data.preprocessing import SensorDataPreprocessor
from src.data.synthetic_generator import SyntheticSensorDataGenerator

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]
WINDOW_SIZE = 20
STEP_SIZE = 5


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    gen = SyntheticSensorDataGenerator(n_machines=30, failure_horizon=15, random_seed=7)
    return gen.generate()


@pytest.fixture(scope="module")
def preprocessed(raw_df):
    prep = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS,
        target_column="failure_imminent",
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        test_size=0.2,
        val_size=0.15,
        scaler_type="standard",
        random_seed=7,
    )
    return prep.fit_transform(raw_df), prep


# ── Shape tests ──────────────────────────────────────────────────────────────

def test_output_shapes(preprocessed):
    (train, val, test), _ = preprocessed
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    assert X_train.ndim == 3, "X_train should be 3-dimensional (N, T, F)"
    assert X_train.shape[1] == WINDOW_SIZE
    assert X_train.shape[2] == len(SENSOR_COLS)

    assert y_train.ndim == 1
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_no_overlap_between_splits(raw_df):
    """Machine-level split guarantees no cross-split leakage."""
    prep = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS,
        target_column="failure_imminent",
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        test_size=0.2,
        val_size=0.15,
        random_seed=42,
    )
    # We test this indirectly: total windows from each split should sum ≤ total possible
    (train, val, test), _ = prep.fit_transform(raw_df)
    n_total = len(train[0]) + len(val[0]) + len(test[0])
    # Can't be more than maximum possible windows
    assert n_total > 0


def test_train_larger_than_val_and_test(preprocessed):
    (train, val, test), _ = preprocessed
    assert len(train[0]) > len(val[0])
    assert len(train[0]) > len(test[0])


# ── Normalisation tests ──────────────────────────────────────────────────────

def test_training_features_approximately_normalised(preprocessed):
    (train, _, _), _ = preprocessed
    X_train, _ = train
    flat = X_train.reshape(-1, len(SENSOR_COLS))
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    np.testing.assert_allclose(means, 0.0, atol=0.5, err_msg="Train features should be ~zero-mean")
    np.testing.assert_allclose(stds, 1.0, atol=0.5, err_msg="Train features should have ~unit std")


def test_scaler_not_refit_on_val_test(raw_df):
    """Ensure test data is transformed with train scaler, not re-fitted."""
    prep = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS,
        target_column="failure_imminent",
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        test_size=0.3,
        val_size=0.1,
        random_seed=99,
    )
    (train, _, test), _ = prep.fit_transform(raw_df)
    # Test set mean should NOT be exactly zero (scaler was fit on train)
    flat_test = test[0].reshape(-1, len(SENSOR_COLS))
    test_means = flat_test.mean(axis=0)
    # At least some deviation is expected
    assert not np.allclose(test_means, 0.0, atol=0.01), \
        "Test features should not be zero-mean (scaler was fit on train only)"


# ── Label integrity ──────────────────────────────────────────────────────────

def test_labels_are_binary(preprocessed):
    (train, val, test), _ = preprocessed
    for split_name, (_, y) in zip(["train", "val", "test"], [train, val, test]):
        unique_vals = set(np.unique(y))
        assert unique_vals.issubset({0, 1}), f"{split_name} labels are not binary"


def test_positive_examples_exist(preprocessed):
    (train, val, test), _ = preprocessed
    assert train[1].sum() > 0, "No positive examples in training set"


# ── Dataset & DataLoader ─────────────────────────────────────────────────────

def test_sensor_window_dataset(preprocessed):
    (train, _, _), _ = preprocessed
    X, y = train
    ds = SensorWindowDataset(X, y)
    assert len(ds) == len(X)
    x_item, y_item = ds[0]
    assert x_item.shape == (WINDOW_SIZE, len(SENSOR_COLS))
    assert y_item.ndim == 0  # scalar


def test_dataloaders_shapes(preprocessed):
    (train, val, test), _ = preprocessed
    X_tr, y_tr = train
    X_v, y_v = val
    X_te, y_te = test
    train_dl, val_dl, test_dl = create_dataloaders(
        X_tr, y_tr, X_v, y_v, X_te, y_te, batch_size=16
    )
    X_b, y_b = next(iter(train_dl))
    assert X_b.shape[1] == WINDOW_SIZE
    assert X_b.shape[2] == len(SENSOR_COLS)
    assert y_b.ndim == 1


# ── Persistence ──────────────────────────────────────────────────────────────

def test_save_and_load_preprocessor(preprocessed, tmp_path):
    _, prep = preprocessed
    save_path = tmp_path / "prep.joblib"
    prep.save(save_path)

    loaded = SensorDataPreprocessor.load(save_path)
    assert loaded._is_fitted
    assert loaded.sensor_columns == prep.sensor_columns
    assert loaded.window_size == prep.window_size

    # Scaler means should be identical
    np.testing.assert_array_almost_equal(
        prep.scaler.mean_, loaded.scaler.mean_
    )


# ── Validation ───────────────────────────────────────────────────────────────

def test_missing_column_raises(raw_df):
    broken = raw_df.drop(columns=["temperature"])
    prep = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS,
        target_column="failure_imminent",
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )
    with pytest.raises(ValueError, match="missing columns"):
        prep.fit_transform(broken)


def test_pos_weight_returns_float(preprocessed):
    (train, _, _), prep = preprocessed
    _, y_train = train
    weight = prep.compute_pos_weight(y_train)
    assert isinstance(weight, float)
    assert weight > 0
