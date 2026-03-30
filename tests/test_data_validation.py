"""Tests for synthetic data generation and structural integrity."""

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import MachineType, SyntheticSensorDataGenerator

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]
VALID_MACHINE_TYPES = {mt.value for mt in MachineType}


@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    gen = SyntheticSensorDataGenerator(n_machines=10, failure_horizon=20, random_seed=0)
    return gen.generate()


def test_required_columns_present(small_dataset):
    required = (
        {"machine_id", "timestep", "failure_event", "failure_imminent",
         "machine_type", "degradation_progress"}
        | set(SENSOR_COLS)
    )
    assert required.issubset(set(small_dataset.columns))


def test_no_nan_values(small_dataset):
    assert not small_dataset[SENSOR_COLS].isnull().any().any()


def test_labels_binary(small_dataset):
    assert set(small_dataset["failure_imminent"].unique()).issubset({0, 1})
    assert set(small_dataset["failure_event"].unique()).issubset({0, 1})


def test_sensor_values_positive(small_dataset):
    for col in SENSOR_COLS:
        assert (small_dataset[col] > 0).all(), f"{col} contains non-positive values"


def test_each_machine_has_exactly_one_failure_event(small_dataset):
    events_per_machine = small_dataset.groupby("machine_id")["failure_event"].sum()
    assert (events_per_machine == 1).all(), "Each machine should have exactly one failure event"


def test_failure_imminent_is_superset_of_failure_event(small_dataset):
    # Every failure_event=1 row must also have failure_imminent=1
    fail_rows = small_dataset[small_dataset["failure_event"] == 1]
    assert (fail_rows["failure_imminent"] == 1).all()


def test_n_machines_matches_request():
    for n in [5, 20, 50]:
        gen = SyntheticSensorDataGenerator(n_machines=n, random_seed=1)
        df = gen.generate()
        assert df["machine_id"].nunique() == n


def test_reproducibility():
    gen1 = SyntheticSensorDataGenerator(n_machines=5, random_seed=42)
    gen2 = SyntheticSensorDataGenerator(n_machines=5, random_seed=42)
    df1 = gen1.generate()
    df2 = gen2.generate()
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    gen1 = SyntheticSensorDataGenerator(n_machines=5, random_seed=1)
    gen2 = SyntheticSensorDataGenerator(n_machines=5, random_seed=2)
    df1 = gen1.generate()
    df2 = gen2.generate()
    assert not df1["temperature"].equals(df2["temperature"])


def test_failure_horizon_affects_label_distribution():
    gen_short = SyntheticSensorDataGenerator(n_machines=20, failure_horizon=10, random_seed=42)
    gen_long = SyntheticSensorDataGenerator(n_machines=20, failure_horizon=50, random_seed=42)
    rate_short = gen_short.generate()["failure_imminent"].mean()
    rate_long = gen_long.generate()["failure_imminent"].mean()
    assert rate_long > rate_short, "Longer horizon should yield more positive labels"


def test_timestep_column_is_monotonic_per_machine(small_dataset):
    for _, group in small_dataset.groupby("machine_id"):
        assert group["timestep"].is_monotonic_increasing


def test_machine_type_column_valid(small_dataset):
    actual = set(small_dataset["machine_type"].unique())
    assert actual.issubset(VALID_MACHINE_TYPES), f"Unexpected machine types: {actual - VALID_MACHINE_TYPES}"


def test_machine_type_diversity(small_dataset):
    # With 10 machines and 4 types, we expect at least 2 distinct types
    n_types = small_dataset["machine_type"].nunique()
    assert n_types >= 2, f"Expected type diversity, got only {n_types} type(s)"


def test_degradation_progress_in_unit_interval(small_dataset):
    prog = small_dataset["degradation_progress"]
    assert (prog >= 0.0).all(), "degradation_progress must be >= 0"
    assert (prog <= 1.0).all(), "degradation_progress must be <= 1"


def test_failure_not_always_at_last_timestep():
    # With enough machines, at least one should fail before the final timestep
    gen = SyntheticSensorDataGenerator(n_machines=30, failure_horizon=10, random_seed=7)
    df = gen.generate()
    last_timesteps = df.groupby("machine_id")["timestep"].max()
    failure_timesteps = (
        df[df["failure_event"] == 1]
        .groupby("machine_id")["timestep"]
        .first()
    )
    is_last = failure_timesteps == last_timesteps
    assert not is_last.all(), "All failures are at the last timestep — variable position is not working"
