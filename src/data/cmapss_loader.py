"""NASA CMAPSS Turbofan Engine Degradation dataset loader.

Dataset
-------
C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) contains
run-to-failure sensor data from simulated turbofan engines.

Download
--------
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

Place the extracted .txt files in data/cmapss/:
  train_FD001.txt   test_FD001.txt   RUL_FD001.txt
  train_FD002.txt   test_FD002.txt   RUL_FD002.txt
  train_FD003.txt   test_FD003.txt   RUL_FD003.txt
  train_FD004.txt   test_FD004.txt   RUL_FD004.txt

File format
-----------
Space-delimited, no header.  26 columns per row:
  col 1   : unit_number  (engine id)
  col 2   : time_in_cycles
  cols 3-5: op_setting_1, op_setting_2, op_setting_3
  cols 6-26: sensor_1 … sensor_21

Sub-datasets
------------
  FD001: 1 fault mode (HPC degradation), 1 operating condition  — 100 train engines
  FD002: 1 fault mode,                   6 operating conditions — 260 train engines
  FD003: 2 fault modes (HPC + fan),      1 operating condition  — 100 train engines
  FD004: 2 fault modes,                  6 operating conditions — 249 train engines

Target construction
-------------------
For binary classification we compute:
  RUL(t) = max_cycle_per_engine - t          (train set, run-to-failure)
  failure_imminent(t) = 1 if RUL(t) <= failure_horizon else 0
  failure_event(t) = 1 if t == max_cycle_per_engine else 0

Output schema (identical to SyntheticSensorDataGenerator v2)
-------------------------------------------------------------
  machine_id, timestep, <sensor_cols>,
  failure_imminent, failure_event,
  machine_type, degradation_progress, rul

This makes CMAPSS a drop-in replacement for the synthetic dataset
throughout the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_RAW_COLUMNS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

# Sensors whose variance is near-zero under steady-state conditions.
# Identified on FD001 training data; consistent across all four subsets.
# Keeping them adds noise without signal.
NEAR_CONSTANT_SENSORS: frozenset[int] = frozenset({1, 5, 6, 10, 16, 18, 19})

# Default informative sensors (all 21 minus near-constant)
DEFAULT_SENSOR_INDICES: list[int] = sorted(
    set(range(1, 22)) - NEAR_CONSTANT_SENSORS
)  # → [2,3,4,7,8,9,11,12,13,14,15,17,20,21]

# Subsets with 6 operating conditions — operational settings carry information
MULTI_CONDITION_SUBSETS: frozenset[str] = frozenset({"FD002", "FD004"})

VALID_SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")


# ──────────────────────────────────────────────────────────────────────────────
# Public loader class
# ──────────────────────────────────────────────────────────────────────────────

class CMAPSSLoader:
    """Load and preprocess NASA CMAPSS data into the project's standard format.

    The output DataFrame is schema-compatible with ``SyntheticSensorDataGenerator``,
    so it can be passed directly into ``SensorDataPreprocessor.fit_transform()``.

    Args:
        data_dir: Directory containing raw CMAPSS .txt files.
        subset: Sub-dataset to load: "FD001", "FD002", "FD003", or "FD004".
        failure_horizon: Number of cycles ahead to label as failure_imminent=1.
        sensor_indices: Sensor numbers to include (1-based).  Defaults to all
            non-near-constant sensors.
        include_op_settings: Include operational settings as input features.
            ``None`` (default) auto-enables for multi-condition subsets (FD002, FD004).
        clip_rul: Cap maximum RUL at this value.  A piece-wise linear RUL target
            (no degradation assumed before clip_rul) is standard in CMAPSS literature.
            Set ``None`` to disable.  Default 125.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/cmapss",
        subset: str = "FD001",
        failure_horizon: int = 30,
        sensor_indices: list[int] | None = None,
        include_op_settings: bool | None = None,
        clip_rul: int | None = 125,
    ) -> None:
        subset = subset.upper()
        if subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset!r}")

        self.data_dir = Path(data_dir)
        self.subset = subset
        self.failure_horizon = failure_horizon
        self.sensor_indices = sensor_indices if sensor_indices is not None else DEFAULT_SENSOR_INDICES
        self.clip_rul = clip_rul

        # Auto-detect multi-condition subsets
        if include_op_settings is None:
            self.include_op_settings = subset in MULTI_CONDITION_SUBSETS
        else:
            self.include_op_settings = include_op_settings

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sensor_columns(self) -> list[str]:
        """Feature column names in the output DataFrame."""
        cols: list[str] = []
        if self.include_op_settings:
            cols += ["op_setting_1", "op_setting_2"]
        cols += [f"sensor_{i:02d}" for i in self.sensor_indices]
        return cols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_train(self) -> pd.DataFrame:
        """Load train split (run-to-failure trajectories).

        RUL is computed as ``max_cycle_per_engine - current_cycle``.

        Returns:
            Standard project DataFrame (see module docstring for schema).
        """
        df = self._read_raw(f"train_{self.subset}.txt")
        df = self._add_rul_from_data(df)
        df = self._apply_clip(df)
        df = self._add_binary_labels(df)
        df = self._add_metadata(df)
        df = self._select_output_columns(df)
        logger.info(
            f"CMAPSS {self.subset} train: {df['machine_id'].nunique()} engines | "
            f"{len(df):,} rows | failure_imminent={df['failure_imminent'].mean():.2%}"
        )
        return df

    def load_test(self) -> pd.DataFrame:
        """Load test split using ground-truth RUL from RUL_FD00X.txt.

        CMAPSS test files are truncated (we don't observe the actual failure).
        The true RUL for the last observed cycle of each engine is provided in
        the companion RUL file.  We reconstruct per-cycle RUL backwards from there.

        Returns:
            Standard project DataFrame.
        """
        df = self._read_raw(f"test_{self.subset}.txt")
        rul_series = self._read_rul_file(f"RUL_{self.subset}.txt")
        df = self._add_rul_from_rul_file(df, rul_series)
        df = self._apply_clip(df)
        df = self._add_binary_labels(df)
        df = self._add_metadata(df)
        df = self._select_output_columns(df)
        logger.info(
            f"CMAPSS {self.subset} test: {df['machine_id'].nunique()} engines | "
            f"{len(df):,} rows | failure_imminent={df['failure_imminent'].mean():.2%}"
        )
        return df

    def files_present(self) -> dict[str, bool]:
        """Return which CMAPSS files exist in ``data_dir``."""
        return {
            fname: (self.data_dir / fname).exists()
            for fname in [
                f"train_{self.subset}.txt",
                f"test_{self.subset}.txt",
                f"RUL_{self.subset}.txt",
            ]
        }

    def summary(self, split: str = "train") -> str:
        """Human-readable summary of the dataset after loading."""
        try:
            df = self.load_train() if split == "train" else self.load_test()
        except FileNotFoundError as e:
            return str(e)

        units = df["machine_id"].nunique()
        pos_rate = df["failure_imminent"].mean()
        max_rul = df["rul"].max()
        avg_life = df.groupby("machine_id")["timestep"].max().mean()
        lines = [
            f"CMAPSS {self.subset} ({split})",
            "─" * 42,
            f"  Engines              : {units}",
            f"  Total cycles (rows)  : {len(df):,}",
            f"  Avg lifecycle        : {avg_life:.0f} cycles",
            f"  Max RUL (post-clip)  : {max_rul:.0f} cycles",
            f"  Failure horizon      : {self.failure_horizon} cycles",
            f"  Positive rate        : {pos_rate:.2%}",
            f"  Feature columns ({len(self.sensor_columns):2d}) : {self.sensor_columns}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_raw(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"CMAPSS file not found: {path}\n\n"
                f"Download the dataset:\n"
                f"  curl -L -o cmapss.zip https://phm-datasets.s3.amazonaws.com/NASA/"
                f"6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip\n"
                f"  unzip cmapss.zip -d data/cmapss/\n\n"
                f"Expected location: {self.data_dir.resolve()}"
            )
        df = pd.read_csv(path, sep=r"\s+", header=None, names=_RAW_COLUMNS)
        # Drop trailing empty columns that occasionally appear in the raw files
        df = df.dropna(axis=1, how="all")
        return df.rename(columns={"unit_number": "machine_id", "time_in_cycles": "timestep"})

    def _read_rul_file(self, filename: str) -> pd.Series:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"RUL file not found: {path}")
        return pd.read_csv(path, header=None, names=["rul"]).squeeze()

    def _add_rul_from_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RUL from max cycle per engine (train split)."""
        max_cycle = df.groupby("machine_id")["timestep"].transform("max")
        df = df.copy()
        df["rul"] = max_cycle - df["timestep"]
        return df

    def _add_rul_from_rul_file(
        self, df: pd.DataFrame, rul_series: pd.Series
    ) -> pd.DataFrame:
        """Reconstruct per-cycle RUL using ground-truth last-cycle RUL (test split).

        For each engine, the RUL at its last observed cycle is given.
        Earlier cycles have RUL = rul_at_last + (last_cycle - current_cycle).
        """
        units = df["machine_id"].unique()
        if len(units) != len(rul_series):
            raise ValueError(
                f"RUL file has {len(rul_series)} entries but test data "
                f"has {len(units)} engines."
            )
        rul_map: dict[int, int] = dict(zip(units, rul_series.values))
        df = df.copy()

        ruls: list[int] = []
        for uid, group in df.groupby("machine_id", sort=False):
            last_cycle = int(group["timestep"].max())
            last_rul = int(rul_map[uid])
            for t in group["timestep"]:
                ruls.append(last_rul + (last_cycle - int(t)))
        df["rul"] = ruls
        return df

    def _apply_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.clip_rul is not None:
            df = df.copy()
            df["rul"] = df["rul"].clip(upper=self.clip_rul)
        return df

    def _add_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["failure_imminent"] = (df["rul"] <= self.failure_horizon).astype(int)
        # failure_event = 1 only at the actual last observed cycle (proxy for failure)
        last_cycle = df.groupby("machine_id")["timestep"].transform("max")
        df["failure_event"] = (df["timestep"] == last_cycle).astype(int)
        return df

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add machine_type and degradation_progress (compatible with error analysis)."""
        df = df.copy()
        df["machine_type"] = self.subset
        # Normalised position in lifecycle: 0=start, 1=end-of-observation
        max_cycle = df.groupby("machine_id")["timestep"].transform("max")
        df["degradation_progress"] = (df["timestep"] / max_cycle).clip(0.0, 1.0)
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = (
            ["machine_id", "timestep"]
            + self.sensor_columns
            + ["failure_imminent", "failure_event", "machine_type", "degradation_progress", "rul"]
        )
        return df[keep].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def load_cmapss(
    subset: str = "FD001",
    data_dir: str | Path = "data/cmapss",
    failure_horizon: int = 30,
    clip_rul: int | None = 125,
    sensor_indices: list[int] | None = None,
    include_op_settings: bool | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load CMAPSS train split and return (df, sensor_columns).

    Convenience wrapper around ``CMAPSSLoader`` for use in training scripts.

    Returns:
        df: Standard project DataFrame.
        sensor_cols: List of feature column names used in df.
    """
    loader = CMAPSSLoader(
        data_dir=data_dir,
        subset=subset,
        failure_horizon=failure_horizon,
        sensor_indices=sensor_indices,
        include_op_settings=include_op_settings,
        clip_rul=clip_rul,
    )
    return loader.load_train(), loader.sensor_columns
