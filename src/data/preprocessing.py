"""Data preprocessing pipeline: windowing, normalisation, and splitting.

Design decisions
----------------
* Windows are extracted per-machine so no window ever straddles two machines.
* Train/val/test splits are done at the **machine** level, not window level,
  preventing data leakage across machines.
* Normalisation is fit only on the training set and applied to val/test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

ScalerType = Literal["standard", "minmax", "robust"]


class SensorDataPreprocessor:
    """Transform raw sensor DataFrame into windowed numpy arrays ready for training.

    Args:
        sensor_columns: Feature column names.
        target_column: Binary label column name.
        window_size: Number of timesteps per input sample.
        step_size: Stride between consecutive windows.
        test_size: Fraction of machines held out for test.
        val_size: Fraction of train machines used for validation.
        scaler_type: Normalisation strategy.
        random_seed: Reproducibility seed.
    """

    SCALER_MAP: dict[str, type] = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(
        self,
        sensor_columns: list[str],
        target_column: str = "failure_imminent",
        window_size: int = 50,
        step_size: int = 10,
        test_size: float = 0.2,
        val_size: float = 0.1,
        scaler_type: ScalerType = "standard",
        random_seed: int = 42,
    ) -> None:
        self.sensor_columns = sensor_columns
        self.target_column = target_column
        self.window_size = window_size
        self.step_size = step_size
        self.test_size = test_size
        self.val_size = val_size
        self.scaler_type = scaler_type
        self.random_seed = random_seed

        self.scaler = self.SCALER_MAP[scaler_type]()
        self._is_fitted = False

        # Populated after fit_transform — used by downstream analysis
        self.train_machine_ids: np.ndarray = np.array([])
        self.val_machine_ids: np.ndarray = np.array([])
        self.test_machine_ids: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Split machines, fit scaler on train, produce windowed splits.

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
            where X shape is (N, window_size, n_features) and y shape (N,).
        """
        self._validate_dataframe(df)

        machine_ids = df["machine_id"].unique()
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(machine_ids)

        n_test = max(1, int(len(machine_ids) * self.test_size))
        n_val = max(1, int((len(machine_ids) - n_test) * self.val_size))

        test_ids = machine_ids[:n_test]
        val_ids = machine_ids[n_test : n_test + n_val]
        train_ids = machine_ids[n_test + n_val :]

        # Store for downstream use (error analysis, interpretability)
        self.test_machine_ids = test_ids
        self.val_machine_ids = val_ids
        self.train_machine_ids = train_ids

        logger.info(
            f"Machine split — train: {len(train_ids)}, "
            f"val: {len(val_ids)}, test: {len(test_ids)}"
        )

        train_df = df[df["machine_id"].isin(train_ids)]
        val_df = df[df["machine_id"].isin(val_ids)]
        test_df = df[df["machine_id"].isin(test_ids)]

        # Fit scaler only on training features
        self.scaler.fit(train_df[self.sensor_columns].values)
        self._is_fitted = True

        train_data = self._df_to_windows(train_df, fit_scaler=False)
        val_data = self._df_to_windows(val_df, fit_scaler=False)
        test_data = self._df_to_windows(test_df, fit_scaler=False)

        self._log_split_stats("train", train_data)
        self._log_split_stats("val", val_data)
        self._log_split_stats("test", test_data)

        return train_data, val_data, test_data

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Apply fitted scaler and extract windows (inference-time use)."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first.")
        return self._df_to_windows(df, fit_scaler=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist scaler and config to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "sensor_columns": self.sensor_columns,
                "target_column": self.target_column,
                "window_size": self.window_size,
                "step_size": self.step_size,
                "scaler_type": self.scaler_type,
            },
            path,
        )
        logger.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SensorDataPreprocessor":
        """Load a previously saved preprocessor."""
        data = joblib.load(path)
        obj = cls(
            sensor_columns=data["sensor_columns"],
            target_column=data["target_column"],
            window_size=data["window_size"],
            step_size=data["step_size"],
            scaler_type=data["scaler_type"],
        )
        obj.scaler = data["scaler"]
        obj._is_fitted = True
        logger.info(f"Preprocessor loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _df_to_windows(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract sliding windows grouped by machine_id."""
        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for machine_id, group in df.groupby("machine_id"):
            group = group.sort_values("timestep")
            features = group[self.sensor_columns].values.astype(np.float32)
            labels = group[self.target_column].values.astype(np.int64)

            if fit_scaler:
                features = self.scaler.fit_transform(features)
            elif self._is_fitted:
                features = self.scaler.transform(features)

            n_timesteps = len(features)
            for start in range(0, n_timesteps - self.window_size + 1, self.step_size):
                end = start + self.window_size
                X_list.append(features[start:end])
                # Label is determined by the last timestep in the window
                y_list.append(int(labels[end - 1]))

        if not X_list:
            return np.empty((0, self.window_size, len(self.sensor_columns))), np.empty(0)

        return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        required = {"machine_id", "timestep", self.target_column} | set(self.sensor_columns)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        if df[self.sensor_columns].isnull().any().any():
            raise ValueError("Sensor columns contain NaN values.")

    @staticmethod
    def _log_split_stats(name: str, data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = data
        pos_rate = y.mean() if len(y) > 0 else 0.0
        logger.info(f"  {name}: {len(X):,} windows, failure-imminent rate: {pos_rate:.2%}")

    def compute_pos_weight(self, y_train: np.ndarray) -> float:
        """Compute BCEWithLogitsLoss pos_weight to handle class imbalance.

        Returns:
            n_negative / n_positive (safe for zero-division).
        """
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos == 0:
            return 1.0
        return float(n_neg / n_pos)
