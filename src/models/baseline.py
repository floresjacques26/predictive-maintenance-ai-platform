"""Sklearn-based baseline models for predictive maintenance.

Both baselines consume *flattened* windows (N, window_size * n_features)
rather than raw 3-D sequences, making them a fair comparison point against
the LSTM which processes the full temporal structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _flatten(X: np.ndarray) -> np.ndarray:
    """Reshape (N, T, F) → (N, T*F) for sklearn models."""
    n = X.shape[0]
    return X.reshape(n, -1)


class RandomForestBaseline:
    """Random Forest classifier on flattened time-series windows.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (None = unlimited).
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf node.
        max_features: Feature sub-sampling strategy.
        class_weight: Class weighting for imbalance.
        random_state: Reproducibility seed.
        n_jobs: Parallelism (-1 = all cores).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        """Train on windowed features.

        Args:
            X: Shape (N, T, F).
            y: Shape (N,).
        """
        logger.info(f"Training RandomForest on {len(X):,} samples…")
        self.model.fit(_flatten(X), y)
        logger.info("RandomForest training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities, shape (N,)."""
        return self.model.predict_proba(_flatten(X))[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"RandomForestBaseline saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestBaseline":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        logger.info(f"RandomForestBaseline loaded from {path}")
        return obj


class LogisticRegressionBaseline:
    """Logistic Regression on standardised-flattened windows.

    A simple linear baseline useful for establishing a lower bound
    and verifying that the data contains learnable signal at all.

    Args:
        C: Inverse regularisation strength.
        max_iter: Solver maximum iterations.
        class_weight: Class weighting for imbalance.
        random_state: Reproducibility seed.
        solver: Solver algorithm.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
        solver: str = "lbfgs",
    ) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        random_state=random_state,
                        solver=solver,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        logger.info(f"Training LogisticRegression on {len(X):,} samples…")
        self.pipeline.fit(_flatten(X), y)
        logger.info("LogisticRegression training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(_flatten(X))[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"LogisticRegressionBaseline saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionBaseline":
        obj = cls.__new__(cls)
        obj.pipeline = joblib.load(path)
        logger.info(f"LogisticRegressionBaseline loaded from {path}")
        return obj
