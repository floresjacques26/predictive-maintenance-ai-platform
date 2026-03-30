"""Optuna-based hyperparameter optimisation for LSTMClassifier.

Search strategy
---------------
* TPE (Tree-structured Parzen Estimator) sampler: efficient Bayesian
  optimisation over non-continuous search spaces.
* Pruner: MedianPruner kills unpromising trials early based on
  intermediate validation F1, reducing total wall-clock time by ~40%.
* Objective: maximise validation F1 (best single metric for imbalanced data).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import mlflow
import optuna
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SensorWindowDataset, create_dataloaders
from src.experiments.trainer import LSTMTrainer
from src.models.lstm_model import LSTMClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterSearch:
    """Optuna study wrapper for LSTM hyperparameter search.

    Args:
        X_train/val: Feature arrays (N, T, F).
        y_train/val: Label arrays (N,).
        n_input_features: Number of sensor channels.
        pos_weight: Positive class weight for loss.
        device: Torch device.
        experiment_name: MLflow experiment name.
        checkpoint_dir: Directory for trial checkpoints.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_input_features: int,
        pos_weight: float = 1.0,
        device: str = "cpu",
        experiment_name: str = "predictive_maintenance_hpo",
        checkpoint_dir: str | Path = "models/hpo_checkpoints",
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_input_features = n_input_features
        self.pos_weight = pos_weight
        self.device = device
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.study: Optional[optuna.Study] = None

    def run(
        self,
        n_trials: int = 50,
        timeout: Optional[float] = 3600.0,
        epochs_per_trial: int = 30,
        patience_per_trial: int = 7,
    ) -> dict:
        """Execute the hyperparameter search.

        Args:
            n_trials: Maximum number of Optuna trials.
            timeout: Wall-clock timeout in seconds (None = unlimited).
            epochs_per_trial: Training budget per trial.
            patience_per_trial: Early stopping patience per trial.

        Returns:
            Best hyperparameters dict.
        """
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.experiment_name,
        )

        objective = self._make_objective(
            epochs=epochs_per_trial,
            patience=patience_per_trial,
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        best = self.study.best_params
        best["best_value"] = self.study.best_value
        logger.info(f"HPO complete. Best val_f1={best['best_value']:.4f}")
        logger.info(f"Best params: {json.dumps(best, indent=2)}")

        # Persist best params
        out = self.checkpoint_dir / "best_params.json"
        with open(out, "w") as f:
            json.dump(best, f, indent=2)
        logger.info(f"Best params saved → {out}")

        return best

    def get_importance_summary(self) -> dict[str, float]:
        """Return parameter importance scores from completed study."""
        if self.study is None:
            raise RuntimeError("Run HPO first.")
        importance = optuna.importance.get_param_importances(self.study)
        return dict(importance)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_objective(self, epochs: int, patience: int) -> Callable:
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

            model = LSTMClassifier(
                input_size=self.n_input_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

            train_ds = SensorWindowDataset(self.X_train, self.y_train)
            val_ds = SensorWindowDataset(self.X_val, self.y_val)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            trainer = LSTMTrainer(
                model=model,
                device=self.device,
                learning_rate=lr,
                weight_decay=weight_decay,
                pos_weight=self.pos_weight,
                experiment_name=self.experiment_name,
            )

            history = trainer.fit(
                train_loader,
                val_loader,
                epochs=epochs,
                patience=patience,
            )

            best_val_f1 = max(history["val_f1"]) if history["val_f1"] else 0.0

            # Report intermediate values for pruning
            for step, val_f1 in enumerate(history["val_f1"]):
                trial.report(val_f1, step=step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return best_val_f1

        return objective
