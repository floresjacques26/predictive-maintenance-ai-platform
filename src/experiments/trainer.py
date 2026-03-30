"""LSTM training loop with early stopping, checkpointing, and MLflow tracking.

Design notes
------------
* Raw logits + BCEWithLogitsLoss is used instead of sigmoid + BCELoss for
  numerical stability (fused log-sum-exp prevents underflow at extreme logits).
* ``pos_weight`` is passed to the loss to up-weight the minority (failure) class.
* Early stopping monitors validation F1 (configurable) with patience.
* Gradient clipping prevents exploding gradients common in RNNs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics
from src.utils.checkpointing import save_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        mode: 'max' for metrics like F1/AUC, 'min' for loss.
        delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 15, mode: str = "max", delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.delta
        return score < self.best_score - self.delta


class LSTMTrainer:
    """Manages the complete training lifecycle for LSTMClassifier.

    Args:
        model: LSTMClassifier instance.
        device: torch device.
        learning_rate: Initial LR for AdamW.
        weight_decay: L2 penalty coefficient.
        pos_weight: BCEWithLogitsLoss positive class weight (handles imbalance).
        gradient_clip_norm: Max gradient norm; 0 disables clipping.
        checkpoint_dir: Directory to save best checkpoint.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str | Path = "models/checkpoints",
        experiment_name: str = "predictive_maintenance",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        pw = torch.tensor([pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        mlflow.set_experiment(experiment_name)
        self.run_id: Optional[str] = None
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "val_roc_auc": [],
            "val_pr_auc": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        early_stopping_metric: str = "val_f1",
        model_config: Optional[dict] = None,
        training_config: Optional[dict] = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            early_stopping_metric: Metric key to monitor.
            model_config: Config dict logged to MLflow.
            training_config: Config dict logged to MLflow.

        Returns:
            Training history dict.
        """
        # Determine early stopping mode from the metric name
        es_mode = "min" if "loss" in early_stopping_metric else "max"
        early_stopping = EarlyStopping(patience=patience, mode=es_mode)
        best_val_score = -1.0

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            self._log_params(model_config, training_config)

            logger.info(f"Training started | device={self.device} | MLflow run: {self.run_id}")

            for epoch in range(1, epochs + 1):
                train_metrics = self._train_epoch(train_loader)
                val_metrics = self._validate(val_loader)

                self._update_history(train_metrics, val_metrics)

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["lr"].append(current_lr)
                self.scheduler.step(val_metrics["f1"])

                self._log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)

                val_score = val_metrics[early_stopping_metric.replace("val_", "")]
                improved = early_stopping(val_score)

                if improved:
                    best_val_score = val_score
                    self._save_best_checkpoint(epoch, val_metrics, model_config or {})
                    mlflow.log_metric("best_val_f1", best_val_score, step=epoch)

                if early_stopping.should_stop:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

            self._log_final_artifacts()
            logger.info(f"Training complete. Best {early_stopping_metric}: {best_val_score:.4f}")

        return self.history

    # ------------------------------------------------------------------
    # Epoch-level methods
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_labels: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)

            with torch.no_grad():
                proba = torch.sigmoid(logits).cpu().numpy()
            all_proba.append(proba)
            all_labels.append(y_batch.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_proba = np.concatenate(all_proba)
        metrics = compute_classification_metrics(y_true, y_proba)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            proba = torch.sigmoid(logits).cpu().numpy()
            all_proba.append(proba)
            all_labels.append(y_batch.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_proba = np.concatenate(all_proba)
        metrics = compute_classification_metrics(y_true, y_proba)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _log_params(
        self,
        model_config: Optional[dict],
        training_config: Optional[dict],
    ) -> None:
        params: dict = {}
        if model_config:
            params.update({f"model/{k}": v for k, v in model_config.items()})
        if training_config:
            params.update({f"train/{k}": v for k, v in training_config.items()})
        mlflow.log_params(params)

    def _log_epoch_metrics(
        self,
        epoch: int,
        train: dict[str, float],
        val: dict[str, float],
        lr: float,
    ) -> None:
        metrics = {
            "train_loss": train["loss"],
            "train_f1": train["f1"],
            "val_loss": val["loss"],
            "val_f1": val["f1"],
            "val_roc_auc": val["roc_auc"],
            "val_pr_auc": val["pr_auc"],
            "lr": lr,
        }
        mlflow.log_metrics(metrics, step=epoch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d} | "
                f"train_loss={train['loss']:.4f} train_f1={train['f1']:.4f} | "
                f"val_loss={val['loss']:.4f} val_f1={val['f1']:.4f} "
                f"val_auc={val['roc_auc']:.4f} | lr={lr:.2e}"
            )

    def _log_final_artifacts(self) -> None:
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path), artifact_path="model")

    def _update_history(
        self, train: dict[str, float], val: dict[str, float]
    ) -> None:
        self.history["train_loss"].append(train["loss"])
        self.history["train_f1"].append(train["f1"])
        self.history["val_loss"].append(val["loss"])
        self.history["val_f1"].append(val["f1"])
        self.history["val_roc_auc"].append(val["roc_auc"])
        self.history["val_pr_auc"].append(val["pr_auc"])

    def _save_best_checkpoint(
        self, epoch: int, val_metrics: dict[str, float], config: dict
    ) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=val_metrics,
            config=config,
            path=self.checkpoint_dir / "best_model.pt",
        )
