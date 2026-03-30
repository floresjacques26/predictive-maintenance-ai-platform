"""Evaluation visualisation: ROC, PR curves, confusion matrix, calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from src.evaluation.metrics import (
    compute_confusion_matrix,
    get_pr_curve_data,
    get_roc_curve_data,
    threshold_analysis,
)
from src.evaluation.statistical_analysis import calibration_analysis

# Consistent style across all plots
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class EvaluationVisualizer:
    """Factory for all evaluation plots.

    Args:
        output_dir: Directory where figures are saved.
    """

    def __init__(self, output_dir: str | Path = "reports/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_name: str = "roc_curve.png",
    ) -> Path:
        """Plot ROC curves for one or multiple models.

        Args:
            predictions: {model_name: y_pred_proba} dict.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

        for name, proba in predictions.items():
            data = get_roc_curve_data(y_true, proba)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, proba)
            ax.plot(data["fpr"], data["tpr"], lw=2, label=f"{name} (AUC = {auc:.3f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Failure Prediction")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        return self._save(fig, save_name)

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_name: str = "pr_curve.png",
    ) -> Path:
        """Precision-Recall curve — preferred for imbalanced datasets."""
        baseline = y_true.mean()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"No-skill (AP = {baseline:.3f})")

        for name, proba in predictions.items():
            data = get_pr_curve_data(y_true, proba)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, proba)
            ax.plot(data["recall"], data["precision"], lw=2, label=f"{name} (AP = {ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve — Failure Prediction")
        ax.legend(loc="upper right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        return self._save(fig, save_name)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "Model",
        save_name: str = "confusion_matrix.png",
    ) -> Path:
        """Annotated normalised + absolute confusion matrix."""
        cm = compute_confusion_matrix(y_true, y_pred_proba, threshold)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2%"],
            ["Absolute counts", "Row-normalised"],
        ):
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"],
                ax=ax,
                cbar=False,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{model_name} — {title} (threshold={threshold:.2f})")

        fig.tight_layout()
        return self._save(fig, save_name)

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save_name: str = "threshold_analysis.png",
    ) -> Path:
        """Precision, Recall, F1 vs decision threshold."""
        data = threshold_analysis(y_true, y_pred_proba)
        best_idx = np.argmax(data["f1"])
        best_t = data["thresholds"][best_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(data["thresholds"], data["precision"], label="Precision", lw=2)
        ax.plot(data["thresholds"], data["recall"], label="Recall", lw=2)
        ax.plot(data["thresholds"], data["f1"], label="F1-Score", lw=2.5)
        ax.axvline(best_t, color="red", linestyle="--", lw=1.5, label=f"Best F1 threshold = {best_t:.2f}")

        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{model_name} — Metric vs Threshold")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        return self._save(fig, save_name)

    def plot_calibration(
        self,
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_name: str = "calibration.png",
    ) -> Path:
        """Reliability diagram: predicted probability vs observed frequency."""
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

        for name, proba in predictions.items():
            cal = calibration_analysis(y_true, proba)
            mask = cal["counts"] > 0
            ax.plot(
                cal["mean_predicted"][mask],
                cal["fraction_positive"][mask],
                "o-",
                lw=2,
                label=name,
            )

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Plot (Reliability Diagram)")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        return self._save(fig, save_name)

    def plot_training_history(
        self,
        history: dict[str, list[float]],
        save_name: str = "training_history.png",
    ) -> Path:
        """Loss and F1 curves across training epochs."""
        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Loss
        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (BCE)")
        axes[0].set_title("Training / Validation Loss")
        axes[0].legend()

        # F1
        if "val_f1" in history:
            axes[1].plot(epochs, history.get("train_f1", [0] * len(epochs)), label="Train F1")
            axes[1].plot(epochs, history["val_f1"], label="Val F1")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("F1-Score")
            axes[1].set_title("F1-Score Progression")
            axes[1].legend()

        fig.tight_layout()
        return self._save(fig, save_name)

    def plot_model_comparison(
        self,
        comparison: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
        save_name: str = "model_comparison.png",
    ) -> Path:
        """Bar chart comparing multiple models across key metrics."""
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

        model_names = list(comparison.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(model_names):
            vals = [comparison[name].get(m, 0.0) for m in metrics]
            ax.bar(x + i * width, vals, width, label=name)

        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=15)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

        return self._save(fig, save_name)

    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save_name: str = "prediction_distribution.png",
    ) -> Path:
        """Histogram of predicted probabilities split by true class."""
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 40)

        ax.hist(
            y_pred_proba[y_true == 0],
            bins=bins,
            alpha=0.6,
            label="True Negative (no failure)",
            color="steelblue",
            density=True,
        )
        ax.hist(
            y_pred_proba[y_true == 1],
            bins=bins,
            alpha=0.6,
            label="True Positive (failure imminent)",
            color="crimson",
            density=True,
        )
        ax.set_xlabel("Predicted Failure Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name} — Prediction Score Distribution")
        ax.legend()

        return self._save(fig, save_name)

    def plot_cost_curves(
        self,
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        cost_matrix,
        save_name: str = "cost_curves.png",
    ) -> Path:
        """Normalised expected cost vs decision threshold for each model.

        Helps identify the cost-optimal operating point for each model.
        Cost is normalised by the worst-case (predicting all negative) so
        values are in [0, 1] regardless of absolute cost magnitudes.
        """
        from src.evaluation.cost_analysis import compute_expected_cost_curve

        fig, ax = plt.subplots(figsize=(9, 6))
        for name, proba in predictions.items():
            curve = compute_expected_cost_curve(y_true, proba, cost_matrix)
            best_idx = int(np.argmin(curve["expected_cost"]))
            best_t = float(curve["thresholds"][best_idx])
            ax.plot(
                curve["thresholds"],
                curve["normalised_cost"],
                lw=2,
                label=f"{name} (opt θ={best_t:.2f})",
            )
            ax.axvline(best_t, linestyle=":", lw=1, alpha=0.6)

        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Normalised Expected Cost")
        ax.set_title(
            f"Cost Curve — FN={cost_matrix.cost_fn:,} | FP={cost_matrix.cost_fp:,} "
            f"(ratio {cost_matrix.fn_fp_ratio:.0f}:1)"
        )
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, None])
        return self._save(fig, save_name)

    def plot_error_breakdown(
        self,
        breakdown: dict[str, dict],
        group_by: str = "machine_type",
        model_name: str = "Model",
        save_name: str = "error_breakdown.png",
    ) -> Path:
        """Bar chart of FP-rate and FN-rate by categorical group.

        Args:
            breakdown: Output of ``error_analysis_by_group``.
            group_by: Column used for grouping (label only, for title).
        """
        groups = list(breakdown.keys())
        fn_rates = [breakdown[g]["fn_rate"] for g in groups]
        fp_rates = [breakdown[g]["fp_rate"] for g in groups]

        x = np.arange(len(groups))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 5))
        ax.bar(x - width / 2, fn_rates, width, label="FN Rate (missed failures)", color="crimson", alpha=0.8)
        ax.bar(x + width / 2, fp_rates, width, label="FP Rate (false alarms)", color="steelblue", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylabel("Rate")
        ax.set_ylim([0, 1])
        ax.set_title(f"{model_name} — Error Rates by {group_by}")
        ax.legend()
        fig.tight_layout()
        return self._save(fig, save_name)

    def plot_proximity_error(
        self,
        proximity_data: dict[str, list],
        model_name: str = "Model",
        save_name: str = "proximity_error.png",
    ) -> Path:
        """FN/FP rates vs timesteps-before-failure proximity bins."""
        bins = proximity_data["bins"]
        fn_rates = proximity_data["fn_rates"]
        fp_rates = proximity_data["fp_rates"]
        n_windows = proximity_data["n_windows"]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(bins))

        ax1.bar(x, fn_rates, alpha=0.6, label="FN Rate", color="crimson")
        ax1.bar(x, fp_rates, alpha=0.6, label="FP Rate", color="steelblue", bottom=fn_rates)
        ax1.set_xticks(x)
        ax1.set_xticklabels(bins, rotation=45, ha="right")
        ax1.set_ylabel("Rate")
        ax1.set_title(f"{model_name} — Error Rates by Proximity to Failure")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(x, n_windows, "o--", color="gray", lw=1.5, label="n windows")
        ax2.set_ylabel("Window count", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        fig.tight_layout()
        return self._save(fig, save_name)

    def plot_feature_importance(
        self,
        importance_matrix: np.ndarray,
        sensor_cols: list[str],
        model_name: str = "Model",
        save_name: str = "feature_importance.png",
    ) -> Path:
        """Heatmap of (window_size, n_features) importance matrix.

        Rows = timestep position, Columns = sensor.
        Useful for understanding WHAT the model attends to AND WHEN.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap: temporal × feature
        im = axes[0].imshow(
            importance_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest"
        )
        axes[0].set_xlabel("Sensor")
        axes[0].set_ylabel("Timestep (position in window)")
        axes[0].set_xticks(range(len(sensor_cols)))
        axes[0].set_xticklabels(sensor_cols, rotation=30)
        axes[0].set_title(f"{model_name} — Temporal × Feature Importance")
        plt.colorbar(im, ax=axes[0])

        # Bar: aggregate per sensor
        sensor_importance = importance_matrix.mean(axis=0)
        sensor_importance /= sensor_importance.sum() + 1e-10
        axes[1].barh(sensor_cols, sensor_importance, color="steelblue", alpha=0.8)
        axes[1].set_xlabel("Relative Importance")
        axes[1].set_title(f"{model_name} — Per-Sensor Importance (avg over time)")
        axes[1].invert_yaxis()

        fig.tight_layout()
        return self._save(fig, save_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
