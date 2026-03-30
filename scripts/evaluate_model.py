"""Standalone evaluation script: load any saved model and produce full report.

Usage
-----
python scripts/evaluate_model.py --model-type lstm --checkpoint models/checkpoints/best_model.pt
python scripts/evaluate_model.py --model-type rf --checkpoint models/baselines/randomforest.joblib
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    threshold_analysis,
)
from src.evaluation.statistical_analysis import (
    bootstrap_confidence_intervals,
    calibration_analysis,
    error_analysis,
    prediction_distribution_analysis,
)
from src.evaluation.visualization import EvaluationVisualizer
from src.models.baseline import RandomForestBaseline
from src.models.lstm_model import LSTMClassifier
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate_model")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model-type", choices=["lstm", "rf", "lr"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preprocessor", type=str, default="models/checkpoints/preprocessor.joblib")
    parser.add_argument("--data-path", type=str, default="data/synthetic/sensor_data.csv")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--report-dir", type=str, default="reports/evaluation")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def get_predictions_lstm(
    checkpoint_path: str, preprocessor_path: str, X_test: np.ndarray
) -> np.ndarray:
    preprocessor = SensorDataPreprocessor.load(preprocessor_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = LSTMClassifier(
        input_size=len(SENSOR_COLS),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_t = torch.from_numpy(X_test.astype(np.float32))
    with torch.no_grad():
        proba = model.predict_proba(X_t).numpy()
    return proba


def get_predictions_rf(checkpoint_path: str, X_test: np.ndarray) -> np.ndarray:
    model = RandomForestBaseline.load(checkpoint_path)
    return model.predict_proba(X_test)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info(f"Loading data: {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Reconstruct the test split using the saved preprocessor.
    # IMPORTANT: we call fit_transform (not transform) because the test-set
    # indices are derived from the machine-level random split inside the
    # preprocessor.  The scaler is refitted only on the training machines —
    # the same machines selected at training time — because the same
    # random_seed is preserved in the loaded preprocessor object.
    if Path(args.preprocessor).exists():
        preprocessor = SensorDataPreprocessor.load(args.preprocessor)
    else:
        logger.warning("Preprocessor not found — creating new one with config defaults.")
        preprocessor = SensorDataPreprocessor(
            sensor_columns=SENSOR_COLS,
            target_column="failure_imminent",
            window_size=cfg.data.window_size,
            step_size=cfg.data.step_size,
            test_size=cfg.data.test_size,
            val_size=cfg.data.val_size,
            random_seed=cfg.data.random_seed,
        )

    (_, _), (_, _), (X_test, y_test) = preprocessor.fit_transform(df)

    # Get predictions
    if args.model_type == "lstm":
        y_proba = get_predictions_lstm(args.checkpoint, args.preprocessor, X_test)
    else:
        y_proba = get_predictions_rf(args.checkpoint, X_test)

    y_true = y_test.astype(int)

    # ── Full Evaluation ──────────────────────────────────────────────────
    opt_thresh, best_f1 = find_optimal_threshold(y_true, y_proba, metric="f1")
    logger.info(f"Optimal F1 threshold: {opt_thresh:.3f} (F1={best_f1:.4f})")

    metrics = compute_classification_metrics(y_true, y_proba, threshold=opt_thresh)
    logger.info("\nTest Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:20s}: {v:.4f}")

    # Bootstrap CI
    logger.info(f"\nBootstrap CI (n={args.bootstrap_samples}):")
    ci = bootstrap_confidence_intervals(
        y_true, y_proba, n_samples=args.bootstrap_samples, threshold=opt_thresh
    )
    for m in ["accuracy", "f1", "roc_auc", "pr_auc", "recall", "precision"]:
        if m in ci:
            d = ci[m]
            logger.info(f"  {m:20s}: {d['mean']:.4f} ± {d['std']:.4f}  [{d['lower']:.4f}, {d['upper']:.4f}]")

    # Prediction distribution
    dist = prediction_distribution_analysis(y_true, y_proba)
    logger.info(f"\nPrediction Distribution:")
    logger.info(f"  Positive class — mean: {dist['positive_class']['mean']:.4f}, std: {dist['positive_class']['std']:.4f}")
    logger.info(f"  Negative class — mean: {dist['negative_class']['mean']:.4f}, std: {dist['negative_class']['std']:.4f}")
    logger.info(f"  Imbalance ratio (neg/pos): {dist['class_imbalance_ratio']:.1f}x")

    # Error analysis
    errors = error_analysis(y_true, y_proba, threshold=opt_thresh)
    logger.info(f"\nError Analysis:")
    logger.info(f"  FP: {errors['fp_mask'].sum():4d} | mean proba: {errors['fp_proba'].mean():.3f}" if len(errors['fp_proba']) else "  FP: 0")
    logger.info(f"  FN: {errors['fn_mask'].sum():4d} | mean proba: {errors['fn_proba'].mean():.3f}" if len(errors['fn_proba']) else "  FN: 0")

    # ── Visualisations ───────────────────────────────────────────────────
    model_label = args.model_type.upper()
    viz = EvaluationVisualizer(output_dir=args.report_dir)
    viz.plot_roc_curve(y_true, {model_label: y_proba})
    viz.plot_pr_curve(y_true, {model_label: y_proba})
    viz.plot_confusion_matrix(y_true, y_proba, threshold=opt_thresh, model_name=model_label)
    viz.plot_threshold_analysis(y_true, y_proba, model_name=model_label)
    viz.plot_prediction_distribution(y_true, y_proba, model_name=model_label)
    viz.plot_calibration(y_true, {model_label: y_proba})

    # ── Save full report ─────────────────────────────────────────────────
    report = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "optimal_threshold": opt_thresh,
        "test_metrics": metrics,
        "bootstrap_ci": ci,
        "class_distribution": dist,
    }
    report_path = Path(args.report_dir) / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nFull report saved → {report_path}")


if __name__ == "__main__":
    main()
