"""Train the LSTM model with full pipeline: data → train → evaluate → persist.

Usage
-----
python scripts/train_lstm.py
python scripts/train_lstm.py --data-path data/synthetic/sensor_data.csv --epochs 50
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import create_dataloaders
from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.evaluation.statistical_analysis import bootstrap_confidence_intervals
from src.evaluation.visualization import EvaluationVisualizer
from src.experiments.trainer import LSTMTrainer
from src.models.lstm_model import LSTMClassifier
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train_lstm")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM for predictive maintenance.")
    parser.add_argument("--data-path", type=str, default="data/synthetic/sensor_data.csv")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--report-dir", type=str, default="reports/lstm")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, args.model_config)

    device_str = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    logger.info(f"Loading data: {args.data_path}")
    df = pd.read_csv(args.data_path)

    preprocessor = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS,
        target_column="failure_imminent",
        window_size=cfg.data.window_size,
        step_size=cfg.data.step_size,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        scaler_type=cfg.features.scaler_type,
        random_seed=cfg.data.random_seed,
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.fit_transform(df)

    # Save preprocessor alongside model checkpoint
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    preprocessor_path = Path(args.checkpoint_dir) / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    pos_weight = preprocessor.compute_pos_weight(y_train)
    logger.info(f"Positive class weight: {pos_weight:.2f}")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=cfg.training.batch_size,
        oversample=False,  # using pos_weight in loss instead
    )

    # ── Model ────────────────────────────────────────────────────────────
    model_cfg = cfg.lstm
    model = LSTMClassifier(
        input_size=len(SENSOR_COLS),
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        bidirectional=model_cfg.bidirectional,
    )
    logger.info(f"LSTM parameters: {model.count_parameters():,}")

    # ── Training ─────────────────────────────────────────────────────────
    epochs = args.epochs or cfg.training.epochs
    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        pos_weight=pos_weight,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=cfg.experiment.experiment_name,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=cfg.training.early_stopping_patience,
        early_stopping_metric=cfg.training.early_stopping_metric,
        model_config=dict(cfg.lstm),
        training_config=dict(cfg.training),
    )

    # ── Evaluation ───────────────────────────────────────────────────────
    logger.info("Evaluating best model on test set…")
    ckpt_path = Path(args.checkpoint_dir) / "best_model.pt"
    ckpt = load_checkpoint(ckpt_path, model, device=device)

    model.eval()
    import numpy as np
    all_proba, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            proba = model.predict_proba(X_batch.to(device)).cpu().numpy()
            all_proba.append(proba)
            all_labels.append(y_batch.numpy())

    y_proba = np.concatenate(all_proba)
    y_true = np.concatenate(all_labels).astype(int)

    opt_thresh, _ = find_optimal_threshold(y_true, y_proba, metric="f1")
    metrics = compute_classification_metrics(y_true, y_proba, threshold=opt_thresh)

    logger.info(f"\nTest metrics @ threshold={opt_thresh:.3f}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    # Bootstrap confidence intervals
    logger.info("Computing bootstrap confidence intervals…")
    ci = bootstrap_confidence_intervals(y_true, y_proba, n_samples=500, threshold=opt_thresh)
    logger.info("Bootstrap 95% CI:")
    for metric_name in ["f1", "roc_auc", "pr_auc", "recall"]:
        if metric_name in ci:
            d = ci[metric_name]
            logger.info(f"  {metric_name}: {d['mean']:.4f} [{d['lower']:.4f}, {d['upper']:.4f}]")

    # ── Visualisations ───────────────────────────────────────────────────
    viz = EvaluationVisualizer(output_dir=args.report_dir)

    viz.plot_training_history(history, save_name="training_history.png")
    viz.plot_roc_curve(y_true, {"LSTM": y_proba}, save_name="roc_curve.png")
    viz.plot_pr_curve(y_true, {"LSTM": y_proba}, save_name="pr_curve.png")
    viz.plot_confusion_matrix(
        y_true, y_proba, threshold=opt_thresh,
        model_name="LSTM", save_name="confusion_matrix.png"
    )
    viz.plot_threshold_analysis(y_true, y_proba, model_name="LSTM")
    viz.plot_prediction_distribution(y_true, y_proba, model_name="LSTM")
    viz.plot_calibration(y_true, {"LSTM": y_proba})

    logger.info(f"Reports saved to {args.report_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
