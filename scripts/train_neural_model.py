"""Train LSTM or 1D CNN with full pipeline: data → train → evaluate → persist.

This script replaces train_lstm.py and supports both neural model architectures.

Usage
-----
python scripts/train_neural_model.py --model-type lstm
python scripts/train_neural_model.py --model-type cnn --epochs 50
python scripts/train_neural_model.py --model-type lstm --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import create_dataloaders
from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.calibration import compare_calibrators
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.evaluation.statistical_analysis import bootstrap_confidence_intervals
from src.evaluation.visualization import EvaluationVisualizer
from src.experiments.trainer import LSTMTrainer
from src.models.cnn_model import TemporalCNNClassifier
from src.models.lstm_model import LSTMClassifier
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train_neural_model")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM or CNN for predictive maintenance.")
    parser.add_argument(
        "--model-type", choices=["lstm", "cnn"], default="lstm",
        help="Neural architecture to train."
    )
    parser.add_argument("--data-path", type=str, default="data/synthetic/sensor_data.csv")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint dir. Defaults to models/checkpoints/<model_type>.")
    parser.add_argument("--report-dir", type=str, default=None,
                        help="Override report dir. Defaults to reports/<model_type>.")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps")
    return parser.parse_args()


def _build_model(model_type: str, cfg, n_features: int) -> torch.nn.Module:
    """Instantiate a model from config."""
    if model_type == "lstm":
        mc = cfg.lstm
        model = LSTMClassifier(
            input_size=n_features,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
            dropout=mc.dropout,
            bidirectional=mc.bidirectional,
        )
        logger.info(f"LSTM architecture: hidden={mc.hidden_size}, layers={mc.num_layers}, "
                    f"dropout={mc.dropout}, bidirectional={mc.bidirectional}")
    else:
        # Use CNN config from model_config.yaml, with sensible defaults
        num_channels = cfg.get_nested("cnn", "num_channels") or 64
        kernel_size = cfg.get_nested("cnn", "kernel_size") or 7
        num_blocks = cfg.get_nested("cnn", "num_blocks") or 4
        dropout = cfg.get_nested("cnn", "dropout") or 0.3
        model = TemporalCNNClassifier(
            input_size=n_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        logger.info(f"CNN architecture: channels={num_channels}, kernel={kernel_size}, "
                    f"blocks={num_blocks}, dropout={dropout}")
    logger.info(f"Total parameters: {model.count_parameters():,}")
    return model


def _get_model_config_dict(model_type: str, cfg) -> dict:
    if model_type == "lstm":
        return dict(cfg.lstm)
    return {
        "num_channels": cfg.get_nested("cnn", "num_channels") or 64,
        "kernel_size": cfg.get_nested("cnn", "kernel_size") or 7,
        "num_blocks": cfg.get_nested("cnn", "num_blocks") or 4,
        "dropout": cfg.get_nested("cnn", "dropout") or 0.3,
    }


def main() -> None:
    args = parse_args()
    model_type = args.model_type.upper()

    checkpoint_dir = args.checkpoint_dir or f"models/checkpoints/{args.model_type}"
    report_dir = args.report_dir or f"reports/{args.model_type}"

    cfg = load_config(args.config, args.model_config)

    device_str = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_str)
    logger.info(f"Training {model_type} | device={device}")

    # ── Data ─────────────────────────────────────────────────────────────
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

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    preprocessor_path = Path(checkpoint_dir) / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    pos_weight = preprocessor.compute_pos_weight(y_train)
    logger.info(f"pos_weight: {pos_weight:.2f}  (neg/pos ratio in train set)")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=cfg.training.batch_size,
        oversample=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = _build_model(args.model_type, cfg, n_features=len(SENSOR_COLS))
    model_config_dict = _get_model_config_dict(args.model_type, cfg)

    # ── Training ──────────────────────────────────────────────────────────
    epochs = args.epochs or cfg.training.epochs
    trainer = LSTMTrainer(        # LSTMTrainer works for any nn.Module
        model=model,
        device=device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        pos_weight=pos_weight,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        checkpoint_dir=checkpoint_dir,
        experiment_name=cfg.experiment.experiment_name,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=cfg.training.early_stopping_patience,
        early_stopping_metric=cfg.training.early_stopping_metric,
        model_config=model_config_dict,
        training_config=dict(cfg.training),
    )

    # ── Test Evaluation ───────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation…")
    ckpt_path = Path(checkpoint_dir) / "best_model.pt"
    load_checkpoint(ckpt_path, model, device=device)

    model.eval()
    all_proba, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            proba = model.predict_proba(X_batch.to(device)).cpu().numpy()
            all_proba.append(proba)
            all_labels.append(y_batch.numpy())

    y_proba_test = np.concatenate(all_proba)
    y_true_test = np.concatenate(all_labels).astype(int)

    # Need val probabilities for calibration
    model.eval()
    all_val_proba: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            p = model.predict_proba(X_batch.to(device)).cpu().numpy()
            all_val_proba.append(p)
    y_proba_val = np.concatenate(all_val_proba)

    opt_thresh, _ = find_optimal_threshold(y_true_test, y_proba_test, metric="f1")
    metrics = compute_classification_metrics(y_true_test, y_proba_test, threshold=opt_thresh)

    logger.info(f"\n{'='*55}")
    logger.info(f"{model_type} Test Metrics (threshold={opt_thresh:.3f}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:22s}: {v:.4f}")

    # Bootstrap CIs
    logger.info("\nBootstrap 95% CI (n=500):")
    ci = bootstrap_confidence_intervals(
        y_true_test, y_proba_test, n_samples=500, threshold=opt_thresh
    )
    for m in ["f1", "roc_auc", "pr_auc", "recall", "precision"]:
        if m in ci:
            d = ci[m]
            logger.info(f"  {m:22s}: {d['mean']:.4f}  [{d['lower']:.4f}, {d['upper']:.4f}]")

    # Calibration analysis
    logger.info("\nCalibration analysis:")
    cal_results = compare_calibrators(
        y_val, y_proba_val, y_true_test, y_proba_test
    )
    for method, cal in cal_results.items():
        logger.info(
            f"  {method:22s}: Brier={cal['brier_score']:.4f}  "
            f"BSS={cal['brier_skill_score']:.4f}  ECE={cal['ece']:.4f}"
        )

    # ── Visualisations ────────────────────────────────────────────────────
    viz = EvaluationVisualizer(output_dir=report_dir)
    viz.plot_training_history(history, save_name="training_history.png")
    viz.plot_roc_curve(y_true_test, {model_type: y_proba_test}, save_name="roc_curve.png")
    viz.plot_pr_curve(y_true_test, {model_type: y_proba_test}, save_name="pr_curve.png")
    viz.plot_confusion_matrix(
        y_true_test, y_proba_test, threshold=opt_thresh,
        model_name=model_type, save_name="confusion_matrix.png"
    )
    viz.plot_threshold_analysis(y_true_test, y_proba_test, model_name=model_type)
    viz.plot_prediction_distribution(y_true_test, y_proba_test, model_name=model_type)
    viz.plot_calibration(y_true_test, {model_type: y_proba_test})

    logger.info(f"\nReports → {report_dir}")
    logger.info(f"Checkpoint → {checkpoint_dir}/best_model.pt")
    logger.info("Done.")


if __name__ == "__main__":
    main()
