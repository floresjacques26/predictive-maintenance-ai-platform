"""Ablation study: systematically vary architecture and data choices.

Studies conducted
-----------------
1. **Window size**: T ∈ {10, 25, 50, 100} — does temporal context matter?
2. **pos_weight**: with vs without class imbalance correction
3. **Model capacity**: hidden_size ∈ {32, 64, 128, 256}
4. **LSTM vs CNN**: same conditions, different architecture

All experiments share: same data split, same random seed, same epochs.
Results are logged to MLflow under experiment 'predictive_maintenance_ablation'.

Usage
-----
python scripts/ablation_study.py --data-path data/synthetic/sensor_data.csv
python scripts/ablation_study.py --study window_size --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import create_dataloaders
from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.experiments.trainer import LSTMTrainer
from src.models.cnn_model import TemporalCNNClassifier
from src.models.lstm_model import LSTMClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("ablation")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]
EXPERIMENT_NAME = "predictive_maintenance_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation studies.")
    parser.add_argument("--study", choices=["window_size", "pos_weight", "capacity", "lstm_vs_cnn", "all"],
                        default="all")
    parser.add_argument("--data-path", type=str, default="data/synthetic/sensor_data.csv")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/ablation/results.json")
    return parser.parse_args()


def _get_device(override: str | None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _train_and_eval(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pos_weight: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: str,
    run_name: str,
) -> dict[str, float]:
    """Train and evaluate a single model, returning test metrics."""
    train_dl, val_dl, test_dl = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, oversample=False,
    )

    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        pos_weight=pos_weight,
        checkpoint_dir=f"models/ablation/{run_name}",
        experiment_name=EXPERIMENT_NAME,
    )

    with mlflow.start_run(run_name=run_name):
        history = trainer.fit(
            train_dl, val_dl,
            epochs=epochs, patience=patience,
        )

    # Evaluate on test set with best checkpoint
    ckpt_path = Path(f"models/ablation/{run_name}") / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    all_proba: list[np.ndarray] = []
    X_t = torch.from_numpy(X_test.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(X_t), 64):
            batch = X_t[i : i + 64].to(torch.device(device))
            all_proba.append(model.predict_proba(batch).cpu().numpy())

    y_proba = np.concatenate(all_proba)
    thresh, _ = find_optimal_threshold(y_test, y_proba, metric="f1")
    metrics = compute_classification_metrics(y_test, y_proba, threshold=thresh)
    metrics["best_val_f1"] = max(history["val_f1"]) if history["val_f1"] else 0.0
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    metrics["run_name"] = run_name
    logger.info(f"  {run_name}: F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    return metrics


def study_window_size(df: pd.DataFrame, cfg, device: str, epochs: int, patience: int) -> dict:
    logger.info("\n=== STUDY: Window Size ===")
    window_sizes = [10, 25, 50, 100]
    results = {}

    for ws in window_sizes:
        run_name = f"window_{ws}"
        prep = SensorDataPreprocessor(
            sensor_columns=SENSOR_COLS,
            target_column="failure_imminent",
            window_size=ws,
            step_size=max(1, ws // 5),
            test_size=cfg.data.test_size,
            val_size=cfg.data.val_size,
            random_seed=cfg.data.random_seed,
        )
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = prep.fit_transform(df)
        pos_weight = prep.compute_pos_weight(y_tr)

        model = LSTMClassifier(
            input_size=len(SENSOR_COLS),
            hidden_size=128, num_layers=2, dropout=0.3,
        )
        metrics = _train_and_eval(
            model, X_tr, y_tr, X_v, y_v, X_te, y_te,
            pos_weight=pos_weight, batch_size=64,
            epochs=epochs, patience=patience, device=device,
            run_name=run_name,
        )
        results[run_name] = {"window_size": ws, **metrics}

    return results


def study_pos_weight(df: pd.DataFrame, cfg, device: str, epochs: int, patience: int) -> dict:
    logger.info("\n=== STUDY: pos_weight (class imbalance handling) ===")
    prep = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS, target_column="failure_imminent",
        window_size=cfg.data.window_size, step_size=cfg.data.step_size,
        test_size=cfg.data.test_size, val_size=cfg.data.val_size,
        random_seed=cfg.data.random_seed,
    )
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = prep.fit_transform(df)
    pos_weight = prep.compute_pos_weight(y_tr)

    results = {}
    for name, pw in [("no_pos_weight", 1.0), ("with_pos_weight", pos_weight)]:
        model = LSTMClassifier(
            input_size=len(SENSOR_COLS), hidden_size=128, num_layers=2, dropout=0.3,
        )
        metrics = _train_and_eval(
            model, X_tr, y_tr, X_v, y_v, X_te, y_te,
            pos_weight=pw, batch_size=64,
            epochs=epochs, patience=patience, device=device,
            run_name=name,
        )
        results[name] = {"pos_weight": pw, **metrics}

    return results


def study_capacity(df: pd.DataFrame, cfg, device: str, epochs: int, patience: int) -> dict:
    logger.info("\n=== STUDY: Model Capacity (hidden_size) ===")
    prep = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS, target_column="failure_imminent",
        window_size=cfg.data.window_size, step_size=cfg.data.step_size,
        test_size=cfg.data.test_size, val_size=cfg.data.val_size,
        random_seed=cfg.data.random_seed,
    )
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = prep.fit_transform(df)
    pos_weight = prep.compute_pos_weight(y_tr)

    results = {}
    for hs in [32, 64, 128, 256]:
        run_name = f"hidden_{hs}"
        model = LSTMClassifier(
            input_size=len(SENSOR_COLS), hidden_size=hs, num_layers=2, dropout=0.3,
        )
        metrics = _train_and_eval(
            model, X_tr, y_tr, X_v, y_v, X_te, y_te,
            pos_weight=pos_weight, batch_size=64,
            epochs=epochs, patience=patience, device=device,
            run_name=run_name,
        )
        results[run_name] = {"hidden_size": hs, **metrics}

    return results


def study_lstm_vs_cnn(df: pd.DataFrame, cfg, device: str, epochs: int, patience: int) -> dict:
    logger.info("\n=== STUDY: LSTM vs CNN ===")
    prep = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS, target_column="failure_imminent",
        window_size=cfg.data.window_size, step_size=cfg.data.step_size,
        test_size=cfg.data.test_size, val_size=cfg.data.val_size,
        random_seed=cfg.data.random_seed,
    )
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = prep.fit_transform(df)
    pos_weight = prep.compute_pos_weight(y_tr)

    lstm = LSTMClassifier(
        input_size=len(SENSOR_COLS), hidden_size=128, num_layers=2, dropout=0.3,
    )
    cnn = TemporalCNNClassifier(
        input_size=len(SENSOR_COLS), num_channels=64, kernel_size=7,
        num_blocks=4, dropout=0.3,
    )

    results = {}
    for name, model in [("LSTM", lstm), ("CNN1D", cnn)]:
        metrics = _train_and_eval(
            model, X_tr, y_tr, X_v, y_v, X_te, y_te,
            pos_weight=pos_weight, batch_size=64,
            epochs=epochs, patience=patience, device=device,
            run_name=name,
        )
        results[name] = metrics

    return results


def main() -> None:
    args = parse_args()
    device = _get_device(args.device)
    logger.info(f"Ablation study | device={device} | epochs={args.epochs}")

    cfg = load_config(args.config)
    df = pd.read_csv(args.data_path)
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results: dict[str, dict] = {}

    studies: dict[str, Callable] = {
        "window_size": study_window_size,
        "pos_weight": study_pos_weight,
        "capacity": study_capacity,
        "lstm_vs_cnn": study_lstm_vs_cnn,
    }

    to_run = list(studies.keys()) if args.study == "all" else [args.study]

    for study_name in to_run:
        fn = studies[study_name]
        results = fn(df, cfg, device, args.epochs, args.patience)
        all_results[study_name] = results

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    for study_name, results in all_results.items():
        logger.info(f"\n{study_name}:")
        for variant, metrics in results.items():
            f1 = metrics.get("f1", 0)
            auc = metrics.get("roc_auc", 0)
            pr = metrics.get("pr_auc", 0)
            logger.info(f"  {variant:30s}: F1={f1:.4f} ROC-AUC={auc:.4f} PR-AUC={pr:.4f}")

    # Persist
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAblation results saved → {out_path}")


if __name__ == "__main__":
    main()
