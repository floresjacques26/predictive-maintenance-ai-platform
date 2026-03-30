"""Run Optuna hyperparameter search for the LSTM model.

Usage
-----
python scripts/run_hyperparameter_search.py --n-trials 50 --timeout 3600
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import SensorDataPreprocessor
from src.experiments.hyperparameter_search import HyperparameterSearch
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("hpo")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM hyperparameter search.")
    parser.add_argument("--data-path", type=str, default="data/synthetic/sensor_data.csv")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--epochs-per-trial", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="models/hpo_checkpoints")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"HPO device: {device}")

    df = pd.read_csv(args.data_path)
    preprocessor = SensorDataPreprocessor(
        sensor_columns=SENSOR_COLS,
        target_column="failure_imminent",
        window_size=cfg.data.window_size,
        step_size=cfg.data.step_size,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_seed=cfg.data.random_seed,
    )
    (X_train, y_train), (X_val, y_val), _ = preprocessor.fit_transform(df)

    pos_weight = preprocessor.compute_pos_weight(y_train)

    search = HyperparameterSearch(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_input_features=len(SENSOR_COLS),
        pos_weight=pos_weight,
        device=device,
        checkpoint_dir=args.output_dir,
    )

    best_params = search.run(
        n_trials=args.n_trials,
        timeout=args.timeout,
        epochs_per_trial=args.epochs_per_trial,
    )

    logger.info("\nParameter importance:")
    importance = search.get_importance_summary()
    for param, score in sorted(importance.items(), key=lambda x: -x[1]):
        logger.info(f"  {param:25s}: {score:.4f}")

    logger.info(f"\nBest params: {best_params}")


if __name__ == "__main__":
    main()
