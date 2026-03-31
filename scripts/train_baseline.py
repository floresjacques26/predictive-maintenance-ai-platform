"""Train Random Forest and Logistic Regression baselines.

Supports both the synthetic dataset and NASA CMAPSS.

Usage
-----
python scripts/train_baseline.py                              # synthetic
python scripts/train_baseline.py --dataset cmapss            # CMAPSS FD001
python scripts/train_baseline.py --dataset cmapss --cmapss-subset FD002
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_factory import load_dataset
from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.evaluation.visualization import EvaluationVisualizer
from src.models.baseline import LogisticRegressionBaseline, RandomForestBaseline
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sklearn baseline models.")
    parser.add_argument(
        "--dataset", choices=["synthetic", "cmapss"], default="synthetic",
    )
    parser.add_argument(
        "--cmapss-subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001",
    )
    parser.add_argument("--data-path", type=str, default="data/synthetic/sensor_data.csv")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_paths = [args.config]
    if args.dataset == "cmapss":
        config_paths.append("configs/cmapss_config.yaml")
    cfg = load_config(*config_paths)

    if args.dataset == "cmapss":
        cfg.setdefault("dataset", {}).setdefault("cmapss", {})["subset"] = args.cmapss_subset
        subset = args.cmapss_subset
        output_dir = args.output_dir or f"models/baselines/cmapss/{subset}"
        report_dir = args.report_dir or f"reports/baselines/cmapss/{subset}"
    else:
        output_dir = args.output_dir or "models/baselines"
        report_dir = args.report_dir or "reports/baselines"

    df, sensor_cols = load_dataset(
        dataset_type=args.dataset,
        cfg=cfg,
        data_path=args.data_path,
    )

    preprocessor = SensorDataPreprocessor(
        sensor_columns=sensor_cols,
        target_column="failure_imminent",
        window_size=cfg.data.window_size,
        step_size=cfg.data.step_size,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        scaler_type=cfg.features.scaler_type,
        random_seed=cfg.data.random_seed,
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.fit_transform(df)

    mlflow.set_experiment(cfg.experiment.experiment_name)
    visualizer = EvaluationVisualizer(output_dir=report_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    models = {
        "RandomForest": RandomForestBaseline(),
        "LogisticRegression": LogisticRegressionBaseline(),
    }

    all_metrics: dict[str, dict] = {}

    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {name}…")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            # Evaluate on test set
            y_proba_test = model.predict_proba(X_test)
            opt_thresh, _ = find_optimal_threshold(y_test, y_proba_test, metric="f1")
            metrics = compute_classification_metrics(y_test, y_proba_test, threshold=opt_thresh)

            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_param("optimal_threshold", opt_thresh)

            all_metrics[name] = metrics
            logger.info(f"{name} Test metrics @ threshold={opt_thresh:.3f}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")

            # Save model
            out = Path(output_dir) / f"{name.lower()}.joblib"
            model.save(out)
            mlflow.log_artifact(str(out))

            # Plots
            visualizer.plot_confusion_matrix(
                y_test, y_proba_test, threshold=opt_thresh,
                model_name=name, save_name=f"{name.lower()}_cm.png"
            )
            visualizer.plot_threshold_analysis(
                y_test, y_proba_test, model_name=name,
                save_name=f"{name.lower()}_threshold.png"
            )
            visualizer.plot_prediction_distribution(
                y_test, y_proba_test, model_name=name,
                save_name=f"{name.lower()}_pred_dist.png"
            )

    # Combined ROC & PR
    test_probas = {
        name: models[name].predict_proba(X_test)
        for name in models
    }
    visualizer.plot_roc_curve(y_test, test_probas, save_name="baselines_roc.png")
    visualizer.plot_pr_curve(y_test, test_probas, save_name="baselines_pr.png")
    visualizer.plot_model_comparison(all_metrics, save_name="baselines_comparison.png")

    # Save results summary
    summary_path = Path(report_dir) / "baseline_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Baseline metrics saved → {summary_path}")


if __name__ == "__main__":
    main()
