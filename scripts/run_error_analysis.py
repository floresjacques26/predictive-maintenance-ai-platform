"""Detailed false-positive / false-negative error analysis.

Requires trained models and preprocessor.  Run after training all models.

Outputs
-------
reports/error_analysis/
  ├── error_report.txt                    — human-readable summary
  ├── error_report.json                   — structured results
  ├── <model>_error_by_machine_type.png
  ├── <model>_error_by_degradation_stage.png
  └── <model>_error_proximity.png

Usage
-----
python scripts/run_error_analysis.py
python scripts/run_error_analysis.py --threshold-metric f1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.error_analysis import format_error_report, full_error_report
from src.evaluation.metrics import find_optimal_threshold
from src.evaluation.visualization import EvaluationVisualizer
from src.models.baseline import LogisticRegressionBaseline, RandomForestBaseline
from src.models.cnn_model import TemporalCNNClassifier
from src.models.lstm_model import LSTMClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_error_analysis")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]
REPORT_DIR = Path("reports/error_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/synthetic/sensor_data.csv")
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--threshold-metric", default="f1",
                   choices=["f1", "j_stat"], help="Metric for optimal threshold selection.")
    return p.parse_args()


def _load_neural(ckpt_path: Path, model_cls) -> torch.nn.Module | None:
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    if model_cls is LSTMClassifier:
        model = LSTMClassifier(
            input_size=len(SENSOR_COLS),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=0.0,
            bidirectional=cfg.get("bidirectional", False),
        )
    else:
        model = TemporalCNNClassifier(
            input_size=len(SENSOR_COLS),
            num_channels=cfg.get("num_channels", 64),
            kernel_size=cfg.get("kernel_size", 7),
            num_blocks=cfg.get("num_blocks", 4),
            dropout=0.0,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data: {args.data_path}")
    df = pd.read_csv(args.data_path)
    cfg = load_config(args.config)

    # ── Preprocessor ─────────────────────────────────────────────────────────
    preprocessor: SensorDataPreprocessor | None = None
    for ckpt_dir in [
        Path("models/checkpoints/lstm"),
        Path("models/checkpoints/cnn"),
        Path("models/checkpoints"),
    ]:
        pp_path = ckpt_dir / "preprocessor.joblib"
        if pp_path.exists():
            preprocessor = SensorDataPreprocessor.load(pp_path)
            break

    if preprocessor is None:
        logger.error("No saved preprocessor found. Train at least one model first.")
        sys.exit(1)

    (_, _), (_, _), (X_test, y_test) = preprocessor.fit_transform(df)
    y_true = y_test.astype(int)
    test_ids = preprocessor.test_machine_ids

    logger.info(
        f"Test split: {len(y_true):,} windows | "
        f"{len(test_ids)} machines | positive rate: {y_true.mean():.2%}"
    )

    # ── Collect predictions ───────────────────────────────────────────────────
    predictions: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for name, path, cls in [
        ("LSTM",  Path("models/checkpoints/lstm/best_model.pt"), LSTMClassifier),
        ("LSTM",  Path("models/checkpoints/best_model.pt"),       LSTMClassifier),
        ("CNN1D", Path("models/checkpoints/cnn/best_model.pt"),   TemporalCNNClassifier),
    ]:
        if name not in predictions:
            model = _load_neural(path, cls)
            if model is not None:
                with torch.no_grad():
                    proba = model.predict_proba(
                        torch.from_numpy(X_test.astype(np.float32))
                    ).numpy()
                predictions[name] = proba

    for name, path, cls in [
        ("RandomForest",       Path("models/baselines/randomforest.joblib"),      RandomForestBaseline),
        ("LogisticRegression", Path("models/baselines/logisticregression.joblib"), LogisticRegressionBaseline),
    ]:
        if path.exists():
            m = cls.load(path)
            predictions[name] = m.predict_proba(X_test)

    if not predictions:
        logger.error("No trained models found.")
        sys.exit(1)

    logger.info(f"Models loaded: {list(predictions.keys())}")

    for name, proba in predictions.items():
        thresh, _ = find_optimal_threshold(y_true, proba, metric=args.threshold_metric)
        thresholds[name] = thresh
        logger.info(f"  {name}: threshold={thresh:.3f}")

    # ── Error analysis ────────────────────────────────────────────────────────
    report = full_error_report(
        df=df,
        test_machine_ids=test_ids,
        window_size=preprocessor.window_size,
        step_size=preprocessor.step_size,
        predictions=predictions,
        y_true=y_true,
        thresholds=thresholds,
    )

    if not report:
        logger.error("Error analysis failed (metadata mismatch). See warnings above.")
        sys.exit(1)

    # ── Text report ───────────────────────────────────────────────────────────
    text_report = format_error_report(report)
    logger.info("\n" + text_report)

    report_path = REPORT_DIR / "error_report.txt"
    report_path.write_text(text_report, encoding="utf-8")

    json_path = REPORT_DIR / "error_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Visualisations ────────────────────────────────────────────────────────
    viz = EvaluationVisualizer(output_dir=str(REPORT_DIR))

    for name, data in report.items():
        safe = name.lower().replace(" ", "_")

        viz.plot_error_breakdown(
            data["by_machine_type"],
            group_by="machine_type",
            model_name=name,
            save_name=f"{safe}_error_by_machine_type.png",
        )
        viz.plot_error_breakdown(
            data["by_degradation_stage"],
            group_by="degradation_stage",
            model_name=name,
            save_name=f"{safe}_error_by_degradation_stage.png",
        )
        viz.plot_proximity_error(
            data["by_proximity_to_failure"],
            model_name=name,
            save_name=f"{safe}_error_proximity.png",
        )

    logger.info(f"\nError analysis complete.")
    logger.info(f"  Text report → {report_path}")
    logger.info(f"  JSON report → {json_path}")
    logger.info(f"  Figures     → {REPORT_DIR}/")


if __name__ == "__main__":
    main()
