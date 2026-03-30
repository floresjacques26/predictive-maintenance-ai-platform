"""Side-by-side comparison of all trained models on the held-out test set.

Loads every available checkpoint (LSTM, CNN, RF, LR) and evaluates all on
the identical test split.  Produces:
  - Metric comparison table (stdout)
  - Combined ROC, PR, calibration, comparison bar charts
  - Statistical significance tests (McNemar, DeLong AUC CI)
  - Cost-sensitive threshold analysis
  - JSON report

Usage
-----
python scripts/compare_models.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.calibration import compute_brier_score, compute_expected_calibration_error
from src.evaluation.cost_analysis import CostMatrix, cost_threshold_comparison
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.evaluation.significance_testing import compare_models_statistically
from src.evaluation.visualization import EvaluationVisualizer
from src.models.baseline import LogisticRegressionBaseline, RandomForestBaseline
from src.models.cnn_model import TemporalCNNClassifier
from src.models.lstm_model import LSTMClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("compare_models")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]
REPORT_DIR = Path("reports/comparison")


def _load_neural_predictions(
    checkpoint_path: Path,
    X_test: np.ndarray,
    model_cls,
) -> np.ndarray | None:
    """Load a neural checkpoint and run inference on X_test."""
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    if model_cls is LSTMClassifier:
        model = LSTMClassifier(
            input_size=len(SENSOR_COLS),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=0.0,
            bidirectional=cfg.get("bidirectional", False),
        )
    else:  # TemporalCNNClassifier
        model = TemporalCNNClassifier(
            input_size=len(SENSOR_COLS),
            num_channels=cfg.get("num_channels", 64),
            kernel_size=cfg.get("kernel_size", 7),
            num_blocks=cfg.get("num_blocks", 4),
            dropout=0.0,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        proba = model.predict_proba(
            torch.from_numpy(X_test.astype(np.float32))
        ).numpy()
    return proba


def _load_sklearn_predictions(path: Path, baseline_cls, X_test: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        logger.warning(f"Baseline checkpoint not found: {path}")
        return None
    model = baseline_cls.load(path)
    return model.predict_proba(X_test)


def main() -> None:
    cfg = load_config("configs/base_config.yaml")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/synthetic/sensor_data.csv")

    # ── Determine test split ───────────────────────────────────────────
    # Try LSTM checkpoint dir first, then CNN, then fall back to defaults
    for ckpt_dir in [
        Path("models/checkpoints/lstm"),
        Path("models/checkpoints/cnn"),
        Path("models/checkpoints"),
    ]:
        preprocessor_path = ckpt_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor = SensorDataPreprocessor.load(preprocessor_path)
            break
    else:
        logger.warning("No saved preprocessor found — creating new split from config.")
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
    y_true = y_test.astype(int)
    logger.info(f"Test set: {len(y_true):,} windows | "
                f"positive rate: {y_true.mean():.2%}")

    # ── Collect predictions from all available models ──────────────────
    predictions: dict[str, np.ndarray] = {}

    for name, ckpt_path, model_cls in [
        ("LSTM",  Path("models/checkpoints/lstm/best_model.pt"),  LSTMClassifier),
        ("LSTM",  Path("models/checkpoints/best_model.pt"),       LSTMClassifier),  # legacy path
        ("CNN1D", Path("models/checkpoints/cnn/best_model.pt"),   TemporalCNNClassifier),
    ]:
        if name in predictions:
            continue  # don't overwrite with legacy path
        proba = _load_neural_predictions(ckpt_path, X_test, model_cls)
        if proba is not None:
            predictions[name] = proba

    for name, path, cls in [
        ("RandomForest",       Path("models/baselines/randomforest.joblib"),      RandomForestBaseline),
        ("LogisticRegression", Path("models/baselines/logisticregression.joblib"), LogisticRegressionBaseline),
    ]:
        proba = _load_sklearn_predictions(path, cls, X_test)
        if proba is not None:
            predictions[name] = proba

    if not predictions:
        logger.error("No trained models found. Train at least one model first.")
        return

    logger.info(f"Models available for comparison: {list(predictions.keys())}")

    # ── Compute metrics (optimal threshold per model) ──────────────────
    all_metrics: dict[str, dict] = {}
    for name, proba in predictions.items():
        thresh, _ = find_optimal_threshold(y_true, proba, metric="f1")
        m = compute_classification_metrics(y_true, proba, threshold=thresh)
        m["brier_score"] = compute_brier_score(y_true, proba)
        m["ece"] = compute_expected_calibration_error(y_true, proba)
        all_metrics[name] = m

    # ── Statistical significance tests ────────────────────────────────
    if len(predictions) >= 2:
        logger.info("\nStatistical significance tests (McNemar + DeLong AUC):")
        sig_results = compare_models_statistically(
            y_true, predictions, threshold=0.5, n_permutations=2000
        )
        for pair, results in sig_results.items():
            mc = results["mcnemar"]
            auc = results["auc_permutation"]
            ci_a = results["delong_ci_a"]
            ci_b = results["delong_ci_b"]
            name_a, name_b = pair.split("_vs_")
            logger.info(f"\n  {name_a} vs {name_b}:")
            logger.info(f"    McNemar: p={mc['p_value']:.4f}  ({mc['interpretation']})")
            logger.info(f"    AUC permutation: p={auc['p_value']:.4f}  ({auc['interpretation']})")
            logger.info(f"    DeLong AUC CI: {name_a}=[{ci_a['lower']:.3f},{ci_a['upper']:.3f}]  "
                        f"{name_b}=[{ci_b['lower']:.3f},{ci_b['upper']:.3f}]")
    else:
        sig_results = {}

    # ── Cost-sensitive analysis ────────────────────────────────────────
    cost_matrix = CostMatrix(cost_fn=50_000, cost_fp=5_000)
    logger.info(f"\nCost analysis (FN={cost_matrix.cost_fn:,}, FP={cost_matrix.cost_fp:,}):")
    cost_results: dict[str, dict] = {}
    for name, proba in predictions.items():
        thresh, _ = find_optimal_threshold(y_true, proba, metric="f1")
        comparison = cost_threshold_comparison(y_true, proba, cost_matrix, thresh)
        cost_results[name] = comparison
        saving = comparison["cost_saving_vs_f1"]
        logger.info(
            f"  {name:22s}: cost-optimal threshold={comparison['cost_optimal_decision']['threshold']:.3f}  "
            f"saving vs F1-threshold={saving:+,.0f}"
        )

    # ── Visualisations ─────────────────────────────────────────────────
    viz = EvaluationVisualizer(output_dir=str(REPORT_DIR))
    viz.plot_roc_curve(y_true, predictions, save_name="all_models_roc.png")
    viz.plot_pr_curve(y_true, predictions, save_name="all_models_pr.png")
    viz.plot_model_comparison(all_metrics, save_name="all_models_comparison.png")
    viz.plot_calibration(y_true, predictions, save_name="all_models_calibration.png")

    for name, proba in predictions.items():
        thresh = all_metrics[name]["threshold"]
        viz.plot_confusion_matrix(
            y_true, proba, threshold=thresh,
            model_name=name, save_name=f"{name.lower()}_cm.png"
        )
        viz.plot_prediction_distribution(
            y_true, proba, model_name=name,
            save_name=f"{name.lower()}_dist.png"
        )
        viz.plot_threshold_analysis(
            y_true, proba, model_name=name,
            save_name=f"{name.lower()}_threshold.png"
        )

    # ── Ranking table ──────────────────────────────────────────────────
    cols = ["f1", "roc_auc", "pr_auc", "recall", "precision", "mcc", "brier_score", "ece"]
    logger.info("\n" + "=" * 85)
    header = f"{'Model':<22}" + "".join(f"{c:>10}" for c in cols)
    logger.info(header)
    logger.info("-" * 85)
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1].get("f1", 0)):
        row = f"{name:<22}" + "".join(f"{m.get(c, 0.0):>10.4f}" for c in cols)
        logger.info(row)
    logger.info("=" * 85)

    # ── Persist report ─────────────────────────────────────────────────
    report = {
        "models": list(predictions.keys()),
        "test_metrics": all_metrics,
        "significance_tests": sig_results,
        "cost_analysis": cost_results,
    }
    out = REPORT_DIR / "model_comparison.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nFull comparison report → {out}")


if __name__ == "__main__":
    main()
