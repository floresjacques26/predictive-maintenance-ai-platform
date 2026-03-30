"""Feature importance and interpretability analysis for all trained models.

Outputs
-------
reports/interpretability/
  ├── interpretability_report.txt          — human-readable summary
  ├── interpretability_results.json        — structured results
  ├── <model>_feature_importance.png       — temporal × sensor heatmap
  ├── <model>_sensor_importance_bar.png    — per-sensor bar chart (permutation)
  └── <model>_temporal_importance.png      — temporal bin importance

Usage
-----
python scripts/run_interpretability.py
python scripts/run_interpretability.py --skip-neural   # only baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import SensorDataPreprocessor
from src.evaluation.interpretability import compute_all_importances
from src.evaluation.metrics import find_optimal_threshold
from src.evaluation.visualization import EvaluationVisualizer
from src.models.baseline import LogisticRegressionBaseline, RandomForestBaseline
from src.models.cnn_model import TemporalCNNClassifier
from src.models.lstm_model import LSTMClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_interpretability")

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]
REPORT_DIR = Path("reports/interpretability")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/synthetic/sensor_data.csv")
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--skip-neural", action="store_true",
                   help="Skip LSTM/CNN (faster).")
    p.add_argument("--device", default=None)
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


def _plot_sensor_importance_bar(
    permutation: dict[str, dict],
    model_name: str,
    output_dir: Path,
) -> None:
    """Bar chart of sensor permutation importance with error bars."""
    sensors = list(permutation.keys())
    importances = [permutation[s]["mean_importance"] for s in sensors]
    stds = [permutation[s]["std_importance"] for s in sensors]

    # Sort descending
    order = np.argsort(importances)[::-1]
    sensors = [sensors[i] for i in order]
    importances = [importances[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["crimson" if v > 0 else "steelblue" for v in importances]
    ax.barh(sensors, importances, xerr=stds, color=colors, alpha=0.8,
            error_kw={"elinewidth": 1.5, "capsize": 3})
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("F1 drop when sensor is permuted")
    ax.set_title(f"{model_name} — Sensor Permutation Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    path = output_dir / f"{model_name.lower().replace(' ', '_')}_sensor_importance_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {path}")


def _plot_temporal_importance(
    temporal: dict[str, list],
    model_name: str,
    output_dir: Path,
) -> None:
    """Bar chart of temporal position importance."""
    bins = temporal["bin_labels"]
    importances = temporal["importances"]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["crimson" if v > 0 else "steelblue" for v in importances]
    ax.bar(range(len(bins)), importances, color=colors, alpha=0.8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Window position (0 = oldest timestep)")
    ax.set_ylabel("F1 drop when permuted")
    ax.set_title(
        f"{model_name} — Temporal Position Importance\n"
        f"(baseline F1={temporal['baseline_f1']:.4f})"
    )
    fig.tight_layout()
    path = output_dir / f"{model_name.lower().replace(' ', '_')}_temporal_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {path}")


def _format_text_report(results: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("INTERPRETABILITY REPORT — Predictive Maintenance Models")
    lines.append("=" * 70)

    for model_name, data in results.items():
        lines.append(f"\n{'─'*60}")
        lines.append(f"Model: {model_name}")
        lines.append(f"{'─'*60}")

        # Sensor permutation importance
        perm = data.get("permutation", {})
        if perm:
            lines.append("\nSensor Permutation Importance (F1 drop when permuted):")
            lines.append(f"  Baseline F1: {list(perm.values())[0]['baseline_f1']:.4f}")
            sorted_sensors = sorted(perm.items(), key=lambda x: -x[1]["mean_importance"])
            for sensor, vals in sorted_sensors:
                lines.append(
                    f"  {sensor:<15}: {vals['mean_importance']:+.4f} ± {vals['std_importance']:.4f}  "
                    f"(permuted F1={vals['mean_permuted_f1']:.4f})"
                )

        # Temporal importance summary
        temporal = data.get("temporal_permutation", {})
        if temporal:
            imps = temporal["importances"]
            bins = temporal["bin_labels"]
            max_idx = int(np.argmax(imps))
            lines.append(f"\nTemporal Importance: most important bin = {bins[max_idx]} "
                         f"(importance={imps[max_idx]:+.4f})")

        # Intrinsic/saliency summary
        if "intrinsic_matrix" in data:
            mat = data["intrinsic_matrix"]
            sensor_sums = mat.sum(axis=0) if isinstance(mat, np.ndarray) else []
            if len(sensor_sums) > 0:
                top_sensor_idx = int(np.argmax(sensor_sums))
                lines.append(
                    f"Intrinsic top sensor: sensor index {top_sensor_idx}"
                )
        if "saliency_matrix" in data:
            mat = data["saliency_matrix"]
            if isinstance(mat, np.ndarray) and mat.size > 0:
                sensor_sums = mat.mean(axis=0)
                top_idx = int(np.argmax(sensor_sums))
                lines.append(f"Gradient saliency top sensor: index {top_idx}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    cfg = load_config(args.config)

    device_str = args.device or "cpu"
    device = torch.device(device_str)

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
        logger.error("No preprocessor found. Train at least one model first.")
        sys.exit(1)

    (X_train, y_train), (_, _), (X_test, y_test) = preprocessor.fit_transform(df)
    y_test_int = y_test.astype(int)

    # Subsample test set for speed (gradient saliency on 500 samples is fine)
    rng = np.random.default_rng(42)
    n_sample = min(len(X_test), 500)
    idx = rng.choice(len(X_test), size=n_sample, replace=False)
    X_sample = X_test[idx]
    y_sample = y_test_int[idx]

    all_results: dict[str, dict] = {}
    viz = EvaluationVisualizer(output_dir=str(REPORT_DIR))

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_path = Path("models/baselines/randomforest.joblib")
    if rf_path.exists():
        logger.info("\n── Random Forest ──")
        rf = RandomForestBaseline.load(rf_path)
        thresh, _ = find_optimal_threshold(y_test_int, rf.predict_proba(X_test))
        result = compute_all_importances(
            rf, X_sample, y_sample, SENSOR_COLS,
            threshold=thresh, model_type="rf"
        )
        all_results["RandomForest"] = result
        if "intrinsic_matrix" in result:
            viz.plot_feature_importance(
                result["intrinsic_matrix"], SENSOR_COLS,
                model_name="RandomForest (intrinsic)",
                save_name="randomforest_feature_importance.png",
            )
        _plot_sensor_importance_bar(result["permutation"], "RandomForest", REPORT_DIR)
        _plot_temporal_importance(result["temporal_permutation"], "RandomForest", REPORT_DIR)

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_path = Path("models/baselines/logisticregression.joblib")
    if lr_path.exists():
        logger.info("\n── Logistic Regression ──")
        lr = LogisticRegressionBaseline.load(lr_path)
        thresh, _ = find_optimal_threshold(y_test_int, lr.predict_proba(X_test))
        result = compute_all_importances(
            lr, X_sample, y_sample, SENSOR_COLS,
            threshold=thresh, model_type="lr"
        )
        all_results["LogisticRegression"] = result
        if "intrinsic_matrix" in result:
            viz.plot_feature_importance(
                result["intrinsic_matrix"], SENSOR_COLS,
                model_name="LogisticRegression (coefficients)",
                save_name="logisticregression_feature_importance.png",
            )
        _plot_sensor_importance_bar(result["permutation"], "LogisticRegression", REPORT_DIR)
        _plot_temporal_importance(result["temporal_permutation"], "LogisticRegression", REPORT_DIR)

    if args.skip_neural:
        logger.info("--skip-neural set — skipping LSTM/CNN interpretability.")
    else:
        # ── LSTM ─────────────────────────────────────────────────────────────
        for lstm_path in [
            Path("models/checkpoints/lstm/best_model.pt"),
            Path("models/checkpoints/best_model.pt"),
        ]:
            if lstm_path.exists():
                logger.info("\n── LSTM ──")
                model = _load_neural(lstm_path, LSTMClassifier)
                assert model is not None
                thresh, _ = find_optimal_threshold(
                    y_test_int,
                    model.predict_proba(
                        torch.from_numpy(X_test.astype(np.float32))
                    ).numpy()
                )
                result = compute_all_importances(
                    model, X_sample, y_sample, SENSOR_COLS,
                    threshold=thresh, model_type="lstm", device=device
                )
                all_results["LSTM"] = result
                if "saliency_matrix" in result:
                    viz.plot_feature_importance(
                        result["saliency_matrix"], SENSOR_COLS,
                        model_name="LSTM (gradient saliency)",
                        save_name="lstm_feature_importance.png",
                    )
                _plot_sensor_importance_bar(result["permutation"], "LSTM", REPORT_DIR)
                _plot_temporal_importance(result["temporal_permutation"], "LSTM", REPORT_DIR)
                break

        # ── CNN ───────────────────────────────────────────────────────────────
        cnn_path = Path("models/checkpoints/cnn/best_model.pt")
        if cnn_path.exists():
            logger.info("\n── CNN1D ──")
            model = _load_neural(cnn_path, TemporalCNNClassifier)
            assert model is not None
            thresh, _ = find_optimal_threshold(
                y_test_int,
                model.predict_proba(
                    torch.from_numpy(X_test.astype(np.float32))
                ).numpy()
            )
            result = compute_all_importances(
                model, X_sample, y_sample, SENSOR_COLS,
                threshold=thresh, model_type="cnn", device=device
            )
            all_results["CNN1D"] = result
            if "saliency_matrix" in result:
                viz.plot_feature_importance(
                    result["saliency_matrix"], SENSOR_COLS,
                    model_name="CNN1D (gradient saliency)",
                    save_name="cnn_feature_importance.png",
                )
            _plot_sensor_importance_bar(result["permutation"], "CNN1D", REPORT_DIR)
            _plot_temporal_importance(result["temporal_permutation"], "CNN1D", REPORT_DIR)

    if not all_results:
        logger.error("No models could be analysed.")
        sys.exit(1)

    # ── Reports ───────────────────────────────────────────────────────────────
    text_report = _format_text_report(all_results)
    logger.info("\n" + text_report)

    (REPORT_DIR / "interpretability_report.txt").write_text(text_report, encoding="utf-8")

    # JSON — convert numpy arrays to lists for serialisation
    def _np_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _np_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_np_to_list(v) for v in obj]
        return obj

    with open(REPORT_DIR / "interpretability_results.json", "w") as f:
        json.dump(_np_to_list(all_results), f, indent=2, default=str)

    logger.info(f"\nInterpretability analysis complete → {REPORT_DIR}/")


if __name__ == "__main__":
    main()
