"""Master pipeline: generate → train all models → benchmark → report.

Produces a complete reproducible benchmark of all four model families
(LSTM, CNN, Random Forest, Logistic Regression) on the held-out test set.

Outputs
-------
reports/benchmark/             (synthetic)
reports/benchmark/cmapss/FD001 (CMAPSS)
  ├── metrics_table.txt          — human-readable comparison table
  ├── benchmark_results.json     — full structured report
  ├── roc_all_models.png
  ├── pr_all_models.png
  ├── calibration_all_models.png
  ├── comparison_bar.png
  ├── cost_curve_all_models.png
  ├── <model>_confusion_matrix.png
  ├── <model>_threshold_analysis.png
  └── <model>_prediction_dist.png

Usage
-----
python scripts/run_full_benchmark.py                          # synthetic, train + benchmark
python scripts/run_full_benchmark.py --skip-training          # benchmark only
python scripts/run_full_benchmark.py --dataset cmapss         # NASA CMAPSS FD001
python scripts/run_full_benchmark.py --dataset cmapss --cmapss-subset FD002
python scripts/run_full_benchmark.py --n-machines 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_factory import load_dataset
from src.data.preprocessing import SensorDataPreprocessor
from src.data.synthetic_generator import SyntheticSensorDataGenerator
from src.evaluation.calibration import compute_brier_score, compute_expected_calibration_error
from src.evaluation.cost_analysis import CostMatrix, cost_threshold_comparison, find_cost_optimal_threshold
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.evaluation.significance_testing import compare_models_statistically
from src.evaluation.statistical_analysis import bootstrap_confidence_intervals
from src.evaluation.visualization import EvaluationVisualizer
from src.models.baseline import LogisticRegressionBaseline, RandomForestBaseline
from src.models.cnn_model import TemporalCNNClassifier
from src.models.lstm_model import LSTMClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_full_benchmark")

DATA_PATH = Path("data/synthetic/sensor_data.csv")
COST_FN = 50_000   # Missed failure: unplanned downtime + emergency repair
COST_FP = 5_000    # False alarm: unnecessary planned maintenance stop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full benchmark pipeline.")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training and use existing checkpoints.")
    p.add_argument("--n-machines", type=int, default=200,
                   help="Number of simulated machines to generate (synthetic only).")
    p.add_argument("--epochs", type=int, default=50,
                   help="Max training epochs for neural models.")
    p.add_argument("--config", type=str, default="configs/base_config.yaml")
    p.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    p.add_argument(
        "--dataset", choices=["synthetic", "cmapss"], default="synthetic",
        help="Dataset to benchmark.",
    )
    p.add_argument(
        "--cmapss-subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001",
        help="CMAPSS sub-dataset (only used when --dataset cmapss).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Data generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_data(n_machines: int) -> None:
    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        n_existing = existing["machine_id"].nunique()
        if n_existing >= n_machines:
            logger.info(f"Synthetic data already present ({n_existing} machines) — skipping generation.")
            return
    logger.info(f"Generating synthetic dataset with {n_machines} machines…")
    gen = SyntheticSensorDataGenerator(n_machines=n_machines, failure_horizon=30, random_seed=42)
    gen.generate(output_dir=DATA_PATH.parent)
    logger.info(f"Dataset saved → {DATA_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Model training
# ──────────────────────────────────────────────────────────────────────────────

def _run_script(cmd: list[str]) -> None:
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.warning(f"Script exited with code {result.returncode}")


def train_all_models(
    epochs: int, config: str, model_config: str,
    dataset: str = "synthetic", cmapss_subset: str = "FD001",
) -> None:
    dataset_flags = ["--dataset", dataset]
    if dataset == "cmapss":
        dataset_flags += ["--cmapss-subset", cmapss_subset]

    logger.info("=" * 60)
    logger.info("Training LSTM…")
    _run_script([
        sys.executable, "scripts/train_neural_model.py",
        "--model-type", "lstm",
        "--epochs", str(epochs),
        "--config", config,
        "--model-config", model_config,
        *dataset_flags,
    ])

    logger.info("=" * 60)
    logger.info("Training CNN…")
    _run_script([
        sys.executable, "scripts/train_neural_model.py",
        "--model-type", "cnn",
        "--epochs", str(epochs),
        "--config", config,
        "--model-config", model_config,
        *dataset_flags,
    ])

    logger.info("=" * 60)
    logger.info("Training baselines (RF + LR)…")
    _run_script([
        sys.executable, "scripts/train_baseline.py",
        "--config", config,
        *dataset_flags,
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Load predictions
# ──────────────────────────────────────────────────────────────────────────────

def _load_neural_model(ckpt_path: Path, model_cls, sensor_cols: list[str]) -> torch.nn.Module | None:
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    if model_cls is LSTMClassifier:
        model = LSTMClassifier(
            input_size=len(sensor_cols),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=0.0,
            bidirectional=cfg.get("bidirectional", False),
        )
    else:
        model = TemporalCNNClassifier(
            input_size=len(sensor_cols),
            num_channels=cfg.get("num_channels", 64),
            kernel_size=cfg.get("kernel_size", 7),
            num_blocks=cfg.get("num_blocks", 4),
            dropout=0.0,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def collect_predictions(
    X_test: np.ndarray,
    sensor_cols: list[str],
    ckpt_base: str = "models/checkpoints",
    baseline_base: str = "models/baselines",
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}

    # LSTM
    for lstm_path in [
        Path(ckpt_base) / "lstm" / "best_model.pt",
        Path(ckpt_base) / "best_model.pt",
    ]:
        if "LSTM" not in predictions:
            model = _load_neural_model(lstm_path, LSTMClassifier, sensor_cols)
            if model is not None:
                with torch.no_grad():
                    proba = model.predict_proba(
                        torch.from_numpy(X_test.astype(np.float32))
                    ).numpy()
                predictions["LSTM"] = proba
                logger.info(f"LSTM loaded from {lstm_path}")

    # CNN
    cnn_path = Path(ckpt_base) / "cnn" / "best_model.pt"
    model = _load_neural_model(cnn_path, TemporalCNNClassifier, sensor_cols)
    if model is not None:
        with torch.no_grad():
            proba = model.predict_proba(
                torch.from_numpy(X_test.astype(np.float32))
            ).numpy()
        predictions["CNN1D"] = proba
        logger.info(f"CNN loaded from {cnn_path}")

    # Random Forest
    rf_path = Path(baseline_base) / "randomforest.joblib"
    if rf_path.exists():
        rf = RandomForestBaseline.load(rf_path)
        predictions["RandomForest"] = rf.predict_proba(X_test)
        logger.info("RandomForest loaded.")

    # Logistic Regression
    lr_path = Path(baseline_base) / "logisticregression.joblib"
    if lr_path.exists():
        lr = LogisticRegressionBaseline.load(lr_path)
        predictions["LogisticRegression"] = lr.predict_proba(X_test)
        logger.info("LogisticRegression loaded.")

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Compute all metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    cost_matrix: CostMatrix,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name, proba in predictions.items():
        thresh, _ = find_optimal_threshold(y_true, proba, metric="f1")
        cost_thresh, _ = find_cost_optimal_threshold(y_true, proba, cost_matrix)
        metrics = compute_classification_metrics(y_true, proba, threshold=thresh)
        metrics["brier_score"] = compute_brier_score(y_true, proba)
        metrics["ece"] = compute_expected_calibration_error(y_true, proba)

        # Cost at cost-optimal threshold
        cost_comparison = cost_threshold_comparison(y_true, proba, cost_matrix, thresh)
        metrics["expected_cost_f1_threshold"] = cost_comparison["f1_threshold_decision"]["total_cost"]
        metrics["expected_cost_optimal_threshold"] = cost_comparison["cost_optimal_decision"]["total_cost"]
        metrics["cost_saving_vs_f1"] = cost_comparison["cost_saving_vs_f1"]
        metrics["cost_optimal_threshold"] = cost_thresh

        results[name] = metrics
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Bootstrap CIs (key models)
# ──────────────────────────────────────────────────────────────────────────────

def compute_bootstrap_cis(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    all_metrics: dict[str, dict],
) -> dict[str, dict]:
    ci_results: dict[str, dict] = {}
    for name, proba in predictions.items():
        thresh = all_metrics[name]["threshold"]
        ci = bootstrap_confidence_intervals(
            y_true, proba, n_samples=500, threshold=thresh, random_seed=42
        )
        ci_results[name] = ci
    return ci_results


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Print metrics table
# ──────────────────────────────────────────────────────────────────────────────

DISPLAY_METRICS = [
    "f1", "roc_auc", "pr_auc", "recall", "precision", "mcc",
    "brier_score", "ece", "expected_cost_optimal_threshold",
]

def print_metrics_table(all_metrics: dict[str, dict], ci_results: dict[str, dict]) -> str:
    col_w = 12
    header_cols = ["Model"] + DISPLAY_METRICS
    widths = [22] + [col_w] * len(DISPLAY_METRICS)

    sep = "+" + "+".join("-" * w for w in widths) + "+"
    header = "|" + "|".join(
        f" {c:<{w-2}} " if i == 0 else f" {c:>{w-2}} "
        for i, (c, w) in enumerate(zip(header_cols, widths))
    ) + "|"

    lines = [sep, header, sep]
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1].get("f1", 0)):
        vals = [name]
        for col in DISPLAY_METRICS:
            v = m.get(col, 0.0)
            if col == "expected_cost_optimal_threshold":
                vals.append(f"${v:,.0f}")
            else:
                vals.append(f"{v:.4f}")
        row = "|" + "|".join(
            f" {v:<{w-2}} " if i == 0 else f" {v:>{w-2}} "
            for i, (v, w) in enumerate(zip(vals, widths))
        ) + "|"
        lines.append(row)
        # Add CI row for f1 and roc_auc
        if name in ci_results:
            ci = ci_results[name]
            f1_ci = ci.get("f1", {})
            auc_ci = ci.get("roc_auc", {})
            ci_str = f"  95% CI  F1=[{f1_ci.get('lower',0):.3f},{f1_ci.get('upper',0):.3f}]  AUC=[{auc_ci.get('lower',0):.3f},{auc_ci.get('upper',0):.3f}]"
            lines.append("|" + f" {ci_str:<{sum(widths) + len(widths) - 3}} " + "|")
    lines.append(sep)
    table = "\n".join(lines)
    logger.info("\n" + table)
    return table



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Paths (dataset-aware) ─────────────────────────────────────────────────
    if args.dataset == "cmapss":
        subset = args.cmapss_subset
        report_dir  = Path(f"reports/benchmark/cmapss/{subset}")
        ckpt_base   = f"models/checkpoints/cmapss/{subset}"
        baseline_base = f"models/baselines/cmapss/{subset}"
    else:
        report_dir  = Path("reports/benchmark")
        ckpt_base   = "models/checkpoints"
        baseline_base = "models/baselines"

    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    config_paths = [args.config, args.model_config]
    if args.dataset == "cmapss":
        config_paths.append("configs/cmapss_config.yaml")
    cfg = load_config(*config_paths)
    if args.dataset == "cmapss":
        cfg.setdefault("dataset", {}).setdefault("cmapss", {})["subset"] = args.cmapss_subset

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.dataset == "synthetic":
        generate_data(args.n_machines)

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.skip_training:
        train_all_models(
            args.epochs, args.config, args.model_config,
            dataset=args.dataset, cmapss_subset=args.cmapss_subset,
        )

    # ── Preprocessor / test split ─────────────────────────────────────────────
    df, sensor_cols = load_dataset(
        dataset_type=args.dataset, cfg=cfg, data_path=str(DATA_PATH)
    )

    # Prefer saved preprocessor (same split as during training)
    preprocessor: SensorDataPreprocessor | None = None
    for ckpt_dir in [
        Path(ckpt_base) / "lstm",
        Path(ckpt_base) / "cnn",
        Path(ckpt_base),
    ]:
        pp_path = ckpt_dir / "preprocessor.joblib"
        if pp_path.exists():
            preprocessor = SensorDataPreprocessor.load(pp_path)
            logger.info(f"Using preprocessor from {pp_path}")
            break

    if preprocessor is None:
        logger.warning("No saved preprocessor found. Creating fresh split.")
        preprocessor = SensorDataPreprocessor(
            sensor_columns=sensor_cols,
            target_column="failure_imminent",
            window_size=cfg.data.window_size,
            step_size=cfg.data.step_size,
            test_size=cfg.data.test_size,
            val_size=cfg.data.val_size,
            random_seed=cfg.data.random_seed,
        )

    (_, _), (_, _), (X_test, y_test) = preprocessor.fit_transform(df)
    y_true = y_test.astype(int)
    logger.info(f"Test set: {len(y_true):,} windows | positive rate: {y_true.mean():.2%}")

    # ── Collect predictions ───────────────────────────────────────────────────
    predictions = collect_predictions(X_test, sensor_cols, ckpt_base, baseline_base)
    if not predictions:
        logger.error("No trained models found. Run without --skip-training.")
        sys.exit(1)
    logger.info(f"Models loaded: {list(predictions.keys())}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    cost_matrix = CostMatrix(cost_fn=COST_FN, cost_fp=COST_FP)
    all_metrics = compute_all_metrics(y_true, predictions, cost_matrix)

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    logger.info("Computing bootstrap confidence intervals (n=500)…")
    ci_results = compute_bootstrap_cis(y_true, predictions, all_metrics)

    # ── Statistical significance ───────────────────────────────────────────────
    sig_results: dict = {}
    if len(predictions) >= 2:
        logger.info("Running pairwise significance tests…")
        sig_results = compare_models_statistically(
            y_true, predictions, threshold=0.5, n_permutations=1000
        )

    # ── Table ─────────────────────────────────────────────────────────────────
    table_str = print_metrics_table(all_metrics, ci_results)
    table_path = report_dir / "metrics_table.txt"
    table_path.write_text(table_str, encoding="utf-8")

    # ── Plots ─────────────────────────────────────────────────────────────────
    viz = EvaluationVisualizer(output_dir=str(report_dir))
    viz.plot_roc_curve(y_true, predictions, save_name="roc_all_models.png")
    viz.plot_pr_curve(y_true, predictions, save_name="pr_all_models.png")
    viz.plot_calibration(y_true, predictions, save_name="calibration_all_models.png")
    viz.plot_model_comparison(
        all_metrics,
        metrics=["f1", "roc_auc", "pr_auc", "recall", "precision", "mcc"],
        save_name="comparison_bar.png",
    )
    viz.plot_cost_curves(y_true, predictions, cost_matrix, save_name="cost_curve_all_models.png")
    for name, proba in predictions.items():
        thresh = all_metrics[name]["threshold"]
        safe_name = name.lower().replace(" ", "_")
        viz.plot_confusion_matrix(
            y_true, proba, threshold=thresh,
            model_name=name, save_name=f"{safe_name}_confusion_matrix.png",
        )
        viz.plot_threshold_analysis(
            y_true, proba, model_name=name,
            save_name=f"{safe_name}_threshold_analysis.png",
        )
        viz.plot_prediction_distribution(
            y_true, proba, model_name=name,
            save_name=f"{safe_name}_prediction_dist.png",
        )
    logger.info(f"All plots saved → {report_dir}")

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "dataset": args.dataset,
        "cmapss_subset": args.cmapss_subset if args.dataset == "cmapss" else None,
        "n_test_windows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "cost_matrix": {"cost_fn": COST_FN, "cost_fp": COST_FP},
        "models": list(predictions.keys()),
        "metrics": all_metrics,
        "bootstrap_ci": ci_results,
        "significance_tests": sig_results,
    }
    report_path = report_dir / "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nBenchmark complete.")
    logger.info(f"  Report  → {report_path}")
    logger.info(f"  Table   → {table_path}")
    logger.info(f"  Figures → {report_dir}/")


if __name__ == "__main__":
    main()
