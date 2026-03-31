"""Unified command-line interface for the predictive maintenance platform.

Usage
-----
python cli.py --help
python cli.py generate --n-machines 100
python cli.py train --model lstm --epochs 100
python cli.py train --model baseline
python cli.py evaluate --model-type lstm --checkpoint models/checkpoints/best_model.pt
python cli.py tune --n-trials 50
python cli.py compare
python cli.py ablation --study all
python cli.py serve --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Shared options ─────────────────────────────────────────────────────────────

def _device_option():
    return click.option("--device", default=None, help="cpu | cuda | mps (auto-detect if omitted)")


def _data_option():
    return click.option(
        "--data-path", default="data/synthetic/sensor_data.csv",
        show_default=True, help="Path to sensor CSV dataset."
    )


def _config_option():
    return click.option(
        "--config", default="configs/base_config.yaml",
        show_default=True, help="Base config YAML path."
    )


def _dataset_options():
    """Returns a composed decorator for --dataset / --cmapss-subset."""
    def decorator(f):
        f = click.option(
            "--cmapss-subset",
            type=click.Choice(["FD001", "FD002", "FD003", "FD004"]),
            default="FD001", show_default=True,
            help="CMAPSS sub-dataset (only used when --dataset cmapss).",
        )(f)
        f = click.option(
            "--dataset",
            type=click.Choice(["synthetic", "cmapss"]),
            default="synthetic", show_default=True,
            help="Dataset to use: synthetic (default) or NASA CMAPSS.",
        )(f)
        return f
    return decorator


# ── CLI group ──────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="1.0.0", prog_name="pm-ai")
def cli():
    """Predictive Maintenance AI Platform — unified CLI."""


# ── generate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--n-machines", default=100, show_default=True, help="Number of simulated machines.")
@click.option("--failure-horizon", default=30, show_default=True, help="Lookahead label horizon.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--output-dir", default="data/synthetic", show_default=True)
def generate(n_machines: int, failure_horizon: int, seed: int, output_dir: str):
    """Generate synthetic sensor dataset."""
    from src.data.synthetic_generator import SyntheticSensorDataGenerator
    from src.utils.logger import get_logger

    logger = get_logger("cli.generate")
    logger.info(f"Generating {n_machines} machines (horizon={failure_horizon}, seed={seed})")
    gen = SyntheticSensorDataGenerator(
        n_machines=n_machines, failure_horizon=failure_horizon, random_seed=seed,
    )
    df = gen.generate(output_dir=output_dir)
    logger.info(f"Done — {len(df):,} rows, {df['failure_imminent'].mean():.2%} failure-imminent")


# ── train ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", type=click.Choice(["lstm", "cnn", "baseline"]), required=True)
@click.option("--epochs", default=None, type=int, help="Override config epochs.")
@click.option("--checkpoint-dir", default=None, show_default=False,
              help="Checkpoint directory (auto-derived from dataset/model if omitted).")
@click.option("--report-dir", default=None, show_default=False,
              help="Report directory (auto-derived from dataset/model if omitted).")
@_dataset_options()
@_data_option()
@_config_option()
@_device_option()
def train(model: str, epochs, checkpoint_dir: str | None, report_dir: str | None,
          dataset: str, cmapss_subset: str,
          data_path: str, config: str, device: str | None):
    """Train LSTM, CNN, or sklearn baseline models."""
    import subprocess

    dataset_flags = ["--dataset", dataset]
    if dataset == "cmapss":
        dataset_flags += ["--cmapss-subset", cmapss_subset]

    if model == "baseline":
        cmd = [
            sys.executable, "scripts/train_baseline.py",
            "--data-path", data_path,
            "--config", config,
            *dataset_flags,
        ]
        if checkpoint_dir:
            cmd += ["--output-dir", checkpoint_dir]
        if report_dir:
            cmd += ["--report-dir", report_dir]
    else:
        cmd = [
            sys.executable, "scripts/train_neural_model.py",
            "--model-type", model,
            "--data-path", data_path,
            "--config", config,
            *dataset_flags,
        ]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        if device:
            cmd += ["--device", device]
        if checkpoint_dir:
            cmd += ["--checkpoint-dir", checkpoint_dir]
        if report_dir:
            cmd += ["--report-dir", report_dir]

    subprocess.run(cmd, check=True)


# ── evaluate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model-type", type=click.Choice(["lstm", "rf", "lr"]), required=True)
@click.option("--checkpoint", required=True, help="Path to model checkpoint/joblib file.")
@click.option("--preprocessor", default="models/checkpoints/preprocessor.joblib", show_default=True)
@click.option("--bootstrap-samples", default=1000, show_default=True)
@click.option("--report-dir", default="reports/evaluation", show_default=True)
@_data_option()
@_config_option()
def evaluate(model_type: str, checkpoint: str, preprocessor: str,
             bootstrap_samples: int, report_dir: str, data_path: str, config: str):
    """Evaluate a trained model with full statistical analysis."""
    import subprocess
    cmd = [
        sys.executable, "scripts/evaluate_model.py",
        "--model-type", model_type,
        "--checkpoint", checkpoint,
        "--preprocessor", preprocessor,
        "--data-path", data_path,
        "--config", config,
        "--report-dir", report_dir,
        "--bootstrap-samples", str(bootstrap_samples),
    ]
    subprocess.run(cmd, check=True)


# ── tune ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--n-trials", default=50, show_default=True)
@click.option("--timeout", default=3600.0, show_default=True, help="Wall-clock timeout (seconds).")
@click.option("--epochs-per-trial", default=30, show_default=True)
@click.option("--output-dir", default="models/hpo_checkpoints", show_default=True)
@_data_option()
@_config_option()
@_device_option()
def tune(n_trials: int, timeout: float, epochs_per_trial: int, output_dir: str,
         data_path: str, config: str, device: str | None):
    """Run Optuna hyperparameter search for LSTM."""
    import subprocess
    cmd = [
        sys.executable, "scripts/run_hyperparameter_search.py",
        "--data-path", data_path,
        "--config", config,
        "--n-trials", str(n_trials),
        "--timeout", str(timeout),
        "--epochs-per-trial", str(epochs_per_trial),
        "--output-dir", output_dir,
    ]
    if device:
        cmd += ["--device", device]
    subprocess.run(cmd, check=True)


# ── compare ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--skip-training", is_flag=True, default=False,
              help="Skip training — benchmark existing checkpoints only.")
@click.option("--epochs", default=50, show_default=True, help="Max training epochs.")
@_dataset_options()
@_config_option()
def compare(skip_training: bool, epochs: int, dataset: str, cmapss_subset: str, config: str):
    """Benchmark all models (LSTM, CNN, RF, LR) on the held-out test set."""
    import subprocess
    cmd = [
        sys.executable, "scripts/run_full_benchmark.py",
        "--config", config,
        "--epochs", str(epochs),
        "--dataset", dataset,
        "--cmapss-subset", cmapss_subset,
    ]
    if skip_training:
        cmd.append("--skip-training")
    subprocess.run(cmd, check=True)


# ── ablation ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--study", type=click.Choice(["window_size", "pos_weight", "capacity", "lstm_vs_cnn", "all"]),
              default="all", show_default=True)
@click.option("--epochs", default=30, show_default=True)
@click.option("--output", default="reports/ablation/results.json", show_default=True)
@_data_option()
@_config_option()
@_device_option()
def ablation(study: str, epochs: int, output: str, data_path: str, config: str, device: str | None):
    """Run ablation studies (window size, pos_weight, capacity, LSTM vs CNN)."""
    import subprocess
    cmd = [
        sys.executable, "scripts/ablation_study.py",
        "--study", study,
        "--data-path", data_path,
        "--config", config,
        "--epochs", str(epochs),
        "--output", output,
    ]
    if device:
        cmd += ["--device", device]
    subprocess.run(cmd, check=True)


# ── serve ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Enable hot-reload (development only).")
@click.option("--checkpoint", default="models/checkpoints/best_model.pt",
              envvar="CHECKPOINT_PATH", show_default=True)
@click.option("--preprocessor", default="models/checkpoints/preprocessor.joblib",
              envvar="PREPROCESSOR_PATH", show_default=True)
def serve(host: str, port: int, reload: bool, checkpoint: str, preprocessor: str):
    """Start the FastAPI inference server."""
    import os
    import uvicorn

    os.environ["CHECKPOINT_PATH"] = checkpoint
    os.environ["PREPROCESSOR_PATH"] = preprocessor

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ── predict (single inference from CLI) ───────────────────────────────────────

@cli.command()
@click.option("--checkpoint", default="models/checkpoints/best_model.pt", show_default=True)
@click.option("--preprocessor", default="models/checkpoints/preprocessor.joblib", show_default=True)
@click.option("--temperature", required=True, type=float)
@click.option("--vibration", required=True, type=float)
@click.option("--pressure", required=True, type=float)
@click.option("--rpm", required=True, type=float)
@click.option("--current", required=True, type=float)
@click.option("--threshold", default=0.5, show_default=True)
def predict(checkpoint: str, preprocessor: str, temperature: float, vibration: float,
            pressure: float, rpm: float, current: float, threshold: float):
    """Run single-timestep inference (repeats reading to fill window)."""
    from src.api.predictor import MaintenancePredictor

    pred = MaintenancePredictor(
        checkpoint_path=checkpoint,
        preprocessor_path=preprocessor,
    )
    reading = {
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "current": current,
    }
    # Repeat to create a minimal window (real use should pass a full window)
    window_size = pred._preprocessor.window_size  # type: ignore[union-attr]
    readings = [reading] * window_size

    result = pred.predict(readings, threshold=threshold)
    click.echo(f"Failure probability : {result['failure_probability']:.4f}")
    click.echo(f"Failure imminent    : {result['failure_imminent']}")
    click.echo(f"Threshold used      : {threshold}")


if __name__ == "__main__":
    cli()
