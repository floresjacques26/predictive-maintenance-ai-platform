"""Dataset factory: unified loading for synthetic and CMAPSS datasets.

All training and benchmark scripts call ``load_dataset()`` instead of
directly reading CSVs, keeping dataset-specific logic in one place.

Returned DataFrame is always schema-compatible with SensorDataPreprocessor:
  machine_id | timestep | <sensor_cols> | failure_imminent | failure_event
  machine_type | degradation_progress
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

SYNTHETIC_SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]


def load_dataset(
    dataset_type: str,
    cfg,
    data_path: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load training data for the requested dataset type.

    Args:
        dataset_type: ``"synthetic"`` or ``"cmapss"``.
        cfg: Merged ``Config`` object (base_config + optional cmapss_config).
        data_path: Path to CSV for synthetic dataset.  Ignored for CMAPSS.

    Returns:
        (df, sensor_columns) where df has the standard project schema.

    Raises:
        ValueError: If dataset_type is unknown.
        FileNotFoundError: If required files are missing.
    """
    if dataset_type == "synthetic":
        return _load_synthetic(cfg, data_path)
    elif dataset_type == "cmapss":
        return _load_cmapss(cfg)
    else:
        raise ValueError(
            f"Unknown dataset_type {dataset_type!r}. "
            "Choose 'synthetic' or 'cmapss'."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_synthetic(cfg, data_path: str | None) -> tuple[pd.DataFrame, list[str]]:
    csv = data_path or "data/synthetic/sensor_data.csv"
    logger.info(f"Loading synthetic dataset: {csv}")
    df = pd.read_csv(csv)
    sensor_cols: list[str] = (
        list(cfg.get_nested("features", "sensor_columns") or SYNTHETIC_SENSOR_COLS)
    )
    logger.info(f"  {df['machine_id'].nunique()} machines | {len(df):,} rows | "
                f"failure_imminent={df['failure_imminent'].mean():.2%}")
    return df, sensor_cols


def _load_cmapss(cfg) -> tuple[pd.DataFrame, list[str]]:
    from src.data.cmapss_loader import CMAPSSLoader

    # Pull CMAPSS settings from config, with sensible defaults
    cmapss_cfg = cfg.get_nested("dataset", "cmapss") or {}
    data_dir    = cmapss_cfg.get("data_dir", "data/cmapss")
    subset      = cmapss_cfg.get("subset", "FD001")
    horizon     = cmapss_cfg.get("failure_horizon") or cfg.get_nested("data", "failure_horizon") or 30
    clip_rul    = cmapss_cfg.get("clip_rul", 125)
    sensor_idx  = cmapss_cfg.get("sensor_indices", None)
    inc_ops     = cmapss_cfg.get("include_op_settings", None)

    logger.info(f"Loading CMAPSS subset={subset} from {data_dir}")
    loader = CMAPSSLoader(
        data_dir=data_dir,
        subset=subset,
        failure_horizon=int(horizon),
        sensor_indices=sensor_idx,
        include_op_settings=inc_ops,
        clip_rul=clip_rul,
    )

    # Validate files before loading
    present = loader.files_present()
    missing = [f for f, ok in present.items() if not ok and "train" in f]
    if missing:
        raise FileNotFoundError(
            f"Missing CMAPSS files in {data_dir}: {missing}\n"
            "Run: python scripts/prepare_cmapss.py --help"
        )

    df = loader.load_train()
    return df, loader.sensor_columns
