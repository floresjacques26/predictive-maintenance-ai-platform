"""Model checkpointing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    path: str | Path,
) -> None:
    """Save full training state to disk.

    Args:
        model: PyTorch model.
        optimizer: Current optimizer state.
        epoch: Current epoch number.
        metrics: Validation metrics dict (for provenance).
        config: Model / training configuration.
        path: Destination file path (.pt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Restore model (and optionally optimizer) from a checkpoint.

    Args:
        path: Checkpoint file path.
        model: Model instance with matching architecture.
        optimizer: Optional optimizer to restore state into.
        device: Target device.

    Returns:
        Full checkpoint dict (includes epoch, metrics, config).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint
