"""PyTorch Dataset and DataLoader factory for windowed sensor data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class SensorWindowDataset(Dataset):
    """Dataset wrapping pre-processed (X, y) numpy arrays.

    Args:
        X: Array of shape (N, window_size, n_features), dtype float32.
        y: Array of shape (N,), dtype int64.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))  # float for BCEWithLogitsLoss

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    @property
    def window_size(self) -> int:
        return self.X.shape[1]

    @property
    def pos_rate(self) -> float:
        return float(self.y.mean().item())


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    oversample: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val and test DataLoaders.

    Args:
        X_train/val/test: Feature arrays (N, T, F).
        y_train/val/test: Label arrays (N,).
        batch_size: Mini-batch size.
        num_workers: DataLoader worker processes.
        oversample: If True, use WeightedRandomSampler on training set to
                    handle class imbalance without modifying loss weights.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = SensorWindowDataset(X_train, y_train)
    val_ds = SensorWindowDataset(X_val, y_val)
    test_ds = SensorWindowDataset(X_test, y_test)

    train_sampler = None
    if oversample and y_train.sum() > 0:
        train_sampler = _make_weighted_sampler(y_train)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Create a sampler that up-weights minority class."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    weight_pos = 1.0 / n_pos if n_pos > 0 else 1.0
    weight_neg = 1.0 / n_neg if n_neg > 0 else 1.0
    weights = np.where(y == 1, weight_pos, weight_neg)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )
