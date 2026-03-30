"""1D Temporal CNN for binary classification of sensor time-series.

Architecture
------------
Multi-scale 1D convolutional feature extraction with residual connections,
followed by global average pooling and a classification head.

Design rationale vs LSTM
------------------------
* CNNs are parallelisable (no sequential dependency) → faster training
* Multiple kernel sizes capture patterns at different temporal scales
  simultaneously, whereas LSTM learns one scale at a time through state
* No vanishing gradient problem: gradients flow directly through residuals
* Typically requires more labelled data to match LSTM on long sequences

This model is included to demonstrate:
  (a) architectural diversity in comparative evaluation
  (b) that temporal convolutional networks are a legitimate alternative
      to RNNs for fixed-length classification windows
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock1D(nn.Module):
    """Two-layer 1D conv residual block with BatchNorm and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1×1 projection if channel dimensions differ
        self.projection: nn.Module = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class TemporalCNNClassifier(nn.Module):
    """Multi-scale dilated temporal CNN for binary failure classification.

    Input shape: (batch, seq_len, n_features)
    Conv layers expect (batch, n_features, seq_len) — we transpose internally.

    Architecture:
        Stem conv  →  Stack of dilated residual blocks (dilation: 1, 2, 4, 8)
        →  Global average pooling  →  Dropout  →  Linear(→1)

    Dilation pattern doubles the receptive field each block:
        Block 1: effective receptive field = kernel_size
        Block 2: 2 × kernel_size - 1
        Block k: 2^(k-1) × (kernel_size - 1) + 1

    Args:
        input_size: Number of sensor channels (features per timestep).
        num_channels: Number of convolutional filters per block.
        kernel_size: Temporal convolution kernel size.
        num_blocks: Number of dilated residual blocks.
        dropout: Dropout rate before classification head.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int = 64,
        kernel_size: int = 7,
        num_blocks: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stem: project input features to num_channels
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
        )

        # Dilated residual blocks with exponentially growing receptive field
        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(_ResidualBlock1D(num_channels, num_channels, kernel_size, dilation))
        self.blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(num_channels, 1)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size) — same convention as LSTMClassifier.

        Returns:
            Logits of shape (batch,).
        """
        # (batch, seq_len, F) → (batch, F, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        # Global average pooling over time dimension → (batch, C)
        x = x.mean(dim=2)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)  # (batch,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated failure probability in [0, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
