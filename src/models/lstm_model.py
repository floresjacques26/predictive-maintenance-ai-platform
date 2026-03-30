"""LSTM-based binary classifier for time-series failure prediction.

Architecture
------------
Input  →  LSTM (stacked, dropout between layers)  →  LayerNorm
       →  Dropout  →  Linear(hidden, 1)  →  logit (sigmoid at inference)

Raw logits are returned during training for use with BCEWithLogitsLoss,
which is numerically more stable than applying sigmoid then BCELoss.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Multi-layer LSTM binary classifier.

    Args:
        input_size: Number of input features (sensor channels).
        hidden_size: Number of LSTM hidden units per layer.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability (applied between LSTM layers and before FC).
        bidirectional: Use bidirectional LSTM; doubles effective hidden size.
        output_size: Output dimensionality (1 for binary classification).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM: dropout only applied between layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input_size = hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(fc_input_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(fc_input_size, output_size)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logit tensor of shape (batch,).
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden * D)
        last = lstm_out[:, -1, :]           # take representation at final timestep
        out = self.layer_norm(last)
        out = self.dropout(out)
        out = self.fc(out)                  # (batch, 1)
        return out.squeeze(1)               # (batch,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated failure probability in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        """Xavier / orthogonal initialisation for stable training."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # Set forget gate bias to 1 to mitigate vanishing gradients
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
