"""Tests for LSTM model architecture and forward pass."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.lstm_model import LSTMClassifier


@pytest.fixture
def model_config():
    return {"input_size": 5, "hidden_size": 32, "num_layers": 2, "dropout": 0.1}


@pytest.fixture
def model(model_config):
    return LSTMClassifier(**model_config)


@pytest.fixture
def batch():
    """(batch=4, seq_len=20, features=5)."""
    return torch.randn(4, 20, 5)


# ── Architecture ─────────────────────────────────────────────────────────────

def test_output_shape(model, batch):
    out = model(batch)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_output_is_logit_not_probability(model, batch):
    """Raw forward pass should return logits, not probabilities."""
    out = model(batch)
    # Logits can be outside [0, 1]; at least some should be
    assert not ((out >= 0) & (out <= 1)).all(), "Output looks like probabilities, not logits"


def test_predict_proba_in_zero_one(model, batch):
    proba = model.predict_proba(batch)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert proba.shape == (4,)


def test_parameter_count_reasonable(model):
    n = model.count_parameters()
    assert n > 0
    assert n < 10_000_000  # sanity upper bound


def test_bidirectional_model():
    model_bi = LSTMClassifier(input_size=5, hidden_size=32, num_layers=1, bidirectional=True)
    x = torch.randn(4, 10, 5)
    out = model_bi(x)
    assert out.shape == (4,)


def test_single_layer_no_dropout_error():
    """num_layers=1 should not raise (dropout is disabled between layers)."""
    model = LSTMClassifier(input_size=3, hidden_size=16, num_layers=1, dropout=0.5)
    x = torch.randn(2, 10, 3)
    _ = model(x)


def test_gradient_flows():
    model = LSTMClassifier(input_size=5, hidden_size=16, num_layers=1)
    x = torch.randn(4, 10, 5, requires_grad=False)
    y = torch.ones(4)
    criterion = nn.BCEWithLogitsLoss()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_eval_mode_deterministic(model, batch):
    """In eval mode (no dropout), same input → same output."""
    model.eval()
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    torch.testing.assert_close(out1, out2)


def test_train_mode_can_differ_with_dropout():
    """In train mode with high dropout, outputs should sometimes differ."""
    model = LSTMClassifier(input_size=5, hidden_size=32, num_layers=2, dropout=0.9)
    model.train()
    x = torch.randn(8, 20, 5)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    # With 90% dropout they very likely differ
    assert not torch.allclose(out1, out2), "Outputs identical under high dropout (unexpected)"


def test_variable_sequence_lengths():
    """Model should handle different sequence lengths without error."""
    model = LSTMClassifier(input_size=5, hidden_size=16, num_layers=1)
    model.eval()
    for seq_len in [1, 10, 50, 100]:
        x = torch.randn(2, seq_len, 5)
        out = model(x)
        assert out.shape == (2,), f"Failed for seq_len={seq_len}"


def test_batch_size_one():
    model = LSTMClassifier(input_size=5, hidden_size=16, num_layers=1)
    model.eval()
    x = torch.randn(1, 20, 5)
    out = model(x)
    assert out.shape == (1,)


def test_init_weights_forget_gate():
    """Forget gate bias should be initialised to ~1 for better gradient flow."""
    model = LSTMClassifier(input_size=5, hidden_size=16, num_layers=1)
    for name, param in model.named_parameters():
        if "bias_ih_l0" in name:
            n = param.size(0)
            forget_bias = param.data[n // 4 : n // 2]
            assert (forget_bias == 1.0).all(), "Forget gate bias not initialised to 1"


# ── Loss compatibility ────────────────────────────────────────────────────────

def test_bce_with_logits_loss_compatible(model, batch):
    criterion = nn.BCEWithLogitsLoss()
    y = torch.randint(0, 2, (4,)).float()
    logits = model(batch)
    loss = criterion(logits, y)
    assert torch.isfinite(loss), "Loss is not finite"
    assert loss > 0
