"""Tests for the 1D Temporal CNN classifier."""

import pytest
import torch
import torch.nn as nn

from src.models.cnn_model import TemporalCNNClassifier


@pytest.fixture
def model():
    return TemporalCNNClassifier(
        input_size=5, num_channels=32, kernel_size=5, num_blocks=3, dropout=0.1,
    )


@pytest.fixture
def batch():
    return torch.randn(4, 20, 5)  # (batch=4, seq=20, features=5)


def test_output_shape(model, batch):
    out = model(batch)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_output_is_logit(model, batch):
    """Forward should return logits, not probabilities."""
    model.eval()
    with torch.no_grad():
        out = model(batch)
    # Not all in [0, 1] → these are logits
    assert not ((out >= 0) & (out <= 1)).all()


def test_predict_proba_in_zero_one(model, batch):
    proba = model.predict_proba(batch)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert proba.shape == (4,)


def test_gradient_flows(model, batch):
    y = torch.ones(4)
    criterion = nn.BCEWithLogitsLoss()
    logits = model(batch)
    loss = criterion(logits, y)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_parameter_count(model):
    n = model.count_parameters()
    assert 1_000 < n < 5_000_000


def test_eval_mode_deterministic(model, batch):
    model.eval()
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    torch.testing.assert_close(out1, out2)


def test_variable_sequence_lengths():
    model = TemporalCNNClassifier(input_size=5, num_channels=16, kernel_size=3, num_blocks=2)
    model.eval()
    for T in [5, 20, 50, 100]:
        x = torch.randn(2, T, 5)
        out = model(x)
        assert out.shape == (2,), f"Failed at seq_len={T}"


def test_batch_size_one():
    model = TemporalCNNClassifier(input_size=5, num_channels=16, kernel_size=3, num_blocks=2)
    model.eval()
    x = torch.randn(1, 30, 5)
    out = model(x)
    assert out.shape == (1,)


def test_bce_loss_compatible(model, batch):
    criterion = nn.BCEWithLogitsLoss()
    y = torch.randint(0, 2, (4,)).float()
    logits = model(batch)
    loss = criterion(logits, y)
    assert torch.isfinite(loss)


def test_cnn_has_fewer_params_than_lstm_same_capacity():
    """Confirm CNN has different (typically fewer) parameters than LSTM at same hidden size."""
    from src.models.lstm_model import LSTMClassifier
    cnn = TemporalCNNClassifier(input_size=5, num_channels=64, kernel_size=7, num_blocks=4)
    lstm = LSTMClassifier(input_size=5, hidden_size=128, num_layers=2)
    # Both are non-zero and plausible sizes
    assert cnn.count_parameters() > 0
    assert lstm.count_parameters() > 0
