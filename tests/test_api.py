"""Tests for the FastAPI inference endpoint.

These tests run without a trained model (no checkpoint required).
The API is tested in degraded mode where model is not loaded,
and the schema/validation logic is tested independently.
"""

import pytest
from fastapi.testclient import TestClient

# ── We must patch predictor before importing main ───────────────────────────
from unittest.mock import MagicMock, patch

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse, SensorReading


# ── Schema validation tests (no HTTP needed) ─────────────────────────────────

def make_reading(**kwargs) -> dict:
    defaults = {"temperature": 80.0, "vibration": 0.5, "pressure": 6.0, "rpm": 1800.0, "current": 12.0}
    defaults.update(kwargs)
    return defaults


def test_sensor_reading_valid():
    r = SensorReading(**make_reading())
    assert r.temperature == 80.0


def test_sensor_reading_rejects_negative_vibration():
    with pytest.raises(Exception):
        SensorReading(**make_reading(vibration=-1.0))


def test_sensor_reading_rejects_zero_temperature():
    with pytest.raises(Exception):
        SensorReading(**make_reading(temperature=0.0))


def test_predict_request_valid():
    readings = [SensorReading(**make_reading())] * 5
    req = PredictRequest(sensor_window=readings, machine_id="test-001", threshold=0.5)
    assert len(req.sensor_window) == 5
    assert req.machine_id == "test-001"


def test_predict_request_empty_window_raises():
    with pytest.raises(Exception):
        PredictRequest(sensor_window=[])


def test_predict_request_threshold_validation():
    readings = [SensorReading(**make_reading())]
    with pytest.raises(Exception):
        PredictRequest(sensor_window=readings, threshold=1.5)
    with pytest.raises(Exception):
        PredictRequest(sensor_window=readings, threshold=-0.1)


def test_predict_response_model():
    resp = PredictResponse(
        machine_id="m1",
        failure_probability=0.85,
        failure_imminent=True,
        threshold_used=0.5,
        window_length=50,
        model_version="1.0.0",
    )
    assert resp.failure_imminent is True
    assert 0.0 <= resp.failure_probability <= 1.0


def test_health_response_model():
    h = HealthResponse(status="ok", model_loaded=True, model_version="1.0.0", n_parameters=50000)
    assert h.status == "ok"
    assert h.n_parameters == 50000


# ── HTTP endpoint tests ───────────────────────────────────────────────────────

@pytest.fixture
def client_no_model():
    """TestClient with no model loaded (degraded mode)."""
    with patch("src.api.main.predictor", None):
        from src.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def mock_predictor():
    pred = MagicMock()
    pred.is_loaded = True
    pred.n_parameters = 100_000
    pred.VERSION = "1.0.0"
    pred.predict.return_value = {
        "failure_probability": 0.78,
        "failure_imminent": True,
    }
    return pred


@pytest.fixture
def client_with_model(mock_predictor):
    """TestClient with a mocked predictor."""
    with patch("src.api.main.predictor", mock_predictor):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint_no_model(client_no_model):
    resp = client_no_model.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"


def test_health_endpoint_with_model(client_with_model):
    resp = client_with_model.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_predict_returns_503_when_no_model(client_no_model):
    payload = {
        "sensor_window": [make_reading()] * 3,
        "threshold": 0.5,
    }
    resp = client_no_model.post("/predict", json=payload)
    assert resp.status_code == 503


def test_predict_success(client_with_model):
    payload = {
        "machine_id": "pump-42",
        "threshold": 0.5,
        "sensor_window": [make_reading()] * 5,
    }
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "failure_probability" in body
    assert "failure_imminent" in body
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["machine_id"] == "pump-42"
    assert body["window_length"] == 5


def test_predict_invalid_sensor_value(client_with_model):
    payload = {
        "sensor_window": [make_reading(temperature=-50)],
    }
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 422  # Pydantic validation error


def test_predict_missing_field(client_with_model):
    payload = {
        "sensor_window": [{"temperature": 80.0, "vibration": 0.5}],  # missing fields
    }
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty_window(client_with_model):
    payload = {"sensor_window": []}
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_threshold_propagated(client_with_model):
    payload = {
        "threshold": 0.8,
        "sensor_window": [make_reading()] * 3,
    }
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["threshold_used"] == 0.8
