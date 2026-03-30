"""FastAPI application for predictive maintenance inference.

Endpoints
---------
GET  /health    — liveness + model status
POST /predict   — failure risk prediction from sensor window
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.predictor import MaintenancePredictor
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Configuration from environment (or defaults for local dev)
# ------------------------------------------------------------------
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH", "models/checkpoints/best_model.pt"
)
PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH", "models/checkpoints/preprocessor.joblib"
)
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")

# ------------------------------------------------------------------
# Application state
# ------------------------------------------------------------------
predictor: MaintenancePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release on shutdown."""
    global predictor
    try:
        predictor = MaintenancePredictor(
            checkpoint_path=CHECKPOINT_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            device=INFERENCE_DEVICE,
        )
        logger.info("Predictor initialised successfully.")
    except FileNotFoundError as exc:
        logger.warning(f"Model files not found — API running in degraded mode. {exc}")
        predictor = None
    yield
    predictor = None


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Real-time failure risk prediction from industrial sensor time-series. "
        "Submit a window of sensor readings and receive a failure probability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health() -> HealthResponse:
    """Return API and model status."""
    loaded = predictor is not None and predictor.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MaintenancePredictor.VERSION,
        n_parameters=predictor.n_parameters if loaded else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict failure probability from a sensor time-series window.

    The ``sensor_window`` list should contain readings ordered from oldest
    to most recent.  At least 1 reading is required; the model was trained
    on windows of 50 timesteps and will perform best with similar lengths.

    Returns a failure probability in [0, 1] and a binary ``failure_imminent``
    flag based on the supplied ``threshold``.
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model and provide checkpoint paths.",
        )

    try:
        readings = [r.model_dump() for r in request.sensor_window]
        result = predictor.predict(readings, threshold=request.threshold)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        machine_id=request.machine_id,
        failure_probability=result["failure_probability"],
        failure_imminent=result["failure_imminent"],
        threshold_used=request.threshold,
        window_length=len(request.sensor_window),
        model_version=MaintenancePredictor.VERSION,
    )
