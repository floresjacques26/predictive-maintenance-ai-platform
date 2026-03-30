"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """A single timestep of multi-variate sensor data."""

    temperature: Annotated[float, Field(gt=0, lt=500, description="Temperature in °C")]
    vibration: Annotated[float, Field(ge=0, lt=100, description="Vibration in mm/s")]
    pressure: Annotated[float, Field(ge=0, lt=50, description="Pressure in bar")]
    rpm: Annotated[float, Field(ge=0, lt=10000, description="Rotational speed in RPM")]
    current: Annotated[float, Field(ge=0, lt=100, description="Electrical current in A")]

    model_config = {"json_schema_extra": {
        "example": {
            "temperature": 80.0,
            "vibration": 0.6,
            "pressure": 5.8,
            "rpm": 1790.0,
            "current": 12.5,
        }
    }}


class PredictRequest(BaseModel):
    """Inference request: a time-series window of sensor readings."""

    sensor_window: Annotated[
        list[SensorReading],
        Field(min_length=1, description="Ordered list of sensor readings (oldest → newest)"),
    ]
    machine_id: str | None = Field(default=None, description="Optional machine identifier")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for failure classification",
    )

    @field_validator("sensor_window")
    @classmethod
    def window_not_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("sensor_window must contain at least one reading.")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "pump-001",
            "threshold": 0.5,
            "sensor_window": [
                {"temperature": 80.0, "vibration": 0.6, "pressure": 5.8, "rpm": 1790.0, "current": 12.5}
            ] * 3,
        }
    }}


class PredictResponse(BaseModel):
    """Prediction result with probability, class label, and metadata."""

    machine_id: str | None
    failure_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    failure_imminent: bool
    threshold_used: float
    window_length: int
    model_version: str

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "pump-001",
            "failure_probability": 0.87,
            "failure_imminent": True,
            "threshold_used": 0.5,
            "window_length": 50,
            "model_version": "1.0.0",
        }
    }}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    n_parameters: int | None = None
