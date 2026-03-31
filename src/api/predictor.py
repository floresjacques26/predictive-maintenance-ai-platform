"""Model loading and inference logic, decoupled from the HTTP layer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from src.data.preprocessing import SensorDataPreprocessor
from src.models.lstm_model import LSTMClassifier
from src.utils.checkpointing import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

SENSOR_ORDER = ["temperature", "vibration", "pressure", "rpm", "current"]


class MaintenancePredictor:
    """Load a trained LSTM + preprocessor and run inference.

    Args:
        checkpoint_path: Path to .pt model checkpoint.
        preprocessor_path: Path to joblib-serialised SensorDataPreprocessor.
        device: Inference device.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        checkpoint_path: str | Path,
        preprocessor_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self._model: Optional[LSTMClassifier] = None
        self._preprocessor: Optional[SensorDataPreprocessor] = None
        self._n_params: Optional[int] = None

        self._load(Path(checkpoint_path), Path(preprocessor_path))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._preprocessor is not None

    @property
    def n_parameters(self) -> Optional[int]:
        return self._n_params

    def predict(
        self,
        sensor_readings: list[dict[str, float]],
        threshold: float = 0.5,
    ) -> dict:
        """Run single-window inference.

        Args:
            sensor_readings: List of dicts with keys matching SENSOR_ORDER.
            threshold: Decision threshold.

        Returns:
            Dict with 'failure_probability' and 'failure_imminent'.
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded.")

        window = self._prepare_input(sensor_readings)
        with torch.no_grad():
            proba = self._model.predict_proba(window).item()  # type: ignore[union-attr]

        return {
            "failure_probability": round(proba, 6),
            "failure_imminent": proba >= threshold,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, checkpoint_path: Path, preprocessor_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        self._preprocessor = SensorDataPreprocessor.load(preprocessor_path)

        # Reconstruct model architecture from checkpoint config
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = checkpoint.get("config", {})
        state = checkpoint["model_state_dict"]

        # Derive input_size from actual weights (robust to wrong config values)
        input_size = int(state["lstm.weight_ih_l0"].shape[1])
        hidden_size = int(state["lstm.weight_ih_l0"].shape[0]) // 4
        num_layers = cfg.get("num_layers", 2)
        bidirectional = cfg.get("bidirectional", False)

        self._model = LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,  # no dropout at inference
            bidirectional=bidirectional,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.device)
        self._model.eval()

        self._n_params = self._model.count_parameters()
        logger.info(
            f"Model loaded from {checkpoint_path} "
            f"({self._n_params:,} parameters)"
        )

    def _prepare_input(self, readings: list[dict[str, float]]) -> torch.Tensor:
        """Convert raw sensor dicts → normalised (1, T, F) tensor."""
        arr = np.array(
            [[r[col] for col in SENSOR_ORDER] for r in readings],
            dtype=np.float32,
        )  # (T, F)

        # Apply fitted scaler
        arr = self._preprocessor.scaler.transform(arr)  # type: ignore[union-attr]

        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, T, F)
        return tensor
