"""Synthetic sensor time-series data generator for predictive maintenance.

Simulates multiple industrial machines, each going through:
  1. Normal operation  →  2. Degradation phase  →  3. Failure event

Version 2 improvements over original
--------------------------------------
* **Non-monotonic degradation**: sensors oscillate during degradation
  (machines sometimes appear to recover before worsening again)
* **Step faults**: a fraction of machines experience sudden abrupt failures
  in addition to, or instead of, gradual degradation
* **Correlated sensors**: temperature ↔ current correlation is enforced
  (physically: more electrical current → more heat)
* **Non-Gaussian noise**: heavy-tailed Laplace noise imitates real sensor
  outliers without introducing catastrophic spikes
* **Sensor dropout**: occasional NaN → filled with last valid value
  (forward-fill), simulating sensor outages
* **Variable failure position**: failure event is not always at the last
  timestep — it occurs in the final 10% of the lifecycle with randomness
* **Multiple machine types**: different machines have different sensor
  sensitivity profiles (e.g. vibration-dominant vs temperature-dominant)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MachineType(Enum):
    """Degradation signature profile."""
    VIBRATION_DOMINANT = "vibration_dominant"   # bearing wear
    THERMAL_DOMINANT = "thermal_dominant"       # cooling failure
    ELECTRICAL_DOMINANT = "electrical_dominant" # winding degradation
    MIXED = "mixed"                             # balanced degradation


@dataclass
class MachineConfig:
    """Full lifecycle configuration for one simulated machine."""

    machine_id: int
    machine_type: MachineType
    total_timesteps: int
    normal_phase_fraction: float
    degradation_phase_fraction: float
    failure_position_fraction: float  # where in final 10% failure occurs

    # Normal operating baselines
    temp_baseline: float        # °C
    vibration_baseline: float   # mm/s
    pressure_baseline: float    # bar
    rpm_baseline: float         # RPM
    current_baseline: float     # A

    # Per-sensor degradation sensitivity (multiplier on global severity)
    sensor_sensitivity: dict[str, float]

    noise_std: float
    noise_tail: float           # Laplace tail parameter (higher = heavier tails)
    degradation_severity: float
    has_step_fault: bool        # abrupt sudden jump before failure
    step_fault_fraction: float  # when in degradation phase step fault occurs
    dropout_probability: float  # per-timestep sensor dropout probability


class SyntheticSensorDataGenerator:
    """Generate labelled multi-variate sensor time-series for failure prediction.

    Each record contains:
      - machine_id: int
      - timestep: int
      - temperature, vibration, pressure, rpm, current: float sensor readings
      - failure_event: 1 at the exact failure timestep, 0 otherwise
      - failure_imminent: 1 if a failure_event occurs within the next
        ``failure_horizon`` steps (supervised training target)
      - machine_type: str (for stratified analysis)
      - degradation_progress: float in [0, 1] (true latent state — NOT used
        in training, useful for analysis only)

    Args:
        n_machines: Number of simulated machines.
        failure_horizon: Lookahead window for labelling failure_imminent.
        step_fault_fraction: Fraction of machines with abrupt step faults.
        sensor_dropout_prob: Per-sensor per-timestep dropout probability.
        random_seed: Reproducibility seed.
    """

    SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "rpm", "current"]

    # Per-type sensor sensitivity: how much each sensor reacts to degradation
    _TYPE_SENSITIVITY: dict[MachineType, dict[str, float]] = {
        MachineType.VIBRATION_DOMINANT: {
            "temperature": 0.5, "vibration": 2.0, "pressure": 0.3,
            "rpm": 0.8, "current": 0.6,
        },
        MachineType.THERMAL_DOMINANT: {
            "temperature": 2.0, "vibration": 0.6, "pressure": 0.4,
            "rpm": 0.5, "current": 1.5,
        },
        MachineType.ELECTRICAL_DOMINANT: {
            "temperature": 1.2, "vibration": 0.4, "pressure": 0.3,
            "rpm": 0.9, "current": 2.0,
        },
        MachineType.MIXED: {
            "temperature": 1.0, "vibration": 1.0, "pressure": 0.8,
            "rpm": 0.8, "current": 1.0,
        },
    }

    def __init__(
        self,
        n_machines: int = 100,
        failure_horizon: int = 30,
        step_fault_fraction: float = 0.25,
        sensor_dropout_prob: float = 0.005,
        random_seed: int = 42,
    ) -> None:
        self.n_machines = n_machines
        self.failure_horizon = failure_horizon
        self.step_fault_fraction = step_fault_fraction
        self.sensor_dropout_prob = sensor_dropout_prob
        self.rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, output_dir: Optional[str | Path] = None) -> pd.DataFrame:
        """Generate full dataset and optionally persist to CSV."""
        dfs: list[pd.DataFrame] = []
        for machine_id in range(self.n_machines):
            cfg = self._make_config(machine_id)
            df = self._simulate_machine(cfg)
            dfs.append(df)

        dataset = pd.concat(dfs, ignore_index=True)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "sensor_data.csv"
            dataset.to_csv(out_path, index=False)
            logger.info(f"Synthetic dataset saved → {out_path} ({len(dataset):,} rows)")

        failure_rate = dataset["failure_imminent"].mean()
        type_dist = dataset.groupby("machine_type")["machine_id"].nunique()
        logger.info(
            f"Generated {len(dataset):,} timesteps across {self.n_machines} machines | "
            f"failure-imminent rate: {failure_rate:.2%}"
        )
        logger.info(f"Machine type distribution:\n{type_dist.to_string()}")
        return dataset

    # ------------------------------------------------------------------
    # Configuration builder
    # ------------------------------------------------------------------

    def _make_config(self, machine_id: int) -> MachineConfig:
        machine_type = self.rng.choice(list(MachineType))
        total = int(self.rng.integers(1500, 2500))
        has_step = float(self.rng.uniform(0, 1)) < self.step_fault_fraction

        return MachineConfig(
            machine_id=machine_id,
            machine_type=machine_type,
            total_timesteps=total,
            normal_phase_fraction=float(self.rng.uniform(0.55, 0.72)),
            degradation_phase_fraction=float(self.rng.uniform(0.18, 0.30)),
            failure_position_fraction=float(self.rng.uniform(0.0, 1.0)),
            temp_baseline=float(self.rng.uniform(65.0, 92.0)),
            vibration_baseline=float(self.rng.uniform(0.3, 0.9)),
            pressure_baseline=float(self.rng.uniform(4.5, 8.5)),
            rpm_baseline=float(self.rng.uniform(1400.0, 2300.0)),
            current_baseline=float(self.rng.uniform(9.0, 16.0)),
            sensor_sensitivity=self._TYPE_SENSITIVITY[machine_type],
            noise_std=float(self.rng.uniform(0.02, 0.07)),
            noise_tail=float(self.rng.uniform(0.5, 2.0)),
            degradation_severity=float(self.rng.uniform(0.7, 1.8)),
            has_step_fault=has_step,
            step_fault_fraction=float(self.rng.uniform(0.5, 0.9)),
            dropout_probability=self.sensor_dropout_prob,
        )

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------

    def _simulate_machine(self, cfg: MachineConfig) -> pd.DataFrame:
        T = cfg.total_timesteps
        normal_end = int(T * cfg.normal_phase_fraction)
        degrad_end = int(T * (cfg.normal_phase_fraction + cfg.degradation_phase_fraction))

        # Failure position: anywhere in the last (1 - normal - degradation) fraction
        remaining = T - degrad_end
        failure_offset = int(cfg.failure_position_fraction * max(remaining - 1, 0))
        failure_t = degrad_end + failure_offset

        degrad_len = max(degrad_end - normal_end, 1)
        step_fault_t = normal_end + int(cfg.step_fault_fraction * degrad_len) if cfg.has_step_fault else None

        baselines = {
            "temperature": cfg.temp_baseline,
            "vibration": cfg.vibration_baseline,
            "pressure": cfg.pressure_baseline,
            "rpm": cfg.rpm_baseline,
            "current": cfg.current_baseline,
        }

        # Degradation direction per sensor (+ increases, - decreases with wear)
        degrad_direction = {
            "temperature": +1.0,
            "vibration": +1.0,
            "pressure": -1.0,
            "rpm": -1.0,
            "current": +1.0,
        }
        degrad_magnitude = {
            "temperature": 0.20,
            "vibration": 1.50,
            "pressure": 0.15,
            "rpm": 0.12,
            "current": 0.25,
        }

        records = {col: np.zeros(T) for col in self.SENSOR_COLUMNS}
        # Oscillation state for non-monotonic degradation
        oscillation = self.rng.standard_normal(5) * 0.3

        for t in range(T):
            # ── Degradation progress ──────────────────────────────────
            if t < normal_end:
                progress = 0.0
                phase = "normal"
            elif t < degrad_end:
                raw_progress = (t - normal_end) / degrad_len
                # Non-monotonic: add sinusoidal oscillation (apparent recovery)
                oscillation += self.rng.standard_normal(5) * 0.05
                oscillation = np.clip(oscillation, -0.3, 0.3)
                progress = float(np.clip(raw_progress + oscillation[0] * 0.15, 0.0, 1.0))
                phase = "degradation"
            else:
                progress = 1.0 + (t - degrad_end) / max(T - degrad_end, 1) * 0.6
                phase = "failure"

            # ── Step fault injection ──────────────────────────────────
            step_multiplier = 1.0
            if step_fault_t is not None and t >= step_fault_t:
                # Sudden 20-40% jump in degradation at step fault time
                step_multiplier = float(1.0 + self.rng.uniform(0.2, 0.4))

            d = cfg.degradation_severity * progress * step_multiplier

            # ── Sensor noise: mix of Gaussian + Laplace heavy tail ────
            gaussian_noise = self.rng.standard_normal(5) * cfg.noise_std
            laplace_noise = self.rng.laplace(0, cfg.noise_tail * cfg.noise_std, 5)
            noise = 0.7 * gaussian_noise + 0.3 * laplace_noise

            # ── Temperature-current correlation (physical coupling) ────
            thermal_coupling = self.rng.normal(0, 0.03)

            for i, col in enumerate(self.SENSOR_COLUMNS):
                sensitivity = cfg.sensor_sensitivity[col]
                direction = degrad_direction[col]
                magnitude = degrad_magnitude[col]
                base = baselines[col]
                degradation_term = direction * magnitude * d * sensitivity

                val = base * (1 + degradation_term) + noise[i] * base

                # Temperature-current physical coupling
                if col == "current" and t > 0:
                    val += thermal_coupling * records["temperature"][t - 1] * 0.02
                if col == "temperature" and t > 0:
                    val += thermal_coupling * records["current"][t - 1] * 0.05

                # Physical minimum: sensors cannot read negative values
                records[col][t] = max(val, 1e-3)

        # ── Sensor dropout (forward-fill) ─────────────────────────────
        for col in self.SENSOR_COLUMNS:
            arr = records[col].copy()
            for t in range(1, T):
                if float(self.rng.uniform(0, 1)) < cfg.dropout_probability:
                    arr[t] = arr[t - 1]  # forward fill
            records[col] = arr

        # ── Labels ────────────────────────────────────────────────────
        failure_event = np.zeros(T, dtype=np.int8)
        failure_event[failure_t] = 1

        failure_imminent = np.zeros(T, dtype=np.int8)
        start = max(0, failure_t - self.failure_horizon)
        failure_imminent[start : failure_t + 1] = 1

        # Latent degradation progress (for analysis — not a training feature)
        latent_progress = np.zeros(T)
        for t in range(T):
            if t < normal_end:
                latent_progress[t] = 0.0
            elif t < degrad_end:
                latent_progress[t] = (t - normal_end) / degrad_len
            else:
                latent_progress[t] = 1.0

        df = pd.DataFrame(records)
        df.insert(0, "machine_id", cfg.machine_id)
        df.insert(1, "timestep", np.arange(T))
        df["machine_type"] = cfg.machine_type.value
        df["failure_event"] = failure_event
        df["failure_imminent"] = failure_imminent
        df["degradation_progress"] = latent_progress.round(4)

        return df
