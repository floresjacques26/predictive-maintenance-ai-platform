"""YAML configuration loader with dot-access support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict subclass that supports attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Safely access nested keys: cfg.get_nested('training', 'lr')."""
        current: Any = self
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
        return current


def load_config(*paths: str | Path) -> Config:
    """Load and merge one or more YAML config files.

    Later files override earlier ones (shallow merge at top level).

    Args:
        *paths: Paths to YAML config files.

    Returns:
        Merged Config object.
    """
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(merged, data)
    return Config(merged)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
