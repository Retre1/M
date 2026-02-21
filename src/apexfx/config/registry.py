"""Global configuration registry — thread-safe singleton."""

from __future__ import annotations

import threading
from pathlib import Path

from apexfx.config.loader import load_config
from apexfx.config.schema import AppConfig

_lock = threading.Lock()
_config: AppConfig | None = None


def init_config(config_dir: str | Path = "configs") -> AppConfig:
    """Initialize the global config. Call once at startup."""
    global _config
    with _lock:
        _config = load_config(config_dir)
    return _config


def get_config() -> AppConfig:
    """Get the global config. Raises if not initialized."""
    if _config is None:
        raise RuntimeError(
            "Config not initialized. Call init_config() at startup."
        )
    return _config


# Convenience alias
cfg = get_config
