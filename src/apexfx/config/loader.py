"""YAML configuration loader with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from apexfx.config.schema import AppConfig


def _interpolate_env_vars(data: dict | list | str) -> dict | list | str:
    """Recursively replace ${ENV_VAR} patterns with environment variable values."""
    if isinstance(data, dict):
        return {k: _interpolate_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate_env_vars(item) for item in data]
    if isinstance(data, str):
        pattern = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            value = os.environ.get(var_name)
            if value is not None:
                return value
            if default is not None:
                return default
            return match.group(0)
        return pattern.sub(replace, data)
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict:
    """Load a single YAML file with env-var interpolation."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _interpolate_env_vars(raw)


def load_config(config_dir: str | Path = "configs") -> AppConfig:
    """
    Load all YAML config files from the directory and merge them
    into a single validated AppConfig.
    """
    config_dir = Path(config_dir)

    base = load_yaml(config_dir / "base.yaml")
    symbols = load_yaml(config_dir / "symbols.yaml")
    data = load_yaml(config_dir / "data.yaml")
    model = load_yaml(config_dir / "model.yaml")
    training = load_yaml(config_dir / "training.yaml")
    risk = load_yaml(config_dir / "risk.yaml")
    execution = load_yaml(config_dir / "execution.yaml")
    dashboard = load_yaml(config_dir / "dashboard.yaml")

    merged = {
        "base": base,
        "symbols": symbols,
        "data": data,
        "model": model,
        "training": training,
        "risk": risk,
        "execution": execution,
        "dashboard": dashboard,
    }

    return AppConfig.model_validate(merged)
