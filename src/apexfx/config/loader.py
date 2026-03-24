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


# Config files that MUST exist in production to avoid silent defaults
_CRITICAL_CONFIGS = {"risk.yaml", "execution.yaml", "symbols.yaml"}


def load_yaml(path: str | Path, required: bool = False) -> dict:
    """Load a single YAML file with env-var interpolation.

    Args:
        path: Path to YAML file.
        required: If True, raise FileNotFoundError when file is missing.
    """
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required config file missing: {path}. "
                f"Copy from configs/ templates or create it."
            )
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _interpolate_env_vars(raw)


def load_config(config_dir: str | Path = "configs") -> AppConfig:
    """Load all YAML config files from the directory and merge them
    into a single validated AppConfig.

    Respects APEXFX_ENV environment variable to load environment-specific
    overrides (e.g., configs/production.yaml, configs/staging.yaml).

    In production mode, critical config files (risk, execution, symbols)
    must exist — missing files cause a hard error instead of silent defaults.
    """
    config_dir = Path(config_dir)
    env = os.environ.get("APEXFX_ENV", "").lower()
    is_production = env == "production"

    base = load_yaml(config_dir / "base.yaml")
    symbols = load_yaml(config_dir / "symbols.yaml", required=is_production)
    data = load_yaml(config_dir / "data.yaml")
    model = load_yaml(config_dir / "model.yaml")
    training = load_yaml(config_dir / "training.yaml")
    risk = load_yaml(config_dir / "risk.yaml", required=is_production)
    execution = load_yaml(config_dir / "execution.yaml", required=is_production)
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

    # Apply environment-specific overrides (production.yaml / staging.yaml)
    if env:
        env_overrides = load_yaml(config_dir / f"{env}.yaml")
        if env_overrides:
            merged = _deep_merge(merged, env_overrides)

    return AppConfig.model_validate(merged)
