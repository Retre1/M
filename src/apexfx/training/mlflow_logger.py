"""MLflow experiment tracking integration for ApexFX Quantum.

Usage
-----
MLflow is **opt-in**: set ``base.mlflow.enabled: true`` in ``configs/base.yaml``
or set ``MLFLOW_ENABLED=true`` in the environment.  When disabled (default),
every call is a no-op so there is zero performance overhead.

Architecture
------------
* :class:`MLflowLogger` — thin wrapper around the MLflow client.
  One logger per training run.  Thread-safe (MLflow client is thread-safe).
* :class:`MLflowCallback` — SB3 BaseCallback that pushes step-level metrics
  (episode return, drawdown, portfolio value) to the active MLflow run.

Tracked artefacts
-----------------
Per training run:
  * All config fields (flat key=value params)
  * Per-stage curriculum metrics
  * Episode-level: return, length, drawdown, portfolio value, total trades
  * Periodic: mean / std return over last 10 episodes
  * Walk-forward fold results (sharpe, max_dd, total_return per fold)
  * Model checkpoint paths (when log_artifacts=True)

To view:
    $ mlflow ui --backend-store-uri ./mlruns
    # open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from apexfx.config.schema import AppConfig
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def _flatten(d: dict, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dict into dot-separated keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)
        else:
            out[key] = v
    return out


class MLflowLogger:
    """Manages a single MLflow run for one training session.

    Parameters
    ----------
    config:
        Full application config.  MLflow settings are read from
        ``config.base.mlflow``.
    run_name:
        Optional override for the MLflow run name (shown in the UI).
    tags:
        Optional dict of MLflow tags to attach to the run.
    """

    def __init__(
        self,
        config: AppConfig,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._cfg = config.base.mlflow
        self._enabled = self._cfg.enabled or os.environ.get("MLFLOW_ENABLED", "").lower() == "true"
        self._run = None
        self._mlflow = None

        if not self._enabled:
            return

        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            logger.warning(
                "mlflow not installed — tracking disabled. "
                "Install with: pip install mlflow"
            )
            self._enabled = False
            return

        mlflow.set_tracking_uri(self._cfg.tracking_uri)
        mlflow.set_experiment(self._cfg.experiment_name)

        self._run = mlflow.start_run(run_name=run_name, tags=tags or {})
        logger.info(
            "MLflow run started",
            experiment=self._cfg.experiment_name,
            run_id=self._run.info.run_id,
            tracking_uri=self._cfg.tracking_uri,
        )

        # Log all config as MLflow params (flat key=value)
        self._log_config(config)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if not self._enabled or self._mlflow is None:
            return
        try:
            self._mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.debug("MLflow log_metric failed", key=key, error=str(e))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._enabled or self._mlflow is None:
            return
        try:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if v is not None},
                step=step,
            )
        except Exception as e:
            logger.debug("MLflow log_metrics failed", error=str(e))

    def log_param(self, key: str, value: Any) -> None:
        if not self._enabled or self._mlflow is None:
            return
        try:
            self._mlflow.log_param(key, value)
        except Exception as e:
            logger.debug("MLflow log_param failed", key=key, error=str(e))

    def log_artifact(self, path: str | Path) -> None:
        if not self._enabled or self._mlflow is None or not self._cfg.log_artifacts:
            return
        try:
            self._mlflow.log_artifact(str(path))
            logger.debug("MLflow artifact logged", path=str(path))
        except Exception as e:
            logger.debug("MLflow log_artifact failed", path=str(path), error=str(e))

    def set_tag(self, key: str, value: str) -> None:
        if not self._enabled or self._mlflow is None:
            return
        try:
            self._mlflow.set_tag(key, value)
        except Exception as e:
            logger.debug("MLflow set_tag failed", key=key, error=str(e))

    def end_run(self, status: str = "FINISHED") -> None:
        """Close the MLflow run.  Call this after training completes."""
        if not self._enabled or self._mlflow is None:
            return
        try:
            self._mlflow.end_run(status=status)
            logger.info("MLflow run ended", status=status)
        except Exception as e:
            logger.debug("MLflow end_run failed", error=str(e))

    # ------------------------------------------------------------------
    # Higher-level helpers used by Trainer
    # ------------------------------------------------------------------

    def log_curriculum_stage_start(self, stage_idx: int, stage_name: str, timesteps: int) -> None:
        self.set_tag(f"stage_{stage_idx}_name", stage_name)
        self.log_param(f"stage_{stage_idx}_timesteps", timesteps)

    def log_curriculum_stage_end(
        self, stage_idx: int, metrics: dict[str, float], step: int
    ) -> None:
        prefixed = {f"stage_{stage_idx}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed, step=step)

    def log_walk_forward_fold(
        self, fold_idx: int, metrics: dict[str, float], step: int
    ) -> None:
        prefixed = {f"wf_fold_{fold_idx}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed, step=step)

    def log_walk_forward_summary(self, aggregate_metrics: dict[str, float]) -> None:
        prefixed = {f"wf_aggregate/{k}": v for k, v in aggregate_metrics.items()}
        self.log_metrics(prefixed)

    def log_backtest_results(self, results: dict[str, Any]) -> None:
        flat = {f"backtest/{k}": v for k, v in results.items()
                if isinstance(v, (int, float))}
        self.log_metrics(flat)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_config(self, config: AppConfig) -> None:
        """Flatten the full config and log as MLflow params."""
        try:
            raw = config.model_dump()
            flat = _flatten(raw)
            # MLflow param values must be strings ≤ 500 chars
            params = {k: str(v)[:500] for k, v in flat.items()}
            # MLflow allows max 100 params per log_params call — batch them
            items = list(params.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))
        except Exception as e:
            logger.debug("MLflow config logging failed", error=str(e))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None
