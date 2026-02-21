"""Hyperparameter optimization via Optuna with walk-forward OOS Sharpe as objective."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd

from apexfx.config.schema import AppConfig
from apexfx.training.walk_forward import WalkForwardValidator
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class HyperoptManager:
    """
    Optuna-based hyperparameter search.
    Objective: walk-forward out-of-sample Sharpe ratio.
    """

    def __init__(
        self,
        base_config: AppConfig,
        data: pd.DataFrame,
        n_trials: int = 50,
        study_name: str = "apexfx_hyperopt",
    ) -> None:
        self._base_config = base_config
        self._data = data
        self._n_trials = n_trials
        self._study_name = study_name

    def optimize(self) -> dict[str, Any]:
        """Run hyperparameter optimization and return best params."""
        study = optuna.create_study(
            study_name=self._study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        study.optimize(self._objective, n_trials=self._n_trials, show_progress_bar=True)

        logger.info(
            "Hyperopt complete",
            best_value=study.best_value,
            best_params=study.best_params,
        )

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
        }

    def _objective(self, trial: optuna.Trial) -> float:
        """Single trial objective: returns OOS Sharpe ratio."""
        config = self._build_config(trial)

        try:
            validator = WalkForwardValidator(config, self._data)
            results = validator.run()

            sharpe = results.aggregate_metrics.get("sharpe_ratio", -10.0)

            trial.set_user_attr("max_drawdown", results.aggregate_metrics.get("max_drawdown", 1.0))
            trial.set_user_attr("total_return", results.aggregate_metrics.get("total_return", 0.0))
            trial.set_user_attr("n_folds", len(results.folds))

            logger.info(
                "Trial complete",
                trial=trial.number,
                sharpe=round(sharpe, 4),
            )

            return sharpe

        except Exception as e:
            logger.error("Trial failed", trial=trial.number, error=str(e))
            return -10.0

    def _build_config(self, trial: optuna.Trial) -> AppConfig:
        """Build config with Optuna-suggested hyperparameters."""
        config = self._base_config.model_copy(deep=True)

        # RL hyperparameters
        config.model.rl.learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )
        config.model.rl.gamma = trial.suggest_float("gamma", 0.95, 0.999)
        config.model.rl.batch_size = trial.suggest_categorical(
            "batch_size", [128, 256, 512]
        )

        # TFT hyperparameters
        config.model.tft.d_model = trial.suggest_categorical(
            "d_model", [32, 64, 128]
        )
        config.model.tft.n_heads = trial.suggest_categorical(
            "n_heads", [2, 4, 8]
        )
        config.model.tft.dropout = trial.suggest_float("dropout", 0.05, 0.3)

        # Reward function
        lambda_dd = trial.suggest_float("lambda_dd", 0.5, 5.0)
        eta = trial.suggest_float("reward_eta", 0.005, 0.05, log=True)

        # Risk parameters
        config.risk.daily_var_limit = trial.suggest_float("var_limit", 0.01, 0.05)
        config.risk.position_sizing.max_position_pct = trial.suggest_float(
            "max_position_pct", 0.05, 0.20
        )

        # Reduce training timesteps for faster trials
        for stage in config.training.curriculum.stages:
            stage.total_timesteps = stage.total_timesteps // 5

        return config
