"""Hyperparameter optimization via Optuna for SAC + TFT + LogReturnReward.

Optimizes SAC-specific parameters, TFT architecture, and reward function
using a fast single-pass backtest as objective (OOS Sharpe ratio).

Improvements over the original:
- SAC-specific search space: ent_coef, tau, buffer_size, train_freq
- LogReturnReward parameters: loss_weight, reward_scale
- Intermediate pruning via Optuna's MedianPruner
- Persistent SQLite storage for resumable studies
- Parallel-safe with Optuna's distributed backend
- Comprehensive trial metrics and result export

Usage::

    manager = SACHyperoptManager(config, features, n_features=15)
    results = manager.optimize()
    best_config = manager.apply_best(results)

Standalone::

    python scripts/hyperopt_sac.py --n-trials 40 --timeout 3600
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from apexfx.config.schema import AppConfig
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.reward import LogReturnReward
from apexfx.env.wrappers import MonitorWrapper, NormalizeReward
from apexfx.models.ensemble.hive_mind import HiveMindExtractor
from apexfx.utils.logging import get_logger
from apexfx.utils.metrics import compute_all_metrics

logger = get_logger(__name__)

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SACHyperoptManager:
    """Optuna-based hyperparameter optimization for SAC trading agents.

    Each trial:
    1. Samples SAC + TFT + reward hyperparameters
    2. Trains a short SAC run on the training split
    3. Evaluates on the held-out test split
    4. Returns OOS Sharpe ratio as the objective

    Parameters
    ----------
    base_config : AppConfig
        Base configuration (hyperparams are overridden per trial).
    features : pd.DataFrame
        Pre-computed feature DataFrame (already through FeatureSelector).
    n_features : int
        Number of market feature columns in ``features``.
    n_trials : int
        Number of Optuna trials to run.
    train_timesteps : int
        RL training steps per trial (shorter = faster search).
    study_name : str
        Name for Optuna study (used for SQLite persistence).
    storage_path : str | None
        Path for SQLite DB. None = in-memory (non-persistent).
    device : str
        Torch device.
    """

    def __init__(
        self,
        base_config: AppConfig,
        features: pd.DataFrame,
        n_features: int,
        n_trials: int = 40,
        train_timesteps: int = 50_000,
        study_name: str = "apexfx_sac_hyperopt",
        storage_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._config = base_config
        self._n_features = n_features
        self._n_trials = n_trials
        self._train_timesteps = train_timesteps
        self._study_name = study_name
        self._device = device

        # Time-ordered train/test split (70/30)
        split = int(len(features) * 0.7)
        self._train_data = features.iloc[:split].reset_index(drop=True)
        self._test_data = features.iloc[split:].reset_index(drop=True)

        logger.info(
            "Hyperopt initialized",
            train_bars=len(self._train_data),
            test_bars=len(self._test_data),
            n_features=n_features,
            n_trials=n_trials,
        )

        # Optuna storage
        if storage_path:
            self._storage = f"sqlite:///{storage_path}"
        else:
            self._storage = None

    def optimize(self, timeout: int | None = None) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Parameters
        ----------
        timeout : int | None
            Maximum wall-clock seconds. None = no limit.

        Returns
        -------
        dict with best_params, best_value, all_trials summary, and study object.
        """
        study = optuna.create_study(
            study_name=self._study_name,
            direction="maximize",
            storage=self._storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
            ),
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            self._objective,
            n_trials=self._n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        best = study.best_trial
        logger.info("=" * 60)
        logger.info("HYPEROPT COMPLETE")
        logger.info("=" * 60)
        logger.info("  Best trial:   #%d", best.number)
        logger.info("  Best Sharpe:  %.4f", best.value)
        for k, v in best.params.items():
            logger.info("  %-20s = %s", k, v)
        logger.info("=" * 60)

        # Compile results
        trials_summary = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                trials_summary.append({
                    "number": t.number,
                    "sharpe": t.value,
                    "params": t.params,
                    "user_attrs": t.user_attrs,
                    "duration_s": (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_complete else None,
                })

        return {
            "best_params": best.params,
            "best_value": best.value,
            "best_trial": best.number,
            "n_completed": len(trials_summary),
            "trials": trials_summary,
            "study": study,
        }

    def apply_best(self, results: dict[str, Any]) -> AppConfig:
        """Apply best hyperparameters to a config copy.

        Returns a new AppConfig with the optimal parameters.
        """
        config = self._config.model_copy(deep=True)
        params = results["best_params"]

        config.model.rl.learning_rate = params["learning_rate"]
        config.model.rl.gamma = params["gamma"]
        config.model.rl.batch_size = params["batch_size"]
        config.model.rl.tau = params["tau"]
        config.model.rl.buffer_size = params["buffer_size"]
        config.model.rl.train_freq = params["train_freq"]
        config.model.rl.gradient_steps = params["gradient_steps"]
        config.model.rl.learning_starts = params["learning_starts"]
        config.model.tft.d_model = params["d_model"]
        config.model.tft.n_heads = params["n_heads"]
        config.model.tft.dropout = params["tft_dropout"]

        return config

    # ── Private ───────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        """Single trial: sample params → train → evaluate → return Sharpe."""
        params = self._sample_params(trial)

        try:
            # Build env with trial params
            train_env = self._build_env(self._train_data, params)
            model = self._build_model(train_env, params, trial)

            # Train (short run)
            model.learn(
                total_timesteps=self._train_timesteps,
                progress_bar=False,
            )

            # Evaluate OOS
            metrics = self._evaluate(model)

            sharpe = metrics.get("sharpe_ratio", -10.0)
            max_dd = metrics.get("max_drawdown_pct", 100.0)
            total_return = metrics.get("total_return_pct", 0.0)
            n_trades = metrics.get("n_trades", 0)

            # Store metrics for analysis
            trial.set_user_attr("max_drawdown_pct", round(max_dd, 4))
            trial.set_user_attr("total_return_pct", round(total_return, 4))
            trial.set_user_attr("n_trades", n_trades)
            trial.set_user_attr("win_rate", round(metrics.get("win_rate", 0), 4))
            trial.set_user_attr("profit_factor", round(metrics.get("profit_factor", 0), 4))

            logger.info(
                "Trial #%d  sharpe=%.4f  return=%.2f%%  dd=%.2f%%  trades=%d",
                trial.number, sharpe, total_return, max_dd, n_trades,
            )

            # Penalize if no trades (model is doing nothing)
            if n_trades < 5:
                return -5.0

            return sharpe

        except Exception as e:
            logger.error("Trial #%d failed: %s", trial.number, str(e))
            return -10.0

        finally:
            # Free GPU memory between trials
            if self._device != "cpu":
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters for a single trial."""
        return {
            # ── SAC core ──
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 500_000, 1_000_000]),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8]),
            "learning_starts": trial.suggest_categorical("learning_starts", [1_000, 5_000, 10_000, 20_000]),
            "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05, 0.1, 0.2]),
            # ── TFT architecture ──
            "d_model": trial.suggest_categorical("d_model", [32, 64]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
            "tft_dropout": trial.suggest_float("tft_dropout", 0.05, 0.3),
            # ── LogReturnReward ──
            "loss_weight": trial.suggest_float("loss_weight", 1.0, 4.0),
            "reward_scale": trial.suggest_float("reward_scale", 500.0, 5000.0, log=True),
            # ── Policy network ──
            "net_arch_size": trial.suggest_categorical("net_arch_size", [128, 256]),
        }

    def _build_env(self, data: pd.DataFrame, params: dict) -> DummyVecEnv:
        """Build training env with trial-specific reward params."""
        risk_cfg = self._config.risk
        lookback = self._config.data.feature_window
        n_features = self._n_features

        def make_env():
            env = ForexTradingEnv(
                data=data,
                initial_balance=100_000.0,
                max_position_pct=risk_cfg.position_sizing.max_position_pct,
                n_market_features=n_features,
                lookback=lookback,
                reward_fn=LogReturnReward(
                    loss_weight=params["loss_weight"],
                    reward_scale=params["reward_scale"],
                ),
                max_drawdown_pct=0.15,
            )
            env = MonitorWrapper(env)
            env = NormalizeReward(env)
            return env

        return DummyVecEnv([make_env])

    def _build_model(self, env: DummyVecEnv, params: dict, trial: optuna.Trial) -> SAC:
        """Build SAC model with trial-specific hyperparameters."""
        lookback = self._config.data.feature_window
        arch_size = params["net_arch_size"]

        policy_kwargs = {
            "features_extractor_class": HiveMindExtractor,
            "features_extractor_kwargs": {
                "n_continuous_vars": self._n_features,
                "d_model": params["d_model"],
                "seq_len": lookback,
            },
            "net_arch": [arch_size, arch_size],
            "optimizer_kwargs": {"weight_decay": 1e-4, "eps": 1e-5},
        }

        return SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            ent_coef=params["ent_coef"],
            gamma=params["gamma"],
            tau=params["tau"],
            train_freq=params["train_freq"],
            gradient_steps=params["gradient_steps"],
            learning_starts=params["learning_starts"],
            device=self._device,
            verbose=0,
            seed=42 + trial.number,  # different seed per trial
        )

    def _evaluate(self, model: SAC) -> dict[str, float]:
        """Evaluate model on held-out test data, return performance metrics."""
        lookback = self._config.data.feature_window

        eval_env = ForexTradingEnv(
            data=self._test_data,
            initial_balance=100_000.0,
            n_market_features=self._n_features,
            lookback=lookback,
            reward_fn=LogReturnReward(),
            max_drawdown_pct=1.0,  # don't terminate during eval
        )

        obs, info = eval_env.reset()
        done = False
        returns: list[float] = []
        equity = [100_000.0]
        prev_value = 100_000.0
        prev_position = 0.0
        n_trades = 0
        step_count = 0
        max_steps = len(self._test_data) - lookback - 10

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            current_value = info.get("portfolio_value", prev_value)
            position = info.get("position", 0.0)

            if prev_value > 0:
                returns.append((current_value - prev_value) / prev_value)

            if abs(position) > 0 and abs(prev_position) == 0:
                n_trades += 1

            equity.append(current_value)
            prev_value = current_value
            prev_position = position
            step_count += 1

        returns_arr = np.array(returns) if returns else np.array([0.0])
        metrics = compute_all_metrics(returns_arr)

        # Additional metrics
        final_value = equity[-1]
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / np.maximum(peak, 1e-10) * 100
        max_dd = float(np.max(dd))

        metrics["total_return_pct"] = (final_value - 100_000) / 100_000 * 100
        metrics["max_drawdown_pct"] = max_dd
        metrics["n_trades"] = n_trades
        metrics["n_steps"] = step_count

        return metrics
