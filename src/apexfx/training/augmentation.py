"""Advanced time-series data augmentation for RL training.

Markets are fractal and self-similar across timescales. Augmentation exploits
this by generating plausible variations of the training data, improving
generalisation without overfitting to specific historical patterns.

4 methods, each with independent probability:

1. **TimeWarp** — Stretch/compress temporal segments via cubic spline.
   Makes the agent robust to variable-speed market moves.

2. **MagnitudeWarp** — Multiplicative smooth random curve.
   Simulates feature scaling drift between training and live.

3. **WindowSlice** — Random crop of lookback window, zero-padded.
   Forces the agent to make decisions with incomplete history.

4. **Mixup** — Blend current obs with previous obs (λ~Beta(α,α)).
   Smooths the observation manifold, reducing overfitting to specific patterns.

Rules:
- `position_state` is NEVER augmented (it's real portfolio state)
- Sequential keys (`market_features`, `time_features`): all 4 methods
- Point keys (`trend_features`, `reversion_features`, `regime_features`): MagnitudeWarp only
- Each method is applied independently with configurable probability
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from scipy.interpolate import CubicSpline

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Keys that contain sequential (lookback × features) data
SEQUENTIAL_KEYS = {"market_features", "time_features"}
# Keys that contain point features (1D, no time dimension)
POINT_KEYS = {"trend_features", "reversion_features", "regime_features"}
# Keys that are NEVER augmented
PROTECTED_KEYS = {"position_state"}


class AugmentedObsWrapper(gym.Wrapper):
    """Gymnasium wrapper that applies stochastic augmentation to Dict observations.

    Parameters
    ----------
    env : gym.Env
        Base environment with Dict observation space.
    config : AugmentationConfig
        Configuration with per-method probabilities and parameters.
    """

    def __init__(self, env: gym.Env, config) -> None:
        super().__init__(env)
        self._cfg = config
        self._rng = np.random.default_rng()
        self._prev_obs: dict[str, np.ndarray] | None = None

        # Infer lookback from market_features shape
        if hasattr(env.observation_space, "spaces") and "market_features" in env.observation_space.spaces:
            mf_shape = env.observation_space["market_features"].shape
            self._lookback = mf_shape[0] if len(mf_shape) > 1 else None
        else:
            self._lookback = None

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._augment(obs)
        self._prev_obs = {k: v.copy() for k, v in obs.items()}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = {k: v.copy() for k, v in obs.items()}
        return obs, info

    def _augment(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply stochastic augmentation pipeline."""
        result = {}
        for key, val in obs.items():
            if key in PROTECTED_KEYS:
                result[key] = val
                continue

            aug_val = val.copy()

            if key in SEQUENTIAL_KEYS:
                # All 4 methods for sequential data
                if self._rng.random() < self._cfg.time_warp_prob:
                    aug_val = self._time_warp(aug_val, self._cfg.time_warp_sigma)
                if self._rng.random() < self._cfg.magnitude_warp_prob:
                    aug_val = self._magnitude_warp(aug_val, self._cfg.magnitude_warp_sigma)
                if self._rng.random() < self._cfg.window_slice_prob:
                    aug_val = self._window_slice(aug_val, self._cfg.window_slice_ratio)
            elif key in POINT_KEYS:
                # Only magnitude warp for point features
                if self._rng.random() < self._cfg.magnitude_warp_prob:
                    aug_val = self._magnitude_warp(aug_val, self._cfg.magnitude_warp_sigma)
            else:
                # Unknown key — apply magnitude warp conservatively
                if self._rng.random() < self._cfg.magnitude_warp_prob:
                    aug_val = self._magnitude_warp(aug_val, self._cfg.magnitude_warp_sigma * 0.5)

            result[key] = aug_val

        # Mixup: blend with previous observation
        if self._prev_obs is not None and self._rng.random() < self._cfg.mixup_prob:
            result = self._mixup(result, self._prev_obs, self._cfg.mixup_alpha)

        return result

    def _time_warp(self, arr: np.ndarray, sigma: float) -> np.ndarray:
        """Stretch/compress temporal segments via cubic spline warping.

        Reshapes flat array → (lookback, n_features), warps time axis,
        then flattens back.
        """
        if arr.ndim != 1 or self._lookback is None:
            return arr

        n_total = arr.shape[0]
        lookback = self._lookback
        if n_total < lookback or lookback < 4:
            return arr

        n_features = n_total // lookback
        if n_features * lookback != n_total:
            return arr

        data = arr.reshape(lookback, n_features)
        orig_steps = np.arange(lookback)

        # Generate random warp knots (4-6 control points)
        n_knots = min(4, lookback // 2)
        knot_indices = np.linspace(0, lookback - 1, n_knots + 2)
        warp_offsets = self._rng.normal(0, sigma, n_knots + 2)
        warp_offsets[0] = 0.0   # Fix start
        warp_offsets[-1] = 0.0  # Fix end
        warped_knots = knot_indices + warp_offsets

        # Ensure monotonicity
        warped_knots = np.sort(warped_knots)
        warped_knots = np.clip(warped_knots, 0, lookback - 1)

        # Build cubic spline mapping: warped_knots → orig indices
        try:
            cs = CubicSpline(warped_knots, knot_indices, bc_type="clamped")
            new_steps = cs(orig_steps)
            new_steps = np.clip(new_steps, 0, lookback - 1)
        except Exception:
            return arr

        # Interpolate each feature along warped time axis
        result = np.zeros_like(data)
        for f in range(n_features):
            result[:, f] = np.interp(new_steps, orig_steps, data[:, f])

        return result.flatten().astype(arr.dtype)

    def _magnitude_warp(self, arr: np.ndarray, sigma: float) -> np.ndarray:
        """Multiplicative smooth random curve deformation.

        Generates a smooth random curve via cubic spline and multiplies
        the features by it. This simulates gradual feature scaling changes.
        """
        if arr.size == 0:
            return arr

        if arr.ndim == 1 and self._lookback is not None:
            n_total = arr.shape[0]
            lookback = self._lookback
            n_features = n_total // lookback if lookback > 0 else 0

            if n_features > 0 and n_features * lookback == n_total:
                # Sequential: generate smooth curve along time axis
                data = arr.reshape(lookback, n_features)
                n_knots = min(4, lookback // 2)
                if n_knots < 2:
                    return arr

                knot_x = np.linspace(0, lookback - 1, n_knots)
                knot_y = self._rng.normal(1.0, sigma, (n_knots, n_features))

                result = np.zeros_like(data)
                x_full = np.arange(lookback)
                for f in range(n_features):
                    try:
                        cs = CubicSpline(knot_x, knot_y[:, f])
                        result[:, f] = data[:, f] * cs(x_full)
                    except Exception:
                        result[:, f] = data[:, f]

                return result.flatten().astype(arr.dtype)

        # Point features or fallback: simple multiplicative noise
        scale = self._rng.normal(1.0, sigma, arr.shape).astype(arr.dtype)
        return arr * scale

    def _window_slice(self, arr: np.ndarray, ratio: float) -> np.ndarray:
        """Random crop of lookback window, zero-padded to original length.

        Keeps `ratio` fraction of the temporal points, starting from a
        random offset. The remaining positions are zero-padded at the front.
        """
        if arr.ndim != 1 or self._lookback is None:
            return arr

        n_total = arr.shape[0]
        lookback = self._lookback
        n_features = n_total // lookback if lookback > 0 else 0
        if n_features == 0 or n_features * lookback != n_total:
            return arr

        keep_len = max(2, int(lookback * ratio))
        max_start = lookback - keep_len
        if max_start <= 0:
            return arr

        start = self._rng.integers(0, max_start + 1)
        data = arr.reshape(lookback, n_features)

        result = np.zeros_like(data)
        # Place the sliced window at the end (most recent data)
        result[lookback - keep_len:] = data[start:start + keep_len]

        return result.flatten().astype(arr.dtype)

    def _mixup(
        self,
        obs: dict[str, np.ndarray],
        prev_obs: dict[str, np.ndarray],
        alpha: float,
    ) -> dict[str, np.ndarray]:
        """Blend current obs with previous obs using Beta-distributed weight."""
        lam = self._rng.beta(alpha, alpha)
        result = {}
        for key, val in obs.items():
            if key in PROTECTED_KEYS:
                result[key] = val
            elif key in prev_obs and prev_obs[key].shape == val.shape:
                result[key] = (lam * val + (1 - lam) * prev_obs[key]).astype(val.dtype)
            else:
                result[key] = val
        return result
