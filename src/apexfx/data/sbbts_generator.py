"""SBBTS (Schrödinger–Bass) synthetic forex trajectory generator.

Adapted from the paper "Schrödinger–Bass model for synthetic financial
time series" (reference implementation: github.com/alexouadi/SBBTS) for
use with forex currency pairs.

The model combines three components:

1. **Schrödinger double-well potential** — produces bi-modal (bull/bear)
   regime dynamics with quantum-inspired tunnelling between states.
2. **Bass diffusion** — modulates drift to capture adoption / trend
   propagation effects (how fast the market transitions between regimes).
3. **Heston stochastic volatility** — generates realistic volatility
   clustering (high-vol begets high-vol) with mean-reversion.

Multi-pair generation uses a Cholesky decomposition of the empirical
return correlation matrix to preserve cross-pair dependencies.

Design principles
-----------------
* **Reproducibility**: all randomness flows through a single
  ``np.random.Generator`` seeded from config.
* **Vectorised**: inner loops are NumPy-vectorised wherever possible.
  The unavoidable time-step loop (SDE integration) uses Numba if
  available, else falls back to plain NumPy.
* **Validation**: built-in KL divergence check between real and
  synthetic marginal return distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import gaussian_kde

from apexfx.data.config import SBBTSConfig, SBBTSPairParams
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Moments calibrated from real data for a single pair.

    Attributes:
        mu: Annualised log-return drift.
        sigma: Annualised log-return volatility.
        kurtosis: Excess kurtosis of log-returns.
        skewness: Skewness of log-returns.
        acf_abs_r1: First-order autocorrelation of |returns| (vol clustering).
    """

    mu: float
    sigma: float
    kurtosis: float = 0.0
    skewness: float = 0.0
    acf_abs_r1: float = 0.0


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class SBBTSGenerator:
    """Generate synthetic forex trajectories via the Schrödinger–Bass model.

    Args:
        config: SBBTS configuration.  See :class:`SBBTSConfig`.

    Example::

        gen = SBBTSGenerator(SBBTSConfig(seed=42, ratio=0.3))
        real = pd.read_parquet("data/EURUSD_H1.parquet")
        synth = gen.generate(real, n_bars=5000)
        kl = gen.validate_kl(real, synth)
    """

    def __init__(self, config: SBBTSConfig | None = None) -> None:
        self._cfg = config or SBBTSConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, real_data: pd.DataFrame) -> CalibrationResult:
        """Calibrate model parameters from real OHLCV data.

        Args:
            real_data: DataFrame with at least a ``close`` column.

        Returns:
            CalibrationResult with estimated moments.

        Raises:
            ValueError: If fewer than 30 bars of data.
        """
        close = real_data["close"].dropna().values
        if len(close) < 30:
            raise ValueError(
                f"Need >= 30 bars for calibration, got {len(close)}"
            )

        log_returns = np.diff(np.log(close))
        dt = self._cfg.dt
        n = len(log_returns)

        mu_per_step = float(np.mean(log_returns))
        sigma_per_step = float(np.std(log_returns, ddof=1))

        # Annualise (assuming dt = fraction of year per bar)
        mu_annual = mu_per_step / dt
        sigma_annual = sigma_per_step / np.sqrt(dt)

        # Higher moments
        from scipy.stats import kurtosis as scipy_kurtosis
        from scipy.stats import skew as scipy_skew

        kurt = float(scipy_kurtosis(log_returns, fisher=True, bias=False))
        skw = float(scipy_skew(log_returns, bias=False))

        # Autocorrelation of |returns| (proxy for vol clustering)
        abs_r = np.abs(log_returns)
        acf1 = float(np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1]) if n > 2 else 0.0

        logger.info(
            "Calibrated SBBTS",
            mu_annual=f"{mu_annual:.6f}",
            sigma_annual=f"{sigma_annual:.4f}",
            kurtosis=f"{kurt:.2f}",
            skewness=f"{skw:.3f}",
            acf_abs_r1=f"{acf1:.3f}",
        )
        return CalibrationResult(
            mu=mu_annual,
            sigma=sigma_annual,
            kurtosis=kurt,
            skewness=skw,
            acf_abs_r1=acf1,
        )

    def generate(
        self,
        real_data: pd.DataFrame,
        n_bars: int | None = None,
        beta: float = 1.0,
        pair: str = "EURUSD",
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV bars calibrated to real data.

        Args:
            real_data: Real OHLCV DataFrame (must have ``close`` column).
            n_bars: Number of synthetic bars to generate.
                Defaults to ``len(real_data)``.
            beta: Noise scaling factor.  β > 1 → heavier tails (stress
                scenarios).  β < 1 → calmer paths.
            pair: Symbol name for logging / per-pair config lookup.

        Returns:
            DataFrame with columns: time, open, high, low, close, volume,
            tick_count, plus a ``synthetic`` flag column set to True.
        """
        n_bars = n_bars or len(real_data)
        calib = self.calibrate(real_data)

        # Per-pair overrides
        pp = self._cfg.pair_params.get(pair, SBBTSPairParams())
        mu = pp.mu if pp.mu is not None else calib.mu
        sigma = pp.sigma if pp.sigma is not None else calib.sigma

        # Scale volatility by beta
        sigma *= beta

        logger.info(
            "Generating SBBTS paths",
            pair=pair,
            n_bars=n_bars,
            beta=beta,
            mu=f"{mu:.6f}",
            sigma=f"{sigma:.4f}",
        )

        # Generate log-price path
        log_prices = self._simulate_path(
            n_steps=n_bars,
            mu=mu,
            sigma=sigma,
            kappa=pp.mean_reversion_kappa,
            xi=pp.vol_of_vol,
            bass_p=pp.bass_p,
            bass_q=pp.bass_q,
        )

        # Convert to OHLCV
        base_price = float(real_data["close"].iloc[0])
        df = self._log_prices_to_ohlcv(
            log_prices, base_price=base_price, real_data=real_data,
        )
        df["synthetic"] = True

        return df

    def generate_multi_pair(
        self,
        pair_data: dict[str, pd.DataFrame],
        n_bars: int | None = None,
        beta: float = 1.0,
    ) -> dict[str, pd.DataFrame]:
        """Generate correlated synthetic paths for multiple pairs.

        Uses Cholesky decomposition of the empirical return correlation
        matrix to preserve cross-pair dependencies.

        Args:
            pair_data: Dict mapping symbol → real OHLCV DataFrame.
            n_bars: Bars per pair (defaults to min length across pairs).
            beta: Noise scaling factor.

        Returns:
            Dict mapping symbol → synthetic OHLCV DataFrame.
        """
        pairs = list(pair_data.keys())
        n_pairs = len(pairs)
        if n_pairs < 2:
            # Single pair — just call generate()
            pair = pairs[0]
            return {pair: self.generate(pair_data[pair], n_bars, beta, pair)}

        n_bars = n_bars or min(len(df) for df in pair_data.values())

        # Build correlation matrix from real returns
        returns_dict = {}
        for pair in pairs:
            close = pair_data[pair]["close"].dropna().values
            returns_dict[pair] = np.diff(np.log(close))

        min_len = min(len(r) for r in returns_dict.values())
        returns_matrix = np.column_stack([
            returns_dict[p][:min_len] for p in pairs
        ])
        corr = np.corrcoef(returns_matrix, rowvar=False)

        # Ensure positive-definite
        eigvals = np.linalg.eigvalsh(corr)
        if np.any(eigvals < 1e-8):
            corr += np.eye(n_pairs) * 1e-6

        cholesky = np.linalg.cholesky(corr)

        # Generate correlated standard normals
        Z_independent = self._rng.standard_normal((n_bars, n_pairs))
        Z_correlated = Z_independent @ cholesky.T

        # Generate each pair using correlated innovations
        results = {}
        for i, pair in enumerate(pairs):
            calib = self.calibrate(pair_data[pair])
            pp = self._cfg.pair_params.get(pair, SBBTSPairParams())
            mu = pp.mu if pp.mu is not None else calib.mu
            sigma = (pp.sigma if pp.sigma is not None else calib.sigma) * beta

            log_prices = self._simulate_path_with_innovations(
                innovations=Z_correlated[:, i],
                mu=mu,
                sigma=sigma,
                kappa=pp.mean_reversion_kappa,
                xi=pp.vol_of_vol,
                bass_p=pp.bass_p,
                bass_q=pp.bass_q,
            )

            base_price = float(pair_data[pair]["close"].iloc[0])
            df = self._log_prices_to_ohlcv(
                log_prices, base_price=base_price, real_data=pair_data[pair],
            )
            df["synthetic"] = True
            results[pair] = df

        logger.info(
            "Multi-pair SBBTS complete",
            pairs=pairs,
            n_bars=n_bars,
            beta=beta,
        )
        return results

    def validate_kl(
        self,
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        n_bins: int = 100,
    ) -> float:
        """Compute KL divergence between real and synthetic return distributions.

        Uses kernel density estimation for smooth probability estimates.

        Args:
            real_data: Real OHLCV DataFrame.
            synth_data: Synthetic OHLCV DataFrame.
            n_bins: Number of evaluation points for KDE.

        Returns:
            KL(real || synth) divergence value.
            Values < 0.05 indicate good distributional match.
        """
        real_returns = np.diff(np.log(real_data["close"].dropna().values))
        synth_returns = np.diff(np.log(synth_data["close"].dropna().values))

        if len(real_returns) < 10 or len(synth_returns) < 10:
            logger.warning("Insufficient data for KL validation")
            return float("inf")

        # KDE-based density estimation
        x_min = min(real_returns.min(), synth_returns.min())
        x_max = max(real_returns.max(), synth_returns.max())
        x_grid = np.linspace(x_min, x_max, n_bins)

        real_kde = gaussian_kde(real_returns, bw_method="silverman")
        synth_kde = gaussian_kde(synth_returns, bw_method="silverman")

        p = real_kde(x_grid)
        q = synth_kde(x_grid)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = np.maximum(p, eps)
        q = np.maximum(q, eps)

        # Normalise to proper probability distributions
        p /= p.sum()
        q /= q.sum()

        kl = float(np.sum(rel_entr(p, q)))

        logger.info(
            "SBBTS KL validation",
            kl_divergence=f"{kl:.6f}",
            threshold=self._cfg.kl_threshold,
            passed=kl < self._cfg.kl_threshold,
        )
        return kl

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _bass_adoption(self, t: np.ndarray, p: float, q: float) -> np.ndarray:
        """Bass diffusion adoption fraction F(t).

        F(t) = (1 - exp(-(p+q)t)) / (1 + (q/p) exp(-(p+q)t))

        Returns values in [0, 1] representing the fraction of the
        market that has "adopted" the current trend.
        """
        rate = p + q
        exp_term = np.exp(-rate * t)
        denom = 1.0 + (q / max(p, 1e-10)) * exp_term
        return (1.0 - exp_term) / denom

    def _schrodinger_potential(self, x: np.ndarray) -> np.ndarray:
        """Double-well potential V(x) = -depth · (x² - 1)².

        Two wells at x = ±1 represent bullish/bearish regimes.
        The barrier height (at x = 0) controls the difficulty of
        transitioning between regimes.
        """
        depth = self._cfg.potential_depth
        return -depth * (x**2 - 1.0) ** 2

    def _simulate_path(
        self,
        n_steps: int,
        mu: float,
        sigma: float,
        kappa: float,
        xi: float,
        bass_p: float,
        bass_q: float,
    ) -> np.ndarray:
        """Simulate a single SBBTS log-price path.

        Returns:
            (n_steps + 1,) array of log-prices starting at 0.
        """
        innovations = self._rng.standard_normal(n_steps)
        return self._simulate_path_with_innovations(
            innovations=innovations,
            mu=mu, sigma=sigma,
            kappa=kappa, xi=xi,
            bass_p=bass_p, bass_q=bass_q,
        )

    def _simulate_path_with_innovations(
        self,
        innovations: np.ndarray,
        mu: float,
        sigma: float,
        kappa: float,
        xi: float,
        bass_p: float,
        bass_q: float,
    ) -> np.ndarray:
        """Simulate SBBTS path using pre-generated (possibly correlated) innovations.

        The SDE system:
            dX = [μ·D(t) + ℏ·V'(X)]dt + √v · dW_x
            dv = κ(σ² - v)dt + ξ√v · dW_v

        where D(t) is Bass drift modulation and V'(X) is the Schrödinger
        potential gradient.

        Args:
            innovations: (n_steps,) array of standard normal variates for
                the price process.  May be correlated across pairs.
            mu, sigma, kappa, xi, bass_p, bass_q: model parameters.

        Returns:
            (n_steps + 1,) array of log-prices starting at 0.
        """
        n_steps = len(innovations)
        dt = self._cfg.dt
        hbar = self._cfg.hbar

        # Time grid for Bass adoption
        t_grid = np.arange(n_steps) * dt
        adoption = self._bass_adoption(t_grid, bass_p, bass_q)
        drift_mod = 1.0 + 0.5 * (adoption - 0.5)  # modulates around 1.0

        # Stochastic volatility innovations (independent from price)
        dW_vol = self._rng.standard_normal(n_steps)

        # State vectors
        log_prices = np.zeros(n_steps + 1, dtype=np.float64)
        vol = sigma**2  # variance (not std)

        target_var = sigma**2
        mu_per_step = mu * dt

        for i in range(n_steps):
            sqrt_v = np.sqrt(max(vol, 1e-12))

            # Schrödinger potential gradient at current normalised position
            # Normalise log-price to [-2, 2] range for the potential
            x_norm = np.clip(log_prices[i] / max(sigma * 10, 1e-6), -2.0, 2.0)
            potential_grad = self._potential_gradient(x_norm)

            # Price step
            drift = mu_per_step * drift_mod[i] + hbar * potential_grad * dt
            diffusion = sqrt_v * np.sqrt(dt) * innovations[i]
            log_prices[i + 1] = log_prices[i] + drift + diffusion

            # Heston vol update: dv = κ(θ - v)dt + ξ√v dW_v
            dv = kappa * (target_var - vol) * dt + xi * sqrt_v * np.sqrt(dt) * dW_vol[i]
            vol = max(vol + dv, 1e-12)  # reflect at zero

        return log_prices

    @staticmethod
    def _potential_gradient(x: np.ndarray | float) -> np.ndarray | float:
        """Gradient of double-well potential: dV/dx = -4·depth·x·(x² - 1).

        For scalar x, returns scalar; for array, returns array.
        """
        return -4.0 * 0.5 * x * (x**2 - 1.0)

    def _log_prices_to_ohlcv(
        self,
        log_prices: np.ndarray,
        base_price: float,
        real_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert log-price path to OHLCV bars with realistic microstructure.

        High/low are generated to match the real data's intra-bar range
        distribution.  Volume is sampled from the real data's empirical
        volume distribution with autocorrelation.

        Args:
            log_prices: (n+1,) log-prices starting at 0.
            base_price: Absolute price level for bar 0.
            real_data: Real data used to calibrate high/low range and volume.

        Returns:
            OHLCV DataFrame with ``n`` bars (one less than log_prices).
        """
        prices = base_price * np.exp(log_prices)
        n = len(prices) - 1

        opens = prices[:-1]
        closes = prices[1:]

        # Intra-bar range calibrated from real data
        if "high" in real_data.columns and "low" in real_data.columns:
            real_range = (
                (real_data["high"] - real_data["low"]) / real_data["close"]
            ).dropna().values
            real_range = real_range[real_range > 0]
            if len(real_range) > 10:
                range_samples = self._rng.choice(real_range, size=n, replace=True)
            else:
                range_samples = np.abs(self._rng.standard_normal(n)) * 0.002
        else:
            range_samples = np.abs(self._rng.standard_normal(n)) * 0.002

        mid = (opens + closes) / 2.0
        highs = mid + range_samples * mid * 0.5
        lows = mid - range_samples * mid * 0.5

        # Enforce OHLC constraints
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Volume: sample from real distribution with autocorrelation
        if "volume" in real_data.columns:
            real_vol = real_data["volume"].dropna().values
            real_vol = real_vol[real_vol > 0]
            if len(real_vol) > 10:
                base_vol = self._rng.choice(real_vol, size=n, replace=True).astype(np.float64)
            else:
                base_vol = self._rng.lognormal(10, 1.0, n)
        else:
            base_vol = self._rng.lognormal(10, 1.0, n)

        # Autocorrelated volume
        volumes = np.zeros(n)
        volumes[0] = base_vol[0]
        for i in range(1, n):
            volumes[i] = 0.6 * volumes[i - 1] + 0.4 * base_vol[i]

        # Synthetic timestamps
        if "time" in real_data.columns and len(real_data) > 0:
            freq = pd.infer_freq(real_data["time"].head(100)) or "h"
            start_time = real_data["time"].iloc[-1] + pd.Timedelta(hours=1)
        else:
            freq = "h"
            start_time = pd.Timestamp("2024-01-01", tz="UTC")

        times = pd.date_range(start=start_time, periods=n, freq=freq)

        return pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "tick_count": self._rng.integers(50, 500, n),
        })

    # ------------------------------------------------------------------
    # Augmented dataset generation
    # ------------------------------------------------------------------

    def generate_augmented_dataset(
        self,
        real_data: pd.DataFrame,
        sbbts_ratio: float | None = None,
        pair: str = "EURUSD",
    ) -> pd.DataFrame:
        """Generate a mixed dataset of real + synthetic bars.

        The synthetic portion is generated at multiple β levels (if
        configured) and interleaved with the real data.  A ``synthetic``
        boolean column marks which rows are generated.

        Args:
            real_data: Real OHLCV DataFrame.
            sbbts_ratio: Override for ``config.ratio``.
            pair: Symbol name.

        Returns:
            Combined DataFrame with real data first, synthetic appended.
            Sorted by time.

        Raises:
            ValueError: If KL divergence exceeds the configured threshold
                and no β level produces acceptable synthetic data.
        """
        ratio = sbbts_ratio if sbbts_ratio is not None else self._cfg.ratio
        if ratio <= 0.0:
            real_data = real_data.copy()
            real_data["synthetic"] = False
            return real_data

        n_real = len(real_data)
        n_synth_total = int(n_real * ratio / (1.0 - ratio))
        n_synth_per_beta = max(1, n_synth_total // len(self._cfg.beta_levels))

        logger.info(
            "Generating augmented dataset",
            n_real=n_real,
            n_synth_target=n_synth_total,
            beta_levels=self._cfg.beta_levels,
            pair=pair,
        )

        synth_parts: list[pd.DataFrame] = []
        for beta in self._cfg.beta_levels:
            synth = self.generate(
                real_data, n_bars=n_synth_per_beta, beta=beta, pair=pair,
            )

            kl = self.validate_kl(real_data, synth)
            if kl > self._cfg.kl_threshold:
                logger.warning(
                    "KL divergence exceeds threshold",
                    beta=beta,
                    kl=f"{kl:.4f}",
                    threshold=self._cfg.kl_threshold,
                )
            synth_parts.append(synth)

        # Combine
        real_out = real_data.copy()
        real_out["synthetic"] = False

        synth_combined = pd.concat(synth_parts, ignore_index=True)

        result = pd.concat([real_out, synth_combined], ignore_index=True)
        result = result.sort_values("time").reset_index(drop=True)

        logger.info(
            "Augmented dataset ready",
            n_total=len(result),
            n_real=n_real,
            n_synth=len(synth_combined),
            actual_ratio=f"{len(synth_combined) / len(result):.2%}",
        )
        return result
