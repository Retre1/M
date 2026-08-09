"""The env must publish realised PnL when an episode ends.

This is the wiring half of Gate 0: ``test_financial_metrics`` proves the
callback computes the right numbers, this proves a real environment actually
hands it the inputs. Without both, the metrics could be correct in isolation
and still never see a trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.env.forex_env import ForexTradingEnv

EPISODE_LENGTH = 120


def _make_env(seed: int = 0) -> ForexTradingEnv:
    rng = np.random.default_rng(seed)
    n = 400
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    data = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": rng.integers(100, 1000, n),
    })
    return ForexTradingEnv(
        data=data,
        episode_length=EPISODE_LENGTH,
        lookback=10,
        n_market_features=5,
    )


def _run_episode(env: ForexTradingEnv, actions) -> dict:
    """Drive an episode to its end and return the final info dict."""
    env.reset(seed=0)
    info: dict = {}
    for action in actions:
        _, _, terminated, truncated, info = env.step(
            np.array([action], dtype=np.float32),
        )
        if terminated or truncated:
            break
    return info


class TestEpisodeFinancialsAreEmitted:
    def test_absent_mid_episode(self):
        """Only the final step carries it — it describes the whole episode."""
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
        assert "episode_financials" not in info

    def test_present_when_the_episode_ends(self):
        # Alternate direction so positions actually open and close.
        actions = [0.8 if (i // 5) % 2 == 0 else -0.8 for i in range(EPISODE_LENGTH + 5)]
        info = _run_episode(_make_env(), actions)
        assert "episode_financials" in info, "no financials on the terminal step"

    def test_payload_shape(self):
        actions = [0.8 if (i // 5) % 2 == 0 else -0.8 for i in range(EPISODE_LENGTH + 5)]
        fin = _run_episode(_make_env(), actions)["episode_financials"]

        assert set(fin) == {
            "returns", "trade_returns", "equity_start",
            "equity_end", "n_trades", "n_bars",
        }
        assert isinstance(fin["returns"], np.ndarray)
        assert np.all(np.isfinite(fin["returns"]))
        assert np.all(np.isfinite(fin["trade_returns"]))
        assert fin["n_bars"] == fin["returns"].size


class TestTradingIsVisible:
    def test_an_agent_that_trades_reports_trades(self):
        """Run 1's agent never traded and nothing in its metrics said so."""
        actions = [0.8 if (i // 5) % 2 == 0 else -0.8 for i in range(EPISODE_LENGTH + 5)]
        fin = _run_episode(_make_env(), actions)["episode_financials"]

        assert fin["n_trades"] > 0
        assert fin["trade_returns"].size > 0

    def test_a_flat_agent_reports_none(self):
        fin = _run_episode(_make_env(), [0.0] * (EPISODE_LENGTH + 5))["episode_financials"]

        assert fin["n_trades"] == 0
        assert fin["trade_returns"].size == 0
        assert fin["equity_end"] == pytest.approx(fin["equity_start"])


class TestReturnsMatchEquity:
    def test_returns_compound_to_the_equity_change(self):
        """Guards against an off-by-one or a wrong denominator in the diff."""
        actions = [0.8 if (i // 5) % 2 == 0 else -0.8 for i in range(EPISODE_LENGTH + 5)]
        fin = _run_episode(_make_env(), actions)["episode_financials"]

        compounded = float(np.prod(1.0 + fin["returns"]))
        expected = fin["equity_end"] / fin["equity_start"]
        assert compounded == pytest.approx(expected, rel=1e-9)

    def test_flat_agent_has_zero_returns(self):
        fin = _run_episode(_make_env(), [0.0] * (EPISODE_LENGTH + 5))["episode_financials"]
        assert np.allclose(fin["returns"], 0.0)
