"""FSD-conditioned gating, and why it could not simply be switched on.

Run 1 collapsed the meta-controller to a gating entropy of 0.031 — one agent
taking every decision. ``HiveMindGating_v2`` exists to counter that, with
entropy regularisation and a stream conditioned on the cross-sectional regime.

That stream is the catch. FSD is the dispersion of skewness *across*
instruments, so on a single symbol it is identically zero by construction:
``std`` of one value is undefined, and the detector's fillna turns it into a
constant. Enabling the gating without a basket would have produced a third
instance of the failure this session already fixed twice — a parameter wired in
with no path to any signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from apexfx.features.fsd import MIN_INSTRUMENTS, FSDExtractor  # noqa: E402
from apexfx.models.ensemble.hive_mind import HiveMind  # noqa: E402

BATCH, SEQ, N_VARS = 2, 8, 4


def _prices(rng, n: int) -> np.ndarray:
    return 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))


def _basket(n: int = 800, n_instruments: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = {"close": _prices(rng, n)}
    for name in ("DXY", "XAUUSD", "SPX")[: n_instruments - 1]:
        frame[f"{name}_close"] = _prices(rng, n)
    return pd.DataFrame(frame)


class TestFSDNeedsABasket:
    def test_one_instrument_produces_a_constant_and_says_so(self, capsys):
        """The whole reason gating v2 was not simply switched on.

        ``capsys``, not ``caplog``: structlog writes to stdout, so the stdlib
        capture sees nothing and an assertion against it would pass on an empty
        string — which is what the first version of this test did.
        """
        out = FSDExtractor().extract(_basket(n_instruments=1))
        assert out["fsd_dispersion"].nunique() == 1
        assert "basket" in capsys.readouterr().out.lower()

    def test_a_basket_produces_a_varying_signal(self):
        out = FSDExtractor().extract(_basket())
        assert out["fsd_dispersion"].nunique() > 50

    def test_the_minimum_is_two_instruments(self):
        assert MIN_INSTRUMENTS == 2

    def test_the_regime_is_a_one_hot(self):
        out = FSDExtractor().extract(_basket())
        bits = [c for c in out.columns if "regime" in c]
        assert set(np.round(out[bits].sum(axis=1).unique(), 6)) == {1.0}

    def test_every_regime_actually_occurs(self):
        """Absolute cut-offs of 0.3 / 0.7 against a dispersion that runs 0-0.3
        put every bar in one class; the thresholds are quantiles of the
        dispersion's own history instead."""
        out = FSDExtractor().extract(_basket())
        bits = [c for c in out.columns if "regime" in c]
        assert all(out[c].sum() > 0 for c in bits), dict(out[bits].sum())

    def test_the_detector_warmup_does_not_anchor_the_quantiles(self):
        """The detector reports zero until its window fills. Left in the
        history those zeros drag the low quantile to zero and RISK_ON can
        never fire, since nothing is below zero."""
        out = FSDExtractor().extract(_basket())
        assert out["fsd_regime_risk_on"].sum() > 0

    def test_the_columns_are_what_the_gating_expects(self):
        from apexfx.models.config import GatingV2Config

        assert len(FSDExtractor().feature_names) == GatingV2Config().d_fsd


class TestGatingV2InHiveMind:
    @staticmethod
    def _hive(use_v2: bool) -> HiveMind:
        torch.manual_seed(0)
        model = HiveMind(
            n_continuous_vars=N_VARS, n_known_future_vars=2, d_model=16, n_heads=2,
            dropout=0.0, use_gating_v2=use_v2,
        )
        model.eval()
        return model

    @staticmethod
    def _inputs():
        gen = torch.Generator().manual_seed(1)
        return dict(
            market_features=torch.randn(BATCH, SEQ, N_VARS, generator=gen),
            time_features=torch.randn(BATCH, SEQ, 2, generator=gen),
            trend_features=torch.randn(BATCH, 8, generator=gen),
            reversion_features=torch.randn(BATCH, 8, generator=gen),
            regime_features=torch.randn(BATCH, 6, generator=gen),
        )

    def test_v1_remains_the_default(self):
        assert not self._hive(use_v2=False).use_gating_v2

    def test_v2_runs_when_given_the_channel(self):
        out = self._hive(use_v2=True)(
            **self._inputs(), fsd_features=torch.randn(BATCH, 4),
        )
        assert out.gating_weights.shape == (BATCH, 3)
        assert torch.isfinite(out.action).all()

    def test_v2_refuses_to_run_on_a_missing_channel(self):
        """Zeros would leave the FSD stream constant — wired but inert, the
        state this gating was enabled to escape."""
        with pytest.raises(ValueError, match="fsd_features"):
            self._hive(use_v2=True)(**self._inputs())

    def test_the_fsd_channel_changes_the_gating(self):
        """If the conditioning did nothing, enabling v2 would be decoration."""
        model = self._hive(use_v2=True)
        args = self._inputs()
        risk_on = model(**args, fsd_features=torch.tensor([[0.1, 1.0, 0.0, 0.0]] * BATCH))
        risk_off = model(**args, fsd_features=torch.tensor([[0.9, 0.0, 0.0, 1.0]] * BATCH))
        assert not torch.allclose(risk_on.gating_weights, risk_off.gating_weights, atol=1e-6)

    def test_diagnostics_surface_the_collapse_metric(self):
        """Run 1 collapsed to entropy 0.031 and it was only visible in logs."""
        out = self._hive(use_v2=True)(
            **self._inputs(), fsd_features=torch.randn(BATCH, 4),
        )
        assert out.gating_diagnostics is not None
        assert any("entropy" in k for k in out.gating_diagnostics)

    def test_v1_reports_no_diagnostics(self):
        assert self._hive(use_v2=False)(**self._inputs()).gating_diagnostics is None
