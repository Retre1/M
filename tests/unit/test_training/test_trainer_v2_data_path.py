"""What TrainerV2 does to the data before the curriculum sees it.

Two things ``Trainer`` did and ``TrainerV2`` did not, while ``TrainerV2`` is
what ``scripts/train_v2.py`` actually runs:

* **Reserve a holdout.** Without it the next run trains on every bar, and any
  evaluation calling itself out-of-sample repeats the in-sample result that
  invalidated runs 1-6.
* **Attach the intermarket basket.** ``IntermarketCorrExtractor`` needs the
  other instruments' closes, and ``FSDExtractor`` measures dispersion *across*
  instruments — on one series that is identically zero, so gating v2 would
  condition on a constant.

Both failures are silent: training proceeds, features exist, and the numbers
look like numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.data.intermarket import merge_intermarket_columns


def _bars(n: int = 12_321, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 500),
    })


class TestHoldoutReachesTrainerV2:
    @staticmethod
    def _trainer(data):
        from apexfx.training.trainer_v2 import TrainerV2

        return TrainerV2(real_data=data)

    def test_the_curriculum_never_receives_the_holdout(self):
        data = _bars()
        trainer = self._trainer(data)
        assert len(trainer._train_data) < trainer._holdout_start

    def test_the_holdout_is_exposed_for_evaluation(self):
        data = _bars()
        trainer = self._trainer(data)
        holdout = trainer.holdout_data
        assert holdout is not None
        assert len(holdout) == len(data) - trainer._holdout_start

    def test_training_and_holdout_do_not_overlap_in_time(self):
        trainer = self._trainer(_bars())
        assert trainer._train_data["time"].max() < trainer.holdout_data["time"].min()

    def test_a_short_history_yields_no_holdout_rather_than_a_fake_one(self):
        trainer = self._trainer(_bars(n=200))
        assert trainer._holdout_start is None
        assert trainer.holdout_data is None

    def test_extra_symbols_are_split_too(self):
        """A second symbol left whole would leak the same period back in."""
        from apexfx.training.trainer_v2 import TrainerV2

        extra = _bars(seed=1)
        trainer = TrainerV2(real_data=_bars(), multi_symbol_data={"GBPUSD": extra})
        assert len(trainer._multi_symbol_data["GBPUSD"]) < len(extra)


class TestIntermarketMerge:
    N = 400

    @staticmethod
    def _store(tmp_path, instruments, times):
        rng = np.random.default_rng(1)
        for instrument in instruments:
            directory = tmp_path / "processed" / instrument / "H1"
            directory.mkdir(parents=True)
            pd.DataFrame({
                "time": times,
                "close": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(times)))),
            }).to_parquet(directory / "data.parquet")

    def _bars(self):
        return _bars(n=self.N)

    def test_present_instruments_are_attached(self, tmp_path):
        bars = self._bars()
        self._store(tmp_path, ["DXY", "XAUUSD"], bars["time"])
        merged, attached = merge_intermarket_columns(
            bars, ["DXY", "XAUUSD"], tmp_path,
        )
        assert attached == ["DXY", "XAUUSD"]
        assert {"DXY_close", "XAUUSD_close"} <= set(merged.columns)

    def test_a_missing_instrument_is_skipped_not_fatal(self, tmp_path):
        bars = self._bars()
        self._store(tmp_path, ["DXY"], bars["time"])
        merged, attached = merge_intermarket_columns(
            bars, ["DXY", "SPX"], tmp_path,
        )
        assert attached == ["DXY"]
        assert "SPX_close" not in merged.columns

    def test_an_empty_basket_warns_rather_than_returning_quietly(self, tmp_path, capsys):
        """The failure that looks like success: training runs, features exist,
        and two whole groups are constants.

        Asserted through ``capsys`` rather than ``caplog``: the project logs via
        structlog to stdout, so the stdlib capture fixture sees nothing.
        """
        _, attached = merge_intermarket_columns(self._bars(), ["DXY"], tmp_path)
        assert attached == []
        assert "identically zero" in capsys.readouterr().out

    def test_no_configured_instruments_is_not_a_warning(self, tmp_path, capsys):
        """Nothing requested is a choice, not a broken store."""
        _, attached = merge_intermarket_columns(self._bars(), [], tmp_path)
        assert attached == []
        assert "identically zero" not in capsys.readouterr().out

    def test_the_basket_is_what_makes_fsd_a_signal(self, tmp_path):
        from apexfx.features.fsd import FSDExtractor

        bars = self._bars()
        self._store(tmp_path, ["DXY", "XAUUSD"], bars["time"])
        merged, _ = merge_intermarket_columns(bars, ["DXY", "XAUUSD"], tmp_path)

        alone = FSDExtractor().extract(bars)["fsd_dispersion"]
        with_basket = FSDExtractor().extract(merged)["fsd_dispersion"]
        assert alone.nunique() == 1
        assert with_basket.nunique() > 50


class TestTrainerV2MergesBeforeFeatures:
    def test_the_merge_runs_when_an_app_config_supplies_a_basket(self, tmp_path):
        """Ordering matters: the basket has to be attached before the pipeline
        computes, or the extractors never see the columns."""
        from apexfx.config.schema import AppConfig
        from apexfx.training.trainer_v2 import TrainerV2

        bars = _bars(n=400)
        rng = np.random.default_rng(2)
        directory = tmp_path / "processed" / "DXY" / "H1"
        directory.mkdir(parents=True)
        pd.DataFrame({
            "time": bars["time"],
            "close": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(bars)))),
        }).to_parquet(directory / "data.parquet")

        app = AppConfig()
        app.symbols.intermarket = ["DXY"]
        app.base.paths.data_dir = str(tmp_path)

        trainer = TrainerV2(real_data=bars, app_config=app)
        merged = trainer._merge_intermarket(bars)
        assert "DXY_close" in merged.columns

    def test_without_an_app_config_the_frame_is_returned_unchanged(self):
        from apexfx.training.trainer_v2 import TrainerV2

        bars = _bars(n=400)
        trainer = TrainerV2(real_data=bars)
        assert list(trainer._merge_intermarket(bars).columns) == list(bars.columns)
