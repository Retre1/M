"""Tests for the TFT encoder.

``TemporalFusionTransformer`` sits in the working path — trainer, pretrainer and
HiveMind all import it — and had no tests at all, which made it the largest
uncovered surface in the repository.

The property that matters most here is the same one the feature extractors were
checked against: the representation of bar *t* must not move when bars after
*t* change. A feature pipeline that is causal feeding a model that is not
produces a backtest that cannot be traded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from apexfx.models.tft.tft_model import TemporalFusionTransformer  # noqa: E402

BATCH, SEQ, N_VARS, D_MODEL = 3, 12, 4, 16


def _model(**kwargs) -> TemporalFusionTransformer:
    defaults = dict(
        n_continuous_vars=N_VARS, n_known_future_vars=2, n_static_vars=0,
        d_model=D_MODEL, n_heads=2, n_lstm_layers=1, dropout=0.0,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    model = TemporalFusionTransformer(**defaults)
    model.eval()  # dropout off: these test the architecture, not sampling noise
    return model


def _past(seq: int = SEQ, n_vars: int = N_VARS, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, seq, n_vars, generator=gen)


class TestShapes:
    def test_encoded_state_is_one_vector_per_sample(self):
        out = _model()(_past())
        assert out.encoded_state.shape == (BATCH, D_MODEL)

    def test_temporal_features_keep_the_time_axis(self):
        out = _model()(_past())
        assert out.temporal_features.shape == (BATCH, SEQ, D_MODEL)

    def test_attention_is_square_over_the_sequence(self):
        out = _model()(_past())
        assert out.attention_weights.shape == (BATCH, SEQ, SEQ)

    def test_variable_importance_is_one_weight_per_variable_per_step(self):
        out = _model()(_past())
        assert out.variable_importance.shape == (BATCH, SEQ, N_VARS)

    def test_future_covariates_extend_the_sequence(self):
        model = _model()
        future = torch.randn(BATCH, 4, 2)
        out = model(_past(), x_future=future)
        assert out.temporal_features.shape == (BATCH, SEQ + 4, D_MODEL)


class TestCausality:
    """The encoding of bar t must not depend on bars after t."""

    def test_a_later_bar_does_not_change_an_earlier_representation(self):
        model = _model()
        x = _past()
        cut = 7

        baseline = model(x).temporal_features
        tampered = x.clone()
        tampered[:, cut:, :] += 5.0  # rewrite the entire future
        after = model(tampered).temporal_features

        torch.testing.assert_close(
            baseline[:, :cut, :], after[:, :cut, :], rtol=1e-4, atol=1e-5,
        )

    def test_the_change_does_reach_the_bar_it_was_made_at(self):
        """Guards the test above: an encoder that ignored its input entirely
        would pass prefix-invariance trivially."""
        model = _model()
        x = _past()
        cut = 7
        tampered = x.clone()
        tampered[:, cut:, :] += 5.0

        baseline = model(x).temporal_features
        after = model(tampered).temporal_features
        assert not torch.allclose(baseline[:, cut:, :], after[:, cut:, :], atol=1e-3)

    def test_attention_never_looks_forward(self):
        weights = _model()(_past()).attention_weights
        upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
        assert weights[:, upper].abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_attention_rows_are_distributions(self):
        weights = _model()(_past()).attention_weights
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(BATCH, SEQ), rtol=1e-4, atol=1e-5,
        )


class TestVariableSelection:
    def test_weights_are_a_distribution_over_variables(self):
        importance = _model()(_past()).variable_importance
        torch.testing.assert_close(
            importance.sum(dim=-1), torch.ones(BATCH, SEQ), rtol=1e-4, atol=1e-5,
        )

    def test_weights_are_non_negative(self):
        assert _model()(_past()).variable_importance.min().item() >= 0.0

    def test_variables_are_not_weighted_identically(self):
        """A selection network that returned a flat 1/n would be decoration."""
        importance = _model()(_past()).variable_importance
        assert importance.std(dim=-1).max().item() > 1e-4


class TestBatchIndependence:
    def test_one_sample_does_not_affect_another(self):
        """Any statistic taken across the batch would leak between samples —
        and in a trading model the batch spans different points in time."""
        model = _model()
        x = _past()
        together = model(x).encoded_state
        alone = model(x[:1]).encoded_state
        torch.testing.assert_close(together[:1], alone, rtol=1e-4, atol=1e-5)


class TestStaticCovariates:
    def test_static_input_changes_the_encoding(self):
        model = _model(n_static_vars=2)
        x = _past()
        a = model(x, x_static=torch.zeros(BATCH, 2)).encoded_state
        b = model(x, x_static=torch.ones(BATCH, 2) * 3.0).encoded_state
        assert not torch.allclose(a, b, atol=1e-4)

    def test_static_covariates_stay_optional(self):
        model = _model(n_static_vars=2)
        assert model(_past()).encoded_state.shape == (BATCH, D_MODEL)


class TestTrainability:
    def test_every_parameter_receives_a_gradient(self):
        """A branch with no gradient is a branch that is not being trained."""
        model = _model(n_static_vars=2)
        model.train()
        out = model(_past(), x_static=torch.ones(BATCH, 2))
        out.encoded_state.sum().backward()

        starved = [
            name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)
        ]
        # The decoder is only exercised when future covariates are supplied.
        starved = [n for n in starved if "decoder" not in n and "future" not in n]
        assert starved == [], f"no gradient reached: {starved}"

    def test_eval_mode_is_deterministic(self):
        model = _model(dropout=0.5)
        x = _past()
        torch.testing.assert_close(
            model(x).encoded_state, model(x).encoded_state, rtol=0, atol=0,
        )

    def test_the_output_is_finite(self):
        out = _model()(_past() * 100.0)
        assert torch.isfinite(out.encoded_state).all()
