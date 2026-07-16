"""
Tests for InferencePipeline diagnostic helpers - Step 6 of MI6.
"""
import numpy as np
import pytest
from vbi.simulator.spec import MonitorSpec
from vbi.feature_extraction import (
    FeaturePipeline, get_features_by_domain, get_features_by_given_names,
)
from vbi.inference import InferencePipeline, BoxUniform
from .conftest import make_mpr_spec


def _make_spec(n_nodes=4):
    return make_mpr_spec(n_nodes=n_nodes, monitors=(MonitorSpec("tavg", period=1.0),))


def _stat_pipeline():
    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal="tavg", t_cut=0.0)


PRIOR    = BoxUniform(np.array([0.1, -6.0]), np.array([2.0, -3.0]), param_names=["G", "eta"])
N_SIM    = 20
DURATION = 200.0


def _trained_inf():
    inf = InferencePipeline(
        sim_spec=_make_spec(), prior=PRIOR, pipeline=_stat_pipeline(),
        integrator_backend="numpy", estimator_backend="numpy", show_progress_bars=False,
    )
    inf.simulate(N_SIM, DURATION, seed=0)
    inf.train(stop_after_epochs=3, max_num_epochs=5)
    return inf


class TestPlotLoss:
    mpl = pytest.importorskip("matplotlib")

    def test_returns_figure(self):
        inf = _trained_inf()
        fig = inf.plot_loss()
        assert hasattr(fig, "savefig")    # it's a matplotlib Figure

    def test_raises_before_train(self):
        inf = InferencePipeline(
            sim_spec=_make_spec(), prior=PRIOR, pipeline=_stat_pipeline(),
            integrator_backend="numpy", estimator_backend="numpy", show_progress_bars=False,
        )
        with pytest.raises(RuntimeError, match="train"):
            inf.plot_loss()


class TestPairplot:
    mpl = pytest.importorskip("matplotlib")

    def test_returns_figure(self):
        inf = _trained_inf()
        x_obs = inf._snpe.get_simulations()[1][0]
        fig = inf.pairplot(x_obs, num_samples=50)
        assert hasattr(fig, "savefig")

    def test_raises_before_train(self):
        inf = InferencePipeline(
            sim_spec=_make_spec(), prior=PRIOR, pipeline=_stat_pipeline(),
            integrator_backend="numpy", estimator_backend="numpy", show_progress_bars=False,
        )
        with pytest.raises(RuntimeError, match="train"):
            inf.pairplot(np.zeros(4))


class TestRunSBC:

    def test_raises_without_duration(self):
        inf = _trained_inf()
        with pytest.raises(RuntimeError, match="duration"):
            inf.run_sbc()

    def test_raises_before_train(self):
        inf = InferencePipeline(
            sim_spec=_make_spec(), prior=PRIOR, pipeline=_stat_pipeline(),
            integrator_backend="numpy", estimator_backend="numpy", show_progress_bars=False,
        )
        inf.simulate(N_SIM, DURATION, seed=1)
        with pytest.raises(RuntimeError, match="train"):
            inf.run_sbc(duration=DURATION)

    def test_returns_ranks_dict(self):
        inf = _trained_inf()
        result = inf.run_sbc(
            duration              = DURATION,
            num_sbc_runs          = 10,
            num_posterior_samples = 20,
            seed                  = 0,
        )
        assert "ranks" in result
        assert result["ranks"].shape[0] == 10   # one rank per SBC run
        assert result["ranks"].shape[1] == 2    # one column per parameter

    def test_make_simulator_fn(self):
        from vbi.inference._inference_pipeline import _make_simulator_fn
        sim_fn = _make_simulator_fn(
            _make_spec(), PRIOR, _stat_pipeline(), "numpy", DURATION
        )
        theta = np.array([1.0, -5.0])
        x = sim_fn(theta)
        assert x.ndim == 1
        assert np.all(np.isfinite(x))
