"""
Tests for simulate_for_vbi_sweep — Step 1 of MI6.
"""
import numpy as np
import pytest
from vbi.simulator.spec import MonitorSpec
from vbi.feature_extraction import (
    FeaturePipeline,
    get_features_by_domain,
    get_features_by_given_names,
)
from vbi.inference import BoxUniform, simulate_for_vbi_sweep
from .conftest import make_mpr_spec


def _make_spec(n_nodes=4):
    return make_mpr_spec(n_nodes=n_nodes, monitors=(MonitorSpec("tavg", period=1.0),))


def _stat_pipeline(t_cut=0.0):
    cfg = get_features_by_domain(domain="statistical")
    cfg = get_features_by_given_names(cfg, names=["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal="tavg", t_cut=t_cut)


N_SIM = 8
DURATION = 200.0


class TestSimulateForVbiSweep:

    def test_shapes(self):
        spec = _make_spec()
        prior = BoxUniform(
            low=np.array([0.1, -6.0]),
            high=np.array([2.0, -3.0]),
            param_names=["G", "eta"],
        )
        theta, x, param_names, feat_labels = simulate_for_vbi_sweep(
            sim_spec=spec,
            prior=prior,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            sim_backend="numpy",
            seed=0,
        )
        assert theta.shape == (N_SIM, 2), f"theta shape {theta.shape}"
        assert x.ndim == 2
        assert x.shape[0] == N_SIM
        assert x.dtype == np.float64
        assert theta.dtype == np.float64

    def test_param_names_and_labels(self):
        spec = _make_spec()
        prior = BoxUniform(
            low=np.array([0.1, -6.0]),
            high=np.array([2.0, -3.0]),
            param_names=["G", "eta"],
        )
        _, _, param_names, feat_labels = simulate_for_vbi_sweep(
            sim_spec=spec,
            prior=prior,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            sim_backend="numpy",
            seed=0,
        )
        assert param_names == ["G", "eta"]
        assert len(feat_labels) > 0
        assert all(isinstance(l, str) for l in feat_labels)

    def test_features_finite(self):
        spec = _make_spec()
        prior = BoxUniform(
            low=np.array([0.1, -6.0]),
            high=np.array([2.0, -3.0]),
        )
        _, x, _, _ = simulate_for_vbi_sweep(
            sim_spec=spec,
            prior=prior,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            sim_backend="numpy",
            seed=1,
        )
        assert np.all(np.isfinite(x)), "All feature values must be finite"

    def test_seed_reproducibility(self):
        spec = _make_spec()
        prior = BoxUniform(
            low=np.array([0.1, -6.0]),
            high=np.array([2.0, -3.0]),
            param_names=["G", "eta"],
        )
        kwargs = dict(
            sim_spec=spec, prior=prior, pipeline=_stat_pipeline(),
            num_simulations=N_SIM, duration=DURATION, sim_backend="numpy",
        )
        theta1, x1, _, _ = simulate_for_vbi_sweep(**kwargs, seed=42)
        theta2, x2, _, _ = simulate_for_vbi_sweep(**kwargs, seed=42)
        np.testing.assert_array_equal(theta1, theta2)
        np.testing.assert_array_equal(x1, x2)

    def test_auto_param_names_when_prior_has_none(self):
        spec = _make_spec()
        prior = BoxUniform(
            low=np.array([0.1]),
            high=np.array([2.0]),
        )
        _, _, param_names, _ = simulate_for_vbi_sweep(
            sim_spec=spec,
            prior=prior,
            pipeline=_stat_pipeline(),
            num_simulations=4,
            duration=DURATION,
            sim_backend="numpy",
            seed=0,
        )
        assert param_names == ["p0"]
