"""
Tests for VBIInference core — Step 3 of MI6.
"""
import numpy as np
import pytest
from vbi.simulator.spec import MonitorSpec
from vbi.feature_extraction import (
    FeaturePipeline,
    get_features_by_domain,
    get_features_by_given_names,
)
from vbi.inference import VBIInference, BoxUniform
from .conftest import make_mpr_spec


def _make_spec(n_nodes=4):
    return make_mpr_spec(n_nodes=n_nodes, monitors=(MonitorSpec("tavg", period=1.0),))


def _stat_pipeline():
    cfg = get_features_by_domain(domain="statistical")
    cfg = get_features_by_given_names(cfg, names=["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal="tavg", t_cut=0.0)


PRIOR = BoxUniform(
    low=np.array([0.1, -6.0]),
    high=np.array([2.0, -3.0]),
    param_names=["G", "eta"],
)
N_SIM   = 20
DURATION = 200.0


def _make_inf():
    return VBIInference(
        sim_spec          = _make_spec(),
        prior             = PRIOR,
        pipeline          = _stat_pipeline(),
        density_estimator = "maf",
        sim_backend       = "numpy",
        backend           = "numpy",
        show_progress_bars = False,
    )


class TestVBIInferenceCore:

    def test_repr(self):
        inf = _make_inf()
        r = repr(inf)
        assert "VBIInference" in r
        assert "maf" in r

    def test_simulate_returns_shapes(self):
        inf = _make_inf()
        theta, x = inf.simulate(N_SIM, DURATION, seed=0)
        assert theta.shape == (N_SIM, 2)
        assert x.ndim == 2
        assert x.shape[0] == N_SIM

    def test_simulate_populates_snpe(self):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=0)
        assert inf._snpe.n_simulations == N_SIM

    def test_simulate_stores_param_names_and_labels(self):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=0)
        assert inf._param_names == ["G", "eta"]
        assert len(inf._feature_labels) > 0

    def test_train_before_simulate_raises(self):
        inf = _make_inf()
        with pytest.raises(RuntimeError, match="simulate"):
            inf.train(stop_after_epochs=2)

    def test_full_round_trip(self):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=1)
        est = inf.train(stop_after_epochs=3, max_num_epochs=5)
        assert est is not None

        post = inf.build_posterior(est)
        x_obs = inf._snpe.get_simulations()[1][0]  # first feature vector
        samples = post.sample((50,), x=x_obs)
        assert samples.shape == (50, 2)

    def test_build_posterior_uses_last_estimator(self):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=2)
        est = inf.train(stop_after_epochs=3, max_num_epochs=5)
        # build_posterior without argument uses last estimator
        post = inf.build_posterior()
        x_obs = inf._snpe.get_simulations()[1][0]
        samples = post.sample((30,), x=x_obs)
        assert samples.shape == (30, 2)

    def test_sequential_rounds_accumulate(self):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=3)
        inf.simulate(N_SIM, DURATION, seed=4)
        assert inf._snpe.n_simulations == N_SIM * 2

        theta, x, _ = inf.get_simulations()
        assert theta.shape[0] == N_SIM * 2
        assert x.shape[0] == N_SIM * 2

    def test_get_simulations_matches_simulated(self):
        inf = _make_inf()
        theta_sim, x_sim = inf.simulate(N_SIM, DURATION, seed=5)
        theta_stored, x_stored, _ = inf.get_simulations()
        # SNPE converts to float32 internally; compare at float32 precision
        np.testing.assert_array_almost_equal(
            theta_sim.astype(np.float32), theta_stored, decimal=5
        )

    def test_simulate_with_cache(self, tmp_path):
        inf = _make_inf()
        theta, x = inf.simulate(
            N_SIM, DURATION, seed=6,
            cache_dir=tmp_path / "cache",
            chunk_size=10,
        )
        assert theta.shape[0] == N_SIM
        assert (tmp_path / "cache" / "metadata.json").exists()

    def test_extract_from_cache_static(self, tmp_path):
        inf = _make_inf()
        inf.simulate(
            N_SIM, DURATION, seed=7,
            cache_dir=tmp_path / "cache",
            chunk_size=N_SIM,
        )
        # Re-extract with the same pipeline via static method
        theta2, x2 = VBIInference.extract_from_cache(
            tmp_path / "cache", _stat_pipeline()
        )
        assert theta2.shape[0] == N_SIM
        assert x2.shape[0] == N_SIM

    def test_default_train_kwargs_used(self):
        inf = _make_inf()
        inf._default_train_kwargs = {"stop_after_epochs": 2, "max_num_epochs": 4}
        inf.simulate(N_SIM, DURATION, seed=8)
        # Should not raise; default kwargs are forwarded
        est = inf.train()
        assert est is not None

    def test_default_train_kwargs_overridden(self):
        inf = _make_inf()
        inf._default_train_kwargs = {"stop_after_epochs": 100, "max_num_epochs": 200}
        inf.simulate(N_SIM, DURATION, seed=9)
        # Explicit kwarg overrides default
        est = inf.train(stop_after_epochs=2, max_num_epochs=4)
        assert est is not None
