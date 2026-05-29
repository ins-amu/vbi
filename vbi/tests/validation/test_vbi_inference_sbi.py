"""
Tests for VBIInference with inference_backend='sbi' — Step 8 of MI6.

All tests are skipped if sbi/torch is not installed.
"""
import numpy as np
import pytest
from vbi.simulator.spec import MonitorSpec
from vbi.feature_extraction import (
    FeaturePipeline, get_features_by_domain, get_features_by_given_names,
)
from vbi.inference import VBIInference, BoxUniform
from .conftest import make_mpr_spec

sbi   = pytest.importorskip("sbi",   reason="sbi not installed")
torch = pytest.importorskip("torch", reason="torch not installed")


def _make_spec(n_nodes=4):
    return make_mpr_spec(n_nodes=n_nodes, monitors=(MonitorSpec("tavg", period=1.0),))


def _stat_pipeline():
    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal="tavg", t_cut=0.0)


PRIOR    = BoxUniform(np.array([0.1, -6.0]), np.array([2.0, -3.0]), param_names=["G", "eta"])
N_SIM    = 20
DURATION = 200.0


def _make_sbi_inf():
    return VBIInference(
        sim_spec          = _make_spec(),
        prior             = PRIOR,
        pipeline          = _stat_pipeline(),
        density_estimator = "maf",
        sim_backend       = "numpy",
        inference_backend = "sbi",
        show_progress_bars = False,
    )


class TestVBIInferenceSBIBackend:

    def test_repr_shows_sbi(self):
        inf = _make_sbi_inf()
        assert "sbi" in repr(inf)

    def test_simulate_returns_shapes(self):
        inf = _make_sbi_inf()
        theta, x = inf.simulate(N_SIM, DURATION, seed=0)
        assert theta.shape == (N_SIM, 2)
        assert x.ndim == 2 and x.shape[0] == N_SIM

    def test_get_simulations_works(self):
        inf = _make_sbi_inf()
        inf.simulate(N_SIM, DURATION, seed=1)
        theta, x, _ = inf.get_simulations()
        assert theta.shape[0] == N_SIM

    def test_full_round_trip(self):
        inf = _make_sbi_inf()
        inf.simulate(N_SIM, DURATION, seed=2)
        est = inf.train(
            training_batch_size = 32,
            stop_after_epochs   = 3,
            max_num_epochs      = 5,
        )
        assert est is not None

        post = inf.build_posterior(est)
        x_obs = inf.get_simulations()[1][0]

        # build_posterior returns a numpy-compatible wrapper
        samples = post.sample((30,), x=x_obs)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (30, 2)

    def test_posterior_log_prob_returns_numpy(self):
        inf = _make_sbi_inf()
        inf.simulate(N_SIM, DURATION, seed=3)
        inf.train(training_batch_size=32, stop_after_epochs=3, max_num_epochs=5)
        post  = inf.build_posterior()
        x_obs = inf.get_simulations()[1][0]
        theta = inf.get_simulations()[0][:5]
        lp    = post.log_prob(theta, x=x_obs)
        assert isinstance(lp, np.ndarray)
        assert lp.shape == (5,)

    def test_prior_to_sbi_box_uniform(self):
        from vbi.inference._vbi_inference import _prior_to_sbi
        sbi_prior = _prior_to_sbi(PRIOR)
        # sbi BoxUniform should sample correctly
        s = sbi_prior.sample((10,))
        assert s.shape == (10, 2)

    def test_prior_to_sbi_generic(self):
        from vbi.inference import Gaussian
        from vbi.inference._vbi_inference import _prior_to_sbi
        g = Gaussian(np.zeros(2), np.ones(2))
        sbi_prior = _prior_to_sbi(g)
        s = sbi_prior.sample((5,))
        assert s.shape == (5, 2)

    def test_save_does_not_crash(self, tmp_path):
        inf = _make_sbi_inf()
        inf.simulate(N_SIM, DURATION, seed=4)
        inf.train(training_batch_size=32, stop_after_epochs=3, max_num_epochs=5)
        # estimator warning goes to log (not Python warnings), just verify no crash
        inf.save(tmp_path / "ckpt.npz")
        assert (tmp_path / "ckpt.npz").exists()

    def test_save_load_restores_simulations(self, tmp_path):
        inf = _make_sbi_inf()
        theta_orig, _ = inf.simulate(N_SIM, DURATION, seed=5)
        inf.save(tmp_path / "ckpt.npz")

        inf2 = VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec = _make_spec(),
            pipeline = _stat_pipeline(),
            prior    = PRIOR,
        )
        assert inf2._inference_backend == "sbi"
        theta2, _, _ = inf2.get_simulations()
        np.testing.assert_array_almost_equal(
            theta_orig.astype(np.float32), theta2, decimal=5
        )

    def test_inference_engine_kwarg(self):
        """Pass an explicit sbi SNPE object via inference_engine."""
        from sbi.inference import SNPE as _SBI_SNPE
        from vbi.inference._vbi_inference import _prior_to_sbi

        torch_prior = _prior_to_sbi(PRIOR)
        sbi_snpe    = _SBI_SNPE(prior=torch_prior, density_estimator="maf",
                                show_progress_bars=False)
        inf = VBIInference(
            sim_spec        = _make_spec(),
            prior           = PRIOR,
            pipeline        = _stat_pipeline(),
            sim_backend     = "numpy",
            inference_engine = sbi_snpe,
        )
        assert inf._inference_backend == "sbi"
        theta, x = inf.simulate(N_SIM, DURATION, seed=6)
        assert theta.shape[0] == N_SIM
