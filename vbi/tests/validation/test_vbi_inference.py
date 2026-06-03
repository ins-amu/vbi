"""
Tests for VBIInference core - Step 3 of MI6.
"""
import numpy as np
import pytest
from vbi.simulator.spec import MonitorSpec
from vbi.feature_extraction import (
    FeaturePipeline,
    FeaturePruner,
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

    def test_append_simulations(self):
        inf = _make_inf()
        theta = np.array([[0.2, -5.5], [0.4, -4.5]], dtype=float)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        inf.append_simulations(
            theta,
            x,
            param_names=["G", "eta"],
            feature_labels=["f0", "f1"],
        )

        theta_stored, x_stored, _ = inf.get_simulations()
        np.testing.assert_array_equal(theta_stored, theta.astype(np.float32))
        np.testing.assert_array_equal(x_stored, x.astype(np.float32))
        assert inf._param_names == ["G", "eta"]
        assert inf._feature_labels == ["f0", "f1"]
        assert inf._snpe.n_simulations == 2

    def test_simulate_with_cache(self, tmp_path):
        inf = _make_inf()
        theta, x = inf.simulate(
            N_SIM, DURATION, seed=6,
            cache_dir=tmp_path / "cache",
            chunk_size=10,
            n_workers=2,
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


class TestVBIInferenceSaveLoad:

    def test_save_creates_file(self, tmp_path):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=10)
        inf.save(tmp_path / "ckpt.npz")
        assert (tmp_path / "ckpt.npz").exists()

    def test_save_adds_npz_extension(self, tmp_path):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=11)
        inf.save(tmp_path / "ckpt")          # no extension
        assert (tmp_path / "ckpt.npz").exists()

    def test_load_restores_simulations(self, tmp_path):
        inf = _make_inf()
        theta_orig, x_orig = inf.simulate(N_SIM, DURATION, seed=12)
        inf.save(tmp_path / "ckpt.npz")

        inf2 = VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec = _make_spec(),
            pipeline = _stat_pipeline(),
            prior    = PRIOR,
        )
        theta2, x2, _ = inf2.get_simulations()
        assert theta2.shape == (N_SIM, 2)
        assert x2.shape[0] == N_SIM
        # Values survive round-trip at float32 precision
        np.testing.assert_array_almost_equal(
            theta_orig.astype("f4"), theta2, decimal=5
        )

    def test_load_restores_metadata(self, tmp_path):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=13)
        inf.save(tmp_path / "ckpt.npz")

        inf2 = VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec = _make_spec(),
            pipeline = _stat_pipeline(),
            prior    = PRIOR,
        )
        assert inf2._param_names    == ["G", "eta"]
        assert len(inf2._feature_labels) > 0

    def test_load_restores_estimator_and_posterior_works(self, tmp_path):
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=14)
        est_orig = inf.train(stop_after_epochs=3, max_num_epochs=5)
        inf.save(tmp_path / "ckpt.npz")

        inf2 = VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec = _make_spec(),
            pipeline = _stat_pipeline(),
            prior    = PRIOR,
        )
        assert inf2._last_estimator is not None

        x_obs = inf2._snpe.get_simulations()[1][0]
        post  = inf2.build_posterior()
        samples = post.sample((50,), x=x_obs)
        assert samples.shape == (50, 2)

    def test_load_posterior_samples_consistent(self, tmp_path):
        """Posterior samples before and after save/load agree at the same seed."""
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=15)
        inf.train(stop_after_epochs=3, max_num_epochs=5)

        x_obs     = inf._snpe.get_simulations()[1][0]
        post_orig = inf.build_posterior()
        s_orig    = post_orig.sample((30,), x=x_obs, seed=99)

        inf.save(tmp_path / "ckpt.npz")
        inf2 = VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec = _make_spec(),
            pipeline = _stat_pipeline(),
            prior    = PRIOR,
        )
        post2 = inf2.build_posterior()
        s2    = post2.sample((30,), x=x_obs, seed=99)

        np.testing.assert_array_almost_equal(s_orig, s2, decimal=4)

    def test_save_without_estimator(self, tmp_path):
        """save() works even if train() was never called."""
        inf = _make_inf()
        inf.simulate(N_SIM, DURATION, seed=16)
        inf.save(tmp_path / "ckpt.npz")   # no train() call

        inf2 = VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec = _make_spec(),
            pipeline = _stat_pipeline(),
            prior    = PRIOR,
        )
        assert inf2._last_estimator is None
        assert inf2._snpe.n_simulations == N_SIM

    def test_load_restores_feature_pruner_state(self, tmp_path):
        pipeline = _stat_pipeline()
        pipeline.pruner = FeaturePruner(min_std=1e-4, max_corr=0.98)
        inf = VBIInference(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=pipeline,
            density_estimator="maf",
            sim_backend="numpy",
            backend="numpy",
            show_progress_bars=False,
        )
        inf.simulate(N_SIM, DURATION, seed=17)
        assert pipeline.pruner.kept_mask_ is not None
        original_mask = pipeline.pruner.kept_mask_.copy()
        original_labels = list(pipeline.pruner.kept_labels_)

        inf.save(tmp_path / "ckpt.npz")

        pipeline2 = _stat_pipeline()
        pipeline2.pruner = FeaturePruner()
        inf2 = VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec=_make_spec(),
            pipeline=pipeline2,
            prior=PRIOR,
        )

        np.testing.assert_array_equal(pipeline2.pruner.kept_mask_, original_mask)
        assert pipeline2.pruner.kept_labels_ == original_labels
        assert inf2._feature_labels == original_labels

    def test_load_attaches_saved_feature_pruner(self, tmp_path):
        pipeline = _stat_pipeline()
        pipeline.pruner = FeaturePruner(min_std=1e-4, max_corr=0.98)
        inf = VBIInference(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=pipeline,
            density_estimator="maf",
            sim_backend="numpy",
            backend="numpy",
            show_progress_bars=False,
        )
        inf.simulate(N_SIM, DURATION, seed=18)
        inf.save(tmp_path / "ckpt.npz")

        pipeline2 = _stat_pipeline()
        assert pipeline2.pruner is None
        VBIInference.load(
            tmp_path / "ckpt.npz",
            sim_spec=_make_spec(),
            pipeline=pipeline2,
            prior=PRIOR,
        )

        assert pipeline2.pruner is not None
        assert pipeline2.pruner.kept_mask_ is not None
