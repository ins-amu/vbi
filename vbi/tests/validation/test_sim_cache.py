"""
Tests for simulation cache - Step 2 of MI6.
"""
import json
import numpy as np
import pytest
from vbi.simulator.spec import MonitorSpec
from vbi.feature_extraction import (
    FeaturePipeline,
    get_features_by_domain,
    get_features_by_given_names,
)
from vbi.inference import BoxUniform, simulate_for_vbi_sweep_cached, extract_from_cache
from .conftest import make_mpr_spec


def _make_spec(n_nodes=4):
    return make_mpr_spec(n_nodes=n_nodes, monitors=(MonitorSpec("tavg", period=1.0),))


def _stat_pipeline(t_cut=0.0):
    cfg = get_features_by_domain(domain="statistical")
    cfg = get_features_by_given_names(cfg, names=["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal="tavg", t_cut=t_cut)


PRIOR = BoxUniform(
    low=np.array([0.1, -6.0]),
    high=np.array([2.0, -3.0]),
    param_names=["G", "eta"],
)
N_SIM = 6
DURATION = 200.0


class TestSimCache:

    def test_cache_files_created(self, tmp_path):
        simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=4,
            sim_backend="numpy",
            seed=0,
        )
        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "chunk_0000.npz").exists()
        assert (tmp_path / "chunk_0001.npz").exists()  # 6 sims / chunk_size=4 → 2 chunks

    def test_metadata_content(self, tmp_path):
        simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=4,
            sim_backend="numpy",
            seed=0,
        )
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["n_samples"] == N_SIM
        assert meta["n_chunks"] == 2
        assert meta["signal"] == "tavg"
        assert meta["param_names"] == ["G", "eta"]

    def test_return_shapes(self, tmp_path):
        theta, x, param_names, feat_labels = simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=N_SIM,  # single chunk
            sim_backend="numpy",
            seed=0,
        )
        assert theta.shape == (N_SIM, 2)
        assert x.shape[0] == N_SIM
        assert x.ndim == 2
        assert x.dtype == np.float64
        assert param_names == ["G", "eta"]
        assert len(feat_labels) > 0

    def test_features_finite(self, tmp_path):
        _, x, _, _ = simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=N_SIM,
            sim_backend="numpy",
            seed=1,
        )
        assert np.all(np.isfinite(x))

    def test_matches_non_cached(self, tmp_path):
        """Cached path returns same theta and same-shaped x as direct sweep."""
        from vbi.inference import simulate_for_vbi_sweep

        theta_cached, x_cached, _, fl_cached = simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=N_SIM,
            sim_backend="numpy",
            seed=42,
        )
        theta_direct, x_direct, _, fl_direct = simulate_for_vbi_sweep(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            sim_backend="numpy",
            seed=42,
        )
        # Same theta drawn from same seed
        np.testing.assert_array_almost_equal(theta_cached, theta_direct)
        # Same feature shape and labels
        assert x_cached.shape == x_direct.shape
        assert fl_cached == fl_direct

    def test_extract_from_cache_same_pipeline(self, tmp_path):
        pipeline = _stat_pipeline()
        simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=pipeline,
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=N_SIM,
            sim_backend="numpy",
            seed=0,
        )
        theta2, x2 = extract_from_cache(tmp_path, pipeline)
        assert theta2.shape == (N_SIM, 2)
        assert x2.shape[0] == N_SIM

    def test_extract_from_cache_different_t_cut(self, tmp_path):
        """Re-extraction with a different t_cut gives different x values."""
        pipeline_a = _stat_pipeline(t_cut=0.0)
        pipeline_b = _stat_pipeline(t_cut=50.0)

        simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=pipeline_a,
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=N_SIM,
            sim_backend="numpy",
            seed=0,
        )
        _, x_a = extract_from_cache(tmp_path, pipeline_a)
        _, x_b = extract_from_cache(tmp_path, pipeline_b)
        assert not np.allclose(x_a, x_b), "Different t_cut should give different features"

    def test_extract_wrong_signal_raises(self, tmp_path):
        pipeline_tavg = _stat_pipeline()  # signal="tavg"
        simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=pipeline_tavg,
            num_simulations=4,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=4,
            sim_backend="numpy",
            seed=0,
        )
        cfg = get_features_by_domain("statistical")
        cfg = get_features_by_given_names(cfg, ["calc_mean"])
        wrong_pipeline = FeaturePipeline(cfg, signal="raw", t_cut=0.0)
        with pytest.raises(ValueError, match="signal"):
            extract_from_cache(tmp_path, wrong_pipeline)

    def test_two_chunk_concatenation(self, tmp_path):
        """Two chunks are concatenated correctly - total rows == N_SIM."""
        theta, x, _, _ = simulate_for_vbi_sweep_cached(
            sim_spec=_make_spec(),
            prior=PRIOR,
            pipeline=_stat_pipeline(),
            num_simulations=N_SIM,
            duration=DURATION,
            cache_dir=tmp_path,
            chunk_size=4,  # forces 2 chunks: 4 + 2
            sim_backend="numpy",
            seed=7,
        )
        assert theta.shape[0] == N_SIM
        assert x.shape[0] == N_SIM
