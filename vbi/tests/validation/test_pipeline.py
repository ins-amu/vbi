"""
Validation tests for FeaturePipeline — cfg-dict integration with Simulator/Sweeper.
"""
import numpy as np
import pytest
from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import MonitorSpec, SweepSpec
from vbi.feature_extraction import (
    FeaturePipeline,
    get_features_by_domain,
    get_features_by_given_names,
    update_cfg,
)
from .conftest import make_mpr_spec


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _stat_pipeline(signal="tavg", t_cut=0.0):
    """FeaturePipeline with calc_mean + calc_std from the statistical domain."""
    cfg = get_features_by_domain(domain="statistical")
    cfg = get_features_by_given_names(cfg, names=["calc_mean", "calc_std"])
    return FeaturePipeline(cfg, signal=signal, t_cut=t_cut)


def _make_spec(n_nodes=4):
    return make_mpr_spec(n_nodes=n_nodes, monitors=(MonitorSpec("tavg", period=1.0),))


# ---------------------------------------------------------------------------
# Single Simulator run
# ---------------------------------------------------------------------------

class TestPipelineSingleRun:
    def test_extract_returns_labels_and_1d_array(self):
        spec = _make_spec()
        result = Simulator(spec, backend="numpy").run(200.0)
        pipeline = _stat_pipeline()
        labels, values = pipeline.extract(result)
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert isinstance(values, np.ndarray)
        assert values.ndim == 1
        assert values.dtype == np.float64
        assert len(labels) == len(values)

    def test_extract_values_finite(self):
        spec = _make_spec()
        result = Simulator(spec, backend="numpy").run(200.0)
        _, values = _stat_pipeline().extract(result)
        assert np.all(np.isfinite(values)), "Feature values must be finite"

    def test_extract_df_shape_and_columns(self):
        pd = pytest.importorskip("pandas")
        spec = _make_spec()
        result = Simulator(spec, backend="numpy").run(200.0)
        pipeline = _stat_pipeline()
        df = pipeline.extract_df(result)
        assert df.shape[0] == 1
        labels, _ = pipeline.extract(result)
        assert list(df.columns) == labels

    def test_t_cut_changes_values(self):
        """Features computed with t_cut=0 and t_cut=100 ms must differ."""
        spec = _make_spec()
        result = Simulator(spec, backend="numpy").run(500.0)
        _, v0 = _stat_pipeline(t_cut=0.0).extract(result)
        _, v1 = _stat_pipeline(t_cut=100.0).extract(result)
        assert not np.allclose(v0, v1), "t_cut must affect the extracted features"

    def test_update_cfg_param_accepted(self):
        """update_cfg must flow through to calc_features without error."""
        cfg = get_features_by_domain(domain="statistical")
        cfg = get_features_by_given_names(cfg, names=["calc_mean"])
        cfg = update_cfg(cfg, "calc_mean", {"indices": None})
        pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=0.0)
        spec = _make_spec()
        result = Simulator(spec, backend="numpy").run(200.0)
        labels, values = pipeline.extract(result)
        assert len(values) > 0


# ---------------------------------------------------------------------------
# NumPy sweeper with FeaturePipeline
# ---------------------------------------------------------------------------

class TestPipelineNumpySweep:
    def setup_method(self):
        self.spec = _make_spec(n_nodes=4)
        self.pipeline = _stat_pipeline(signal="tavg", t_cut=50.0)
        self.cr_vals = np.array([0.2, 0.5, 0.8])
        self.sw_spec = SweepSpec(
            params={"cr": self.cr_vals},
            pipeline=self.pipeline,
        )

    def test_returns_labels_and_values(self):
        labels, values = Sweeper(
            self.spec, self.sw_spec, backend="numpy"
        ).run(duration=200.0)
        assert isinstance(labels, list)
        assert isinstance(values, np.ndarray)
        assert values.shape[0] == 3          # one row per cr value
        assert labels[0] == "cr"
        assert values.shape[1] == len(labels)

    def test_param_column_matches_sweep_values(self):
        labels, values = Sweeper(
            self.spec, self.sw_spec, backend="numpy"
        ).run(duration=200.0)
        np.testing.assert_array_equal(values[:, 0], self.cr_vals)

    def test_run_df_columns(self):
        pd = pytest.importorskip("pandas")
        df = Sweeper(self.spec, self.sw_spec, backend="numpy").run_df(200.0)
        assert df.shape[0] == 3
        assert "cr" in df.columns
        assert df.shape[1] > 1   # at least cr + one feature

    def test_feature_values_finite(self):
        _, values = Sweeper(
            self.spec, self.sw_spec, backend="numpy"
        ).run(duration=200.0)
        assert np.all(np.isfinite(values)), "All sweep feature values must be finite"


# ---------------------------------------------------------------------------
# Numba sweeper with FeaturePipeline
# ---------------------------------------------------------------------------

class TestPipelineNumbaSweep:
    def setup_method(self):
        self.spec = _make_spec(n_nodes=4)
        self.pipeline = _stat_pipeline(signal="tavg", t_cut=50.0)
        self.cr_vals = np.array([0.2, 0.5, 0.8])
        self.sw_spec = SweepSpec(
            params={"cr": self.cr_vals},
            pipeline=self.pipeline,
        )

    def test_returns_labels_and_values(self):
        labels, values = Sweeper(
            self.spec, self.sw_spec, backend="numba"
        ).run(duration=200.0)
        assert isinstance(labels, list)
        assert values.shape[0] == 3
        assert labels[0] == "cr"

    def test_param_column_matches_sweep_values(self):
        _, values = Sweeper(
            self.spec, self.sw_spec, backend="numba"
        ).run(duration=200.0)
        np.testing.assert_array_equal(values[:, 0], self.cr_vals)

    def test_feature_values_finite(self):
        _, values = Sweeper(
            self.spec, self.sw_spec, backend="numba"
        ).run(duration=200.0)
        assert np.all(np.isfinite(values))


# ---------------------------------------------------------------------------
# Consistency: NumPy vs Numba sweep, same pipeline
# ---------------------------------------------------------------------------

class TestPipelineBackendConsistency:
    def test_numpy_numba_feature_labels_match(self):
        """Both backends must produce the same labels."""
        spec = _make_spec(n_nodes=4)
        pipeline = _stat_pipeline()
        sw_spec = SweepSpec(
            params={"cr": np.array([0.5])},
            pipeline=pipeline,
        )
        np_labels, _ = Sweeper(spec, sw_spec, backend="numpy").run(200.0)
        nb_labels, _ = Sweeper(spec, sw_spec, backend="numba").run(200.0)
        assert np_labels == nb_labels

    def test_single_run_labels_match_sweep_labels(self):
        """Labels from pipeline.extract on a single run must equal sweep labels."""
        spec = _make_spec(n_nodes=4)
        pipeline = _stat_pipeline(t_cut=50.0)
        single_result = Simulator(spec, backend="numpy").run(200.0)
        single_labels, _ = pipeline.extract(single_result)

        sw_spec = SweepSpec(
            params={"cr": np.array([0.5])},
            pipeline=pipeline,
        )
        sweep_labels, _ = Sweeper(spec, sw_spec, backend="numpy").run(200.0)
        # sweep_labels includes "cr" as first entry; rest must match
        assert sweep_labels[1:] == single_labels
