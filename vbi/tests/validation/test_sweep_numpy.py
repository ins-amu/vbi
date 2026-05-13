"""
M0 validation: NumPy sweeper — shape, consistency, and FeaturePipeline.
"""
import numpy as np
import pytest
from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import MonitorSpec, SweepSpec
from vbi.feature_extraction.pipeline import FeaturePipeline
from .conftest import make_mpr_spec


# ---------------------------------------------------------------------------
# Sweeper without pipeline
# ---------------------------------------------------------------------------

class TestSweepNoPipeline:
    def test_returns_list_of_dicts(self):
        spec = make_mpr_spec(n_nodes=4, monitors=(MonitorSpec("tavg", period=1.0),))
        sw_spec = SweepSpec(params={"G": np.array([1.0, 2.0, 3.0])})
        results = Sweeper(spec, sw_spec, backend="numpy").run(duration=200.0)
        assert len(results) == 3
        assert all("tavg" in r for r in results)

    def test_grid_sweep_count(self):
        spec = make_mpr_spec(n_nodes=2, monitors=(MonitorSpec("tavg", period=1.0),))
        sw_spec = SweepSpec(params={
            "G":   np.linspace(1.0, 3.0, 4),
            "eta": np.linspace(-5.0, -3.0, 3),
        })
        results = Sweeper(spec, sw_spec, backend="numpy").run(duration=200.0)
        assert len(results) == 4 * 3   # outer product

    def test_arbitrary_samples(self):
        spec = make_mpr_spec(n_nodes=2, monitors=(MonitorSpec("tavg", period=1.0),))
        theta = np.array([[1.0, -4.0], [2.0, -5.0], [3.0, -3.5]])
        sw_spec = SweepSpec(params=theta, param_names=("G", "eta"))
        results = Sweeper(spec, sw_spec, backend="numpy").run(duration=200.0)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Sweeper with FeaturePipeline
# ---------------------------------------------------------------------------

class TestSweepWithPipeline:
    def setup_method(self):
        self.spec = make_mpr_spec(
            n_nodes=4,
            monitors=(MonitorSpec("tavg", period=1.0),),
        )
        self.pipeline = FeaturePipeline(
            features=["mean", "std"],
            signal="tavg",
            t_cut=100.0,
        )
        self.sw_spec = SweepSpec(
            params={"G": np.array([1.0, 2.0, 3.0])},
            pipeline=self.pipeline,
        )

    def test_returns_labels_and_values(self):
        labels, values = Sweeper(
            self.spec, self.sw_spec, backend="numpy"
        ).run(duration=300.0)
        assert isinstance(labels, list)
        assert isinstance(values, np.ndarray)
        # columns: G + mean_0..mean_3 + std_0..std_3 = 1+4+4=9
        assert values.shape == (3, 9)
        assert labels[0] == "G"

    def test_param_values_recorded_correctly(self):
        labels, values = Sweeper(
            self.spec, self.sw_spec, backend="numpy"
        ).run(duration=300.0)
        np.testing.assert_array_equal(
            values[:, 0], [1.0, 2.0, 3.0],
            err_msg="G column must match sweep values"
        )

    def test_run_df_shape(self):
        pd = pytest.importorskip("pandas")
        df = Sweeper(self.spec, self.sw_spec, backend="numpy").run_df(300.0)
        assert df.shape == (3, 9)
        assert "G" in df.columns


# ---------------------------------------------------------------------------
# Consistency: sweep single-param set == single Simulator run
# ---------------------------------------------------------------------------

class TestSweepConsistency:
    def test_single_theta_matches_simulator(self):
        """Sweep with one param set must produce the same features as a
        direct Simulator run with the same parameters."""
        spec = make_mpr_spec(
            n_nodes=4,
            monitors=(MonitorSpec("tavg", period=1.0),),
        )
        pipeline = FeaturePipeline(
            features=["mean"], signal="tavg", t_cut=200.0
        )

        # Sweep: G=2.0 only
        sw_spec = SweepSpec(
            params={"G": np.array([2.0])},
            pipeline=pipeline,
        )
        _, values = Sweeper(spec, sw_spec, backend="numpy").run(duration=500.0)
        sweep_mean = values[0, 1:]   # drop G column

        # Direct run with G=2.0 via node_params override
        from vbi.simulator.spec import SimulationSpec
        direct_spec = SimulationSpec(
            model=spec.model, integrator=spec.integrator,
            coupling=spec.coupling, monitors=spec.monitors,
            weights=spec.weights, tract_lengths=spec.tract_lengths,
            speed=spec.speed,
            node_params={"G": 2.0},
        )
        direct_result = Simulator(direct_spec, backend="numpy").run(500.0)
        _, direct_vals = pipeline.extract(direct_result)

        np.testing.assert_allclose(
            sweep_mean, direct_vals, rtol=1e-10,
            err_msg="Sweep result must match direct Simulator run"
        )
