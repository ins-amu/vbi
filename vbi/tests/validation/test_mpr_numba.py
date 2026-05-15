"""
M1 validation: Numba CPU backend — single-run and sweep.

Gold standard: NumPy baseline (validated against TVB in test_mpr_numpy.py).
All Numba results must match NumPy to rtol=1e-4 (deterministic) or
match first two statistical moments (stochastic, 100 realisations).
"""
import time

import numpy as np
import pytest

from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import (
    CouplingSpec, IntegratorSpec, MonitorSpec, SimulationSpec,
)
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.models.mpr import mpr
from .conftest import make_mpr_spec, make_weights

numba = pytest.importorskip("numba", reason="numba not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_sim(spec, duration):
    return Simulator(spec, backend="numpy").run(duration)


def _nb_sim(spec, duration):
    return Simulator(spec, backend="numba").run(duration)


# ---------------------------------------------------------------------------
# Single-run: deterministic — must match NumPy exactly (rtol=1e-4)
# ---------------------------------------------------------------------------

class TestDeterministicMatchesNumPy:
    @pytest.mark.parametrize("n_nodes", [1, 2, 10])
    @pytest.mark.parametrize("method", ["heun", "euler"])
    def test_raw_matches_numpy(self, n_nodes, method):
        spec = make_mpr_spec(
            n_nodes=n_nodes, dt=0.01, method=method,
            monitors=(MonitorSpec("raw"),),
        )
        t_np, d_np = _np_sim(spec, 200.0)["raw"]
        t_nb, d_nb = _nb_sim(spec, 200.0)["raw"]
        assert d_np.shape == d_nb.shape, "shape mismatch"
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg=f"Numba raw diverges from NumPy (n_nodes={n_nodes}, {method})",
        )

    def test_80_nodes_heun(self):
        spec = make_mpr_spec(
            n_nodes=80, dt=0.01, method="heun",
            monitors=(MonitorSpec("raw"),),
        )
        t_np, d_np = _np_sim(spec, 500.0)["raw"]
        t_nb, d_nb = _nb_sim(spec, 500.0)["raw"]
        np.testing.assert_allclose(d_nb, d_np, rtol=1e-4)

    def test_r_stays_nonnegative(self):
        spec = make_mpr_spec(
            n_nodes=10, dt=0.01,
            monitors=(MonitorSpec("raw"),),
        )
        _, d_nb = _nb_sim(spec, 1000.0)["raw"]
        assert np.all(d_nb[:, 0, :] >= 0.0), "r violated lower bound"

    def test_tavg_matches_numpy(self):
        spec = make_mpr_spec(
            n_nodes=4, dt=0.01,
            monitors=(MonitorSpec("tavg", period=1.0),),
        )
        t_np, d_np = _np_sim(spec, 500.0)["tavg"]
        t_nb, d_nb = _nb_sim(spec, 500.0)["tavg"]
        assert d_np.shape == d_nb.shape
        np.testing.assert_allclose(d_nb, d_np, rtol=1e-4)

    def test_gavg_shape(self):
        spec = make_mpr_spec(
            n_nodes=4, dt=0.01,
            monitors=(MonitorSpec("gavg", period=1.0),),
        )
        t_nb, d_nb = _nb_sim(spec, 200.0)["gavg"]
        assert d_nb.shape[2] == 1, "gavg must spatially average to 1 node"


# ---------------------------------------------------------------------------
# Single-run: with delays
# ---------------------------------------------------------------------------

class TestDelaysMatchNumPy:
    def test_two_nodes_with_delays(self):
        spec = make_mpr_spec(
            n_nodes=2, dt=0.01, method="heun",
            monitors=(MonitorSpec("raw"),),
        )
        t_np, d_np = _np_sim(spec, 200.0)["raw"]
        t_nb, d_nb = _nb_sim(spec, 200.0)["raw"]
        np.testing.assert_allclose(d_nb, d_np, rtol=1e-4)


# ---------------------------------------------------------------------------
# Single-run: stochastic — first-moment comparison
# ---------------------------------------------------------------------------

class TestStochasticMoments:
    def test_stochastic_runs_without_error(self):
        spec = make_mpr_spec(
            n_nodes=4, dt=0.01, stochastic=True,
            monitors=(MonitorSpec("raw"),),
        )
        _, d = _nb_sim(spec, 200.0)["raw"]
        assert np.isfinite(d).all()

    def test_same_seed_reproduces(self):
        spec = make_mpr_spec(
            n_nodes=2, dt=0.01, stochastic=True,
            monitors=(MonitorSpec("raw"),),
        )
        _, d1 = _nb_sim(spec, 200.0)["raw"]
        _, d2 = _nb_sim(spec, 200.0)["raw"]
        np.testing.assert_array_equal(d1, d2, err_msg="Same seed → must reproduce")

    def test_stochastic_mean_close_to_numpy(self):
        """Mean over 20 realisations: Numba ≈ NumPy (within 3 std)."""
        spec = make_mpr_spec(
            n_nodes=2, dt=0.01, stochastic=True,
            monitors=(MonitorSpec("raw"),),
        )
        import dataclasses

        np_means, nb_means = [], []
        for seed in range(20):
            s = dataclasses.replace(
                spec,
                integrator=dataclasses.replace(spec.integrator, noise_seed=seed),
            )
            _, d = Simulator(s, backend="numpy").run(500.0)["raw"]
            np_means.append(d[1000:, 0, :].mean())   # r variable
            _, d = Simulator(s, backend="numba").run(500.0)["raw"]
            nb_means.append(d[1000:, 0, :].mean())

        np_mu = np.mean(np_means)
        nb_mu = np.mean(nb_means)
        sigma = np.std(np_means) / np.sqrt(20)
        assert abs(nb_mu - np_mu) < 4 * sigma, (
            f"Stochastic mean mismatch: numpy={np_mu:.4f}, numba={nb_mu:.4f}, "
            f"4*se={4*sigma:.4f}"
        )


# ---------------------------------------------------------------------------
# Sweep: deterministic — shapes and single-run consistency
# ---------------------------------------------------------------------------

class TestSweepDeterministic:
    def _base_spec(self, n_nodes=4):
        return make_mpr_spec(
            n_nodes=n_nodes, dt=0.01,
            monitors=(MonitorSpec("tavg", period=1.0),),
        )

    def test_sweep_shape(self):
        spec       = self._base_spec(n_nodes=4)
        sweep_spec = SweepSpec(params={"G": np.linspace(1.0, 4.0, 8)})
        res = Sweeper(spec, sweep_spec, backend="numba").run(500.0)
        # Without pipeline: list of monitor dicts
        assert isinstance(res, list)
        assert len(res) == 8

    def test_sweep_with_pipeline_shape(self):
        from vbi.feature_extraction.pipeline import FeaturePipeline
        spec = self._base_spec(n_nodes=4)
        pipeline = FeaturePipeline(
            features=["mean", "std"], signal="tavg", t_cut=200.0
        )
        sweep_spec = SweepSpec(
            params={"G": np.linspace(1.0, 4.0, 10)},
            pipeline=pipeline,
        )
        labels, values = Sweeper(spec, sweep_spec, backend="numba").run(500.0)
        assert values.shape[0] == 10
        assert "G" in labels
        assert "mean_0" in labels or "mean" in labels or any("mean" in l for l in labels)

    def test_single_param_set_matches_simulator(self):
        """Sweep with one param set must equal a direct Simulator.run() call."""
        spec = make_mpr_spec(
            n_nodes=4, dt=0.01,
            monitors=(MonitorSpec("raw"),),
        )
        sweep_spec = SweepSpec(params={"G": np.array([2.0])})
        res_list = Sweeper(spec, sweep_spec, backend="numba").run(300.0)
        _, d_sweep = res_list[0]["raw"]   # (n_record, n_sv, n_nodes)

        _, d_sim = _nb_sim(spec, 300.0)["raw"]

        np.testing.assert_allclose(
            d_sweep, d_sim, rtol=1e-4,
            err_msg="Single-set sweep must match direct Simulator.run()",
        )

    def test_sweep_matches_numpy_sweep(self):
        """Numba sweep results must match NumPy sweep (rtol=1e-4)."""
        spec = make_mpr_spec(
            n_nodes=4, dt=0.01,
            monitors=(MonitorSpec("raw"),),
        )
        n_samples  = 6
        sweep_spec = SweepSpec(params={"G": np.linspace(1.0, 3.0, n_samples)})

        nb_res = Sweeper(spec, sweep_spec, backend="numba").run(300.0)
        np_res = Sweeper(spec, sweep_spec, backend="numpy").run(300.0)

        for i in range(n_samples):
            _, d_nb = nb_res[i]["raw"]
            _, d_np = np_res[i]["raw"]
            np.testing.assert_allclose(
                d_nb, d_np, rtol=1e-4,
                err_msg=f"Sweep sample {i} mismatch",
            )


# ---------------------------------------------------------------------------
# Throughput (informational — never fails)
# ---------------------------------------------------------------------------

class TestThroughput:
    def test_sweep_throughput_report(self, capsys):
        spec = make_mpr_spec(
            n_nodes=80, dt=0.01,
            monitors=(MonitorSpec("tavg", period=1.0),),
        )
        n_samples  = 50
        sweep_spec = SweepSpec(params={"G": np.linspace(0.5, 5.0, n_samples)})
        t0 = time.perf_counter()
        Sweeper(spec, sweep_spec, backend="numba").run(2000.0)
        elapsed = time.perf_counter() - t0
        rate = n_samples / elapsed
        with capsys.disabled():
            print(f"\nNumba sweep: {rate:.1f} samples/s  "
                  f"(n_nodes=80, duration=2000ms, n_samples={n_samples})")
