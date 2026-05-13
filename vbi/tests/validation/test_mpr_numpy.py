"""
M0 validation: NumPy backend — deterministic and stochastic MPR.
Tests correctness of integrators, ring buffer, coupling, monitors.
"""
import numpy as np
import pytest
from vbi.simulator import Simulator
from vbi.simulator.spec import MonitorSpec, IntegratorSpec
from .conftest import make_mpr_spec


# ---------------------------------------------------------------------------
# Deterministic single-node: known fixed point of MPR
# ---------------------------------------------------------------------------

class TestDeterministicSingleNode:
    def setup_method(self):
        self.spec = make_mpr_spec(
            n_nodes=1, dt=0.01, method="heun",
            monitors=(MonitorSpec("tavg", period=1.0),),
        )
        self.sim = Simulator(self.spec, backend="numpy")

    def test_run_returns_expected_keys(self):
        result = self.sim.run(duration=100.0)
        assert "tavg" in result

    def test_tavg_shape(self):
        result = self.sim.run(duration=1000.0)
        t, data = result["tavg"]
        # period=1.0 ms, dt=0.01 ms → istep=100, n_windows=10 000/100=100
        assert t.shape[0] == data.shape[0]
        # data: (n_windows, n_voi, n_nodes)
        assert data.ndim == 3
        assert data.shape[1] == 2   # r and V (all state vars)
        assert data.shape[2] == 1   # single node

    def test_r_stays_nonnegative(self):
        # MPR has lower_bound=0 on r
        result = self.sim.run(duration=2000.0)
        _, data = result["tavg"]
        assert np.all(data[:, 0, :] >= 0.0), "r must be non-negative"

    def test_euler_vs_heun_close(self):
        from .conftest import make_mpr_spec
        spec_euler = make_mpr_spec(n_nodes=1, dt=0.001, method="euler",
                                   monitors=(MonitorSpec("tavg", period=1.0),))
        spec_heun = make_mpr_spec(n_nodes=1, dt=0.001, method="heun",
                                  monitors=(MonitorSpec("tavg", period=1.0),))
        t_e, d_e = Simulator(spec_euler).run(1000.0)["tavg"]
        t_h, d_h = Simulator(spec_heun).run(1000.0)["tavg"]
        # Both should converge to similar trajectories at small dt
        burn = 200   # discard first 200 ms
        np.testing.assert_allclose(
            d_e[burn:, 0, 0], d_h[burn:, 0, 0],
            atol=0.05, err_msg="Euler and Heun diverge too much at dt=0.001"
        )


# ---------------------------------------------------------------------------
# Multi-node: coupling and ring buffer
# ---------------------------------------------------------------------------

class TestMultiNodeDeterministic:
    def test_two_nodes_run(self):
        spec = make_mpr_spec(n_nodes=2, dt=0.01,
                             monitors=(MonitorSpec("tavg", period=1.0),))
        result = Simulator(spec).run(500.0)
        t, data = result["tavg"]
        assert data.shape == (t.shape[0], 2, 2)   # (n_win, n_voi, n_nodes)

    def test_zero_weights_equals_uncoupled(self):
        """With W=0, two independent nodes should have identical trajectories
        from the same initial condition."""
        import copy
        from vbi.simulator.spec import SimulationSpec, CouplingSpec
        spec = make_mpr_spec(n_nodes=2, monitors=(MonitorSpec("raw"),))
        zero_spec = SimulationSpec(
            model=spec.model, integrator=spec.integrator,
            coupling=CouplingSpec("linear", a=1.0),
            monitors=spec.monitors,
            weights=np.zeros((2, 2)),
            tract_lengths=np.zeros((2, 2)),
            speed=spec.speed,
        )
        _, data = Simulator(zero_spec).run(200.0)["raw"]
        # Both nodes start from same default_init → should stay identical
        np.testing.assert_allclose(
            data[:, :, 0], data[:, :, 1], rtol=1e-10,
            err_msg="Uncoupled identical nodes diverged"
        )

    def test_80_nodes_runs_without_error(self):
        spec = make_mpr_spec(n_nodes=80, dt=0.01,
                             monitors=(MonitorSpec("tavg", period=1.0),))
        result = Simulator(spec).run(500.0)
        t, data = result["tavg"]
        assert data.shape[2] == 80
        assert np.isfinite(data).all()


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_stochastic_heun_runs(self):
        spec = make_mpr_spec(n_nodes=2, dt=0.01, stochastic=True,
                             monitors=(MonitorSpec("tavg", period=1.0),))
        result = Simulator(spec).run(500.0)
        _, data = result["tavg"]
        assert np.isfinite(data).all()

    def test_different_seeds_produce_different_trajectories(self):
        from vbi.simulator.spec import IntegratorSpec
        import dataclasses
        spec1 = make_mpr_spec(n_nodes=2, dt=0.01, stochastic=True, seed=0,
                              monitors=(MonitorSpec("raw"),))
        spec2 = dataclasses.replace(
            spec1, integrator=dataclasses.replace(spec1.integrator, noise_seed=99)
        )
        _, d1 = Simulator(spec1).run(200.0)["raw"]
        _, d2 = Simulator(spec2).run(200.0)["raw"]
        assert not np.allclose(d1, d2), "Different seeds produced identical output"

    def test_same_seed_reproduces(self):
        spec = make_mpr_spec(n_nodes=2, dt=0.01, stochastic=True,
                             monitors=(MonitorSpec("raw"),))
        _, d1 = Simulator(spec).run(200.0)["raw"]
        _, d2 = Simulator(spec).run(200.0)["raw"]
        np.testing.assert_array_equal(d1, d2, err_msg="Same seed → must reproduce")


# ---------------------------------------------------------------------------
# Monitor types
# ---------------------------------------------------------------------------

class TestMonitors:
    def test_raw_monitor(self):
        spec = make_mpr_spec(n_nodes=2, dt=0.01,
                             monitors=(MonitorSpec("raw"),))
        result = Simulator(spec).run(100.0)
        t, data = result["raw"]
        # 100 ms / 0.01 ms = 10 000 steps
        assert len(t) == 10_000
        assert data.shape == (10_000, 2, 2)

    def test_subsample_monitor(self):
        spec = make_mpr_spec(n_nodes=2, dt=0.01,
                             monitors=(MonitorSpec("subsample", period=1.0),))
        result = Simulator(spec).run(100.0)
        t, data = result["subsample"]
        # 100 ms / 1.0 ms = 100 samples
        assert len(t) == 100

    def test_tavg_monitor(self):
        spec = make_mpr_spec(n_nodes=2, dt=0.01,
                             monitors=(MonitorSpec("tavg", period=1.0),))
        result = Simulator(spec).run(100.0)
        t, data = result["tavg"]
        assert len(t) == 100

    def test_gavg_shape(self):
        spec = make_mpr_spec(n_nodes=4, dt=0.01,
                             monitors=(MonitorSpec("gavg", period=1.0),))
        result = Simulator(spec).run(100.0)
        t, data = result["gavg"]
        # (n_windows, n_voi, 1)
        assert data.shape[2] == 1

    def test_bold_monitor(self):
        spec = make_mpr_spec(n_nodes=2, dt=0.01,
                             monitors=(MonitorSpec("bold", tr=2000.0),))
        result = Simulator(spec).run(10_000.0)   # 10 s
        t, data = result["bold"]
        # steps 0..999 999; BOLD fires at step>0 and step%200000==0
        # → at steps 200000,400000,600000,800000 = 4 samples
        assert len(t) == 4
        assert data.shape == (4, 2)

    def test_multiple_monitors(self):
        spec = make_mpr_spec(
            n_nodes=2, dt=0.01,
            monitors=(MonitorSpec("tavg", period=1.0),
                      MonitorSpec("bold", tr=2000.0)),
        )
        result = Simulator(spec).run(10_000.0)
        assert "tavg" in result
        assert "bold" in result
