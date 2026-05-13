"""
M0 validation: NumPy backend — deterministic and stochastic MPR.
Tests correctness of integrators, ring buffer, coupling, monitors.
"""
import numpy as np
import pytest
from vbi.simulator import Simulator
from vbi.simulator.spec import (
    CouplingSpec,
    IntegratorSpec,
    MonitorSpec,
    SimulationSpec,
)
from vbi.simulator.models.mpr import mpr
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

    def test_ring_buffer_delay_matches_tvb_convention(self):
        """
        Confirm the ring buffer follows TVB DenseHistory semantics exactly.

        TVB convention (DenseHistory.query / update):
          - update(step, state): writes buf[step % n]
          - query(step):        reads  buf[(step-1-d) % n]
          - loop order: integrate → write → read-for-next

        VBI convention (read-before-write loop):
          - write(_step, state): buf[_step % h]  then _step += 1
          - read_delayed:        buf[(_step-1-d) % h]  (identical formula)

        Result: coupling at loop iteration s uses state written at s-1-d.
          d=0  → state written one iteration ago  (most recent available)
          d=k  → state written k+1 iterations ago

        Impulse test: write impulse at loop step 0 (goes into buf[0]).
        It should be read when (_step - 1 - d) % h == 0, i.e. _step == d+1,
        which is loop iteration s = d+1 (because write happens at end of each
        iteration and _step == s at the start of iteration s).
        """
        from vbi.simulator.backend.numpy_.history import History
        from vbi.simulator.backend.numpy_.coupling import LinearCoupling
        from vbi.simulator.spec.coupling import CouplingSpec

        d = 15            # delay_steps
        n_nodes, n_cvar = 2, 1
        delay_arr = np.array([[0, d], [d, 0]], dtype=np.int32)
        horizon = int(delay_arr.max()) + 1
        history = History(horizon, n_cvar, n_nodes)
        W = np.array([[0., 1.], [1., 0.]], dtype=np.float64)
        coupling_fn = LinearCoupling(CouplingSpec("linear", a=1.0, b=0.0), W, G=1.0)
        history.initialize(np.array([[0.0, 0.0]]))

        received = []
        for step in range(40):
            source_state = 1.0 if step == 0 else 0.0   # impulse written at step 0
            # Read BEFORE write (TVB-equivalent: coupling reads (step-1-d) slot)
            delayed = history.read_delayed(delay_arr)
            c = coupling_fn.compute(delayed)
            received.append(c[0, 1])
            history.write(np.array([[source_state, 0.0]]))

        # Impulse is in buf[0] after step 0's write.
        # It is read when (_step - 1 - d) % h == 0, i.e. _step == d+1 == 16.
        # _step == 16 at the start of loop iteration s=16.
        arrival_step = d + 1   # TVB convention: delay d → arrives at iteration d+1
        assert received[arrival_step] == 1.0, (
            f"Expected impulse at loop step {arrival_step} (TVB d={d}), "
            f"got {received[arrival_step]:.4f}.\n"
            f"received = {received[:arrival_step+5]}"
        )
        for s, v in enumerate(received):
            if s != arrival_step:
                assert v == 0.0, f"Unexpected coupling at step {s}: {v}"

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
    def test_tvb_noise_style_scales_nsig_to_amplitude(self):
        from vbi.simulator.backend.numpy_.simulator import _resolve_noise_amplitude
        from vbi.simulator.spec import IntegratorSpec
        import dataclasses

        spec = make_mpr_spec(n_nodes=2, dt=0.01, stochastic=True,
                             monitors=(MonitorSpec("raw"),))
        spec = dataclasses.replace(
            spec,
            integrator=dataclasses.replace(spec.integrator, noise_style="tvb"),
        )
        nsig = np.array([0.5, 2.0])
        np.testing.assert_allclose(
            _resolve_noise_amplitude(spec, nsig),
            np.sqrt(2.0 * nsig),
        )

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


# ---------------------------------------------------------------------------
# TVB reference validation
# ---------------------------------------------------------------------------

TVB_MPR_PARAMS = {
    "tau": 1.0,
    "I": 0.0,
    "Delta": 1.0,
    "J": 15.0,
    "eta": -5.0,
    "Gamma": 0.0,
    "cr": 1.0,
    "cv": 0.0,
    "G": 0.4,
}


def _make_vbi_tvb_reference_spec(weights, dt, method):
    params = {name: np.full(weights.shape[0], value)
              for name, value in TVB_MPR_PARAMS.items()
              if name != "G"}
    params["G"] = TVB_MPR_PARAMS["G"]
    return SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method=method, dt=dt),
        coupling=CouplingSpec("linear", a=1.0),
        monitors=(MonitorSpec("raw"),),
        weights=weights,
        tract_lengths=np.zeros_like(weights),
        speed=1.0,
        node_params=params,
    )


def _run_tvb_mpr_reference(weights, dt, method, duration):
    tvb = pytest.importorskip("tvb")
    del tvb  # imported only to trigger pytest's skip message when unavailable

    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.coupling import Linear
    from tvb.simulator.integrators import EulerDeterministic, HeunDeterministic
    from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
    from tvb.simulator.monitors import Raw
    from tvb.simulator.simulator import Simulator as TVBSimulator

    n_nodes = weights.shape[0]
    conn = Connectivity(
        weights=weights,
        tract_lengths=np.zeros_like(weights),
        region_labels=np.array([str(i) for i in range(n_nodes)]),
        centres=np.zeros((n_nodes, 3)),
        speed=np.array([1.0]),
    )
    conn.configure()

    tvb_model = MontbrioPazoRoxin(
        tau=np.array([TVB_MPR_PARAMS["tau"]]),
        I=np.array([TVB_MPR_PARAMS["I"]]),
        Delta=np.array([TVB_MPR_PARAMS["Delta"]]),
        J=np.array([TVB_MPR_PARAMS["J"]]),
        eta=np.array([TVB_MPR_PARAMS["eta"]]),
        Gamma=np.array([TVB_MPR_PARAMS["Gamma"]]),
        cr=np.array([TVB_MPR_PARAMS["cr"]]),
        cv=np.array([TVB_MPR_PARAMS["cv"]]),
    )
    integrator_cls = {
        "euler": EulerDeterministic,
        "heun": HeunDeterministic,
    }[method]
    sim = TVBSimulator(
        connectivity=conn,
        model=tvb_model,
        coupling=Linear(a=np.array([TVB_MPR_PARAMS["G"]])),
        integrator=integrator_cls(dt=dt),
        monitors=[Raw()],
        simulation_length=duration,
    ).configure()

    initial_state = np.zeros((tvb_model.nvar, n_nodes, 1))
    initial_state[0, :, 0] = 0.0
    initial_state[1, :, 0] = -2.0
    sim.current_state[:] = initial_state
    sim.history.buffer[:] = initial_state[tvb_model.cvar][np.newaxis, ...]

    (_times, data), = sim.run()
    return data[:, :, :, 0]


class TestAgainstTVBMPR:
    @pytest.mark.parametrize("method", ["euler", "heun"])
    def test_raw_trajectory_matches_tvb_without_delays(self, method):
        weights = np.array([
            [0.0, 0.25, 0.10],
            [0.50, 0.0, 0.20],
            [0.15, 0.35, 0.0],
        ])
        dt = 0.01
        duration = 0.2

        spec = _make_vbi_tvb_reference_spec(weights, dt, method)
        _times, vbi_data = Simulator(spec, backend="numpy").run(duration)["raw"]
        tvb_data = _run_tvb_mpr_reference(weights, dt, method, duration)

        assert vbi_data.shape == tvb_data.shape
        np.testing.assert_allclose(vbi_data, tvb_data, rtol=2e-10, atol=2e-12)
