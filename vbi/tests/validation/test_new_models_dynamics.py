"""
Dynamical behaviour tests for the new models.

These tests verify that each model produces the correct physics:
  - known fixed points are reached (or not, depending on regime)
  - limit cycles have the correct amplitude/frequency
  - bounds are respected throughout
  - node-heterogeneous parameters work
  - key bifurcation parameters shift the regime as expected

All tests use the NumPy backend (backend="numpy") so they are independent
of Numba/C++ compilation.
"""
import dataclasses

import numpy as np
import pytest

from vbi.simulator import Simulator
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
)
from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator
from vbi.simulator.models.kuramoto import kuramoto
from vbi.simulator.models.sup_hopf import sup_hopf
from vbi.simulator.models.linear import linear
from vbi.simulator.models.larter_breakspear import larter_breakspear
from vbi.simulator.models.coombes_byrne_2d import coombes_byrne_2d
from vbi.simulator.models.gast_sd import gast_sd
from vbi.simulator.models.gast_sf import gast_sf

pytestmark = pytest.mark.slow



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isolated_spec(model, n_nodes=1, dt=0.01, node_params=None):
    """Fully disconnected (W=0) single or multi-node spec."""
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
        coupling=CouplingSpec("linear", a=0.0),
        monitors=(MonitorSpec("raw"),),
        weights=np.zeros((n_nodes, n_nodes)),
        node_params=node_params or {},
    )


def _run(spec, duration):
    return Simulator(spec, backend="numpy").run(duration)["raw"]


def _last(t, d):
    """Return the last recorded state (n_sv, n_nodes)."""
    return d[-1]


# ---------------------------------------------------------------------------
# Linear model — exact analytical solution
# ---------------------------------------------------------------------------

class TestLinearDynamics:
    def test_exponential_decay(self):
        """x(t) = x₀ · exp(γ·t) — numerically exact for small dt."""
        gamma = -5.0
        x0 = 0.5    # within bounds [-1, 1]; 2.0 would be clamped to 1.0
        dt = 0.001
        duration = 1.0
        spec = dataclasses.replace(
            _isolated_spec(linear),
            integrator=IntegratorSpec(method="heun", dt=dt),
            node_params={"gamma": gamma},
        )
        # Force initial state by overriding default_init
        from vbi.simulator.models.linear import linear as lin_model
        spec2 = SimulationSpec(
            model=dataclasses.replace(
                lin_model,
                state_variables=(
                    dataclasses.replace(lin_model.state_variables[0], default_init=x0),
                ),
            ),
            integrator=IntegratorSpec(method="heun", dt=dt),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"gamma": gamma},
        )
        t, d = _run(spec2, duration)
        x_numerical = d[-1, 0, 0]
        x_analytical = x0 * np.exp(gamma * duration)
        np.testing.assert_allclose(x_numerical, x_analytical, rtol=1e-3,
                                   err_msg="Linear: numerical solution deviates from exp decay")

    def test_decays_to_zero(self):
        """x must approach 0 for gamma < 0, zero coupling."""
        spec = _isolated_spec(linear, node_params={"gamma": -10.0})
        spec2 = SimulationSpec(
            model=dataclasses.replace(
                linear,
                state_variables=(
                    dataclasses.replace(linear.state_variables[0], default_init=1.0),
                ),
            ),
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"gamma": -10.0},
        )
        t, d = _run(spec2, 5.0)
        assert abs(d[-1, 0, 0]) < 1e-3, "Linear: x did not decay to 0"

    def test_x_stays_in_bounds(self):
        """x must never exceed [-1, 1] bounds."""
        spec = _isolated_spec(linear, n_nodes=5)
        t, d = _run(spec, 10.0)
        assert np.all(d[:, 0, :] >= -1.0)
        assert np.all(d[:, 0, :] <= 1.0)


# ---------------------------------------------------------------------------
# Kuramoto — phase evolves at natural frequency (uncoupled)
# ---------------------------------------------------------------------------

class TestKuramotoDynamics:
    def test_phase_evolves_at_omega(self):
        """Uncoupled theta increases at rate omega (no coupling, no noise)."""
        omega = 2.0
        dt = 0.01
        duration = 5.0
        spec = SimulationSpec(
            model=dataclasses.replace(
                kuramoto,
                state_variables=(
                    dataclasses.replace(kuramoto.state_variables[0], default_init=0.0),
                ),
            ),
            integrator=IntegratorSpec(method="heun", dt=dt),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"omega": omega},
        )
        t, d = _run(spec, duration)
        theta_numerical = d[-1, 0, 0]
        theta_analytical = omega * duration   # theta(T) = omega*T + 0
        np.testing.assert_allclose(theta_numerical, theta_analytical, rtol=1e-4,
                                   err_msg="Kuramoto: theta does not evolve at rate omega")

    def test_heterogeneous_omega(self):
        """Nodes with different omega diverge in phase as expected."""
        n = 3
        omegas = np.array([1.0, 2.0, 3.0])
        duration = 2.0
        spec = SimulationSpec(
            model=dataclasses.replace(
                kuramoto,
                state_variables=(
                    dataclasses.replace(kuramoto.state_variables[0], default_init=0.0),
                ),
            ),
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((n, n)),
            node_params={"omega": omegas},
        )
        t, d = _run(spec, duration)
        for i, om in enumerate(omegas):
            np.testing.assert_allclose(d[-1, 0, i], om * duration, rtol=1e-3,
                                       err_msg=f"Kuramoto node {i}: wrong phase")

    def test_finite_and_monotone(self):
        """Theta must be finite and strictly increasing (omega > 0, no coupling)."""
        spec = _isolated_spec(kuramoto)
        t, d = _run(spec, 10.0)
        assert np.isfinite(d).all()
        # theta should be strictly increasing (omega=1.0 by default)
        assert np.all(np.diff(d[:, 0, 0]) > 0), "Kuramoto: theta not monotonically increasing"


# ---------------------------------------------------------------------------
# Kuramoto — sinusoidal coupling (kind='kuramoto')
# ---------------------------------------------------------------------------

class TestKuramotoCoupling:
    """Physics tests for the proper sinusoidal Kuramoto coupling."""

    def _spec(self, n, omega, G, theta0=None, dt=0.01,
              tract_lengths=None, speed=1.0):
        W = np.ones((n, n), dtype=float)
        np.fill_diagonal(W, 0.0)
        import dataclasses
        if theta0 is not None:
            sv = (dataclasses.replace(kuramoto.state_variables[0],
                                      default_init=theta0),)
            model = dataclasses.replace(kuramoto, state_variables=sv)
        else:
            model = kuramoto
        return SimulationSpec(
            model=model,
            integrator=IntegratorSpec(method="heun", dt=dt),
            coupling=CouplingSpec("kuramoto"),
            monitors=(MonitorSpec("raw"),),
            weights=W,
            tract_lengths=tract_lengths,
            speed=speed,
            node_params={"omega": np.full(n, float(omega)), "G": G},
        )

    def _run(self, spec, duration):
        return Simulator(spec, backend="numpy").run(duration)["raw"]

    def test_synchronization_identical_omega(self):
        """Identical-frequency nodes with strong coupling must phase-lock (R → 1)."""
        n = 6
        theta0 = np.random.default_rng(0).uniform(-np.pi, np.pi, n)
        spec = self._spec(n, omega=1.0, G=5.0, theta0=theta0)
        t, d = self._run(spec, 500.0)
        R = np.abs(np.exp(1j * d[-1, 0, :]).mean())
        assert R > 0.99, f"Identical-omega Kuramoto did not synchronize: R={R:.4f}"

    def test_zero_G_equals_uncoupled(self):
        """G=0 sinusoidal coupling must give the same trajectory as linear a=0."""
        import dataclasses
        n, dt, duration = 3, 0.01, 5.0
        theta0 = np.random.default_rng(1).uniform(-np.pi, np.pi, n)
        omega  = np.array([1.0, 1.5, 2.0])
        W = np.ones((n, n), dtype=float); np.fill_diagonal(W, 0.0)
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)

        def _run(kind, G, a=0.0):
            coup = CouplingSpec(kind, a=a) if kind == "linear" else CouplingSpec(kind)
            spec = SimulationSpec(
                model=mdl,
                integrator=IntegratorSpec(method="heun", dt=dt),
                coupling=coup,
                monitors=(MonitorSpec("raw"),),
                weights=W,
                node_params={"omega": omega, "G": G},
            )
            return Simulator(spec, backend="numpy").run(duration)["raw"][1]

        np.testing.assert_allclose(
            _run("kuramoto", G=0.0), _run("linear", G=0.0, a=0.0),
            rtol=1e-12, err_msg="G=0 Kuramoto must equal uncoupled linear")

    def test_coupling_pulls_lagging_node(self):
        """A node that lags behind all others must increase its rate (positive c_i)."""
        import dataclasses
        n = 4
        # Node 0 lags by π/2; others are ahead
        theta0 = np.array([0.0, np.pi / 2, np.pi / 2, np.pi / 2])
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)
        spec = SimulationSpec(
            model=mdl,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("kuramoto"),
            monitors=(MonitorSpec("raw"),),
            weights=np.ones((n, n), dtype=float) - np.eye(n),
            node_params={"omega": np.ones(n), "G": 1.0},
        )
        t, d = self._run(spec, 0.1)
        # In first step, coupling for node 0 should be positive (pulled forward)
        theta_step1 = d[0, 0, :]
        phase_advance_node0  = theta_step1[0] - theta0[0]
        phase_advance_others = (theta_step1[1:] - theta0[1:]).mean()
        assert phase_advance_node0 > phase_advance_others, \
            "Lagging node should advance faster than leading nodes"

    def test_numpy_numba_match(self):
        """NumPy and Numba backends must produce identical results."""
        pytest.importorskip("numba")
        n = 5
        theta0 = np.random.default_rng(7).uniform(-np.pi, np.pi, n)
        spec = self._spec(n, omega=1.0, G=0.5, theta0=theta0)
        _, d_np = Simulator(spec, backend="numpy").run(50.0)["raw"]
        _, d_nb = Simulator(spec, backend="numba").run(50.0)["raw"]
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg="Numba Kuramoto (sinusoidal) diverges from NumPy")

    def test_delayed_numpy_numba_match(self):
        """Delayed sinusoidal Kuramoto: NumPy and Numba must agree."""
        pytest.importorskip("numba")
        import dataclasses
        n = 3
        theta0 = np.random.default_rng(9).uniform(-np.pi, np.pi, n)
        W = np.ones((n, n), dtype=float); np.fill_diagonal(W, 0.0)
        tract = np.full((n, n), 5.0); np.fill_diagonal(tract, 0.0)
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)
        spec = SimulationSpec(
            model=mdl,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("kuramoto"),
            monitors=(MonitorSpec("raw"),),
            weights=W, tract_lengths=tract, speed=1.0,
            node_params={"omega": np.ones(n), "G": 0.5},
        )
        _, d_np = Simulator(spec, backend="numpy").run(30.0)["raw"]
        _, d_nb = Simulator(spec, backend="numba").run(30.0)["raw"]
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg="Delayed Numba Kuramoto diverges from NumPy")


# ---------------------------------------------------------------------------
# SupHopf — fixed point vs. limit cycle
# ---------------------------------------------------------------------------

class TestSupHopfDynamics:
    def test_stable_fixed_point_at_negative_a(self):
        """a < 0 → damped oscillation, (x,y) → (0,0)."""
        spec = SimulationSpec(
            model=sup_hopf,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"a": -2.0, "omega": 1.0},
        )
        # Start off-centre
        spec2 = SimulationSpec(
            model=dataclasses.replace(
                sup_hopf,
                state_variables=(
                    dataclasses.replace(sup_hopf.state_variables[0], default_init=1.0),
                    dataclasses.replace(sup_hopf.state_variables[1], default_init=0.5),
                ),
            ),
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"a": -2.0, "omega": 1.0},
        )
        t, d = _run(spec2, 50.0)
        r = np.sqrt(d[-1, 0, 0]**2 + d[-1, 1, 0]**2)
        assert r < 0.01, f"SupHopf: expected convergence to 0 for a<0, got r={r:.4f}"

    def test_limit_cycle_radius_at_positive_a(self):
        """a > 0 → stable limit cycle with radius = sqrt(a)."""
        a_val = 1.5
        spec = SimulationSpec(
            model=dataclasses.replace(
                sup_hopf,
                state_variables=(
                    dataclasses.replace(sup_hopf.state_variables[0], default_init=0.1),
                    dataclasses.replace(sup_hopf.state_variables[1], default_init=0.0),
                ),
            ),
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"a": a_val, "omega": 1.0},
        )
        t, d = _run(spec, 500.0)
        # Measure amplitude over last 20% of trajectory
        last = d[int(0.8 * len(t)):]
        r = np.sqrt(last[:, 0, 0]**2 + last[:, 1, 0]**2)
        r_expected = np.sqrt(a_val)
        np.testing.assert_allclose(r.mean(), r_expected, rtol=0.02,
                                   err_msg=f"SupHopf: limit cycle radius wrong for a={a_val}")

    def test_two_cvar_coupling_accepted(self):
        """SupHopf uses cvar=(x,y) — verify both coupling channels are used."""
        assert sup_hopf.cvar == ("x", "y")
        assert len(sup_hopf.cvar_indices) == 2


# ---------------------------------------------------------------------------
# Generic2dOscillator — fixed points and limit cycles
# ---------------------------------------------------------------------------

class TestGeneric2dOscillatorDynamics:
    def test_convergence_to_fixed_point_default_params(self):
        """Default a=-2.0: excitable regime, trajectory converges to a fixed point."""
        spec = SimulationSpec(
            model=generic_2d_oscillator,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
        )
        t, d = _run(spec, 2000.0)
        # Check convergence: last 10% of trajectory should have small variance
        last = d[int(0.9 * len(t)):]
        var = last.var(axis=0)
        assert np.all(var < 1e-4), \
            f"Generic2dOscillator: state not converged, variance={var}"

    def test_fixed_point_location(self):
        """Default params (a=-2): unique stable FP at (V≈-0.1887, W≈-0.1135).

        The cubic V³ - 3V² + 10V + 2 = 0 has exactly one real root at V≈-0.1887.
        TVB's models_test quotes a different value because it uses a different
        initial condition and a very short simulation (20 ms, n_step=2000, dt=0.01)
        that has not yet converged.
        """
        spec = SimulationSpec(
            model=generic_2d_oscillator,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
        )
        t, d = _run(spec, 5000.0)
        V_fp = d[-1, 0, 0]
        W_fp = d[-1, 1, 0]
        np.testing.assert_allclose(V_fp, -0.18865, rtol=1e-3,
                                   err_msg="Generic2dOscillator: V fixed-point off")
        np.testing.assert_allclose(W_fp, -0.11348, rtol=1e-3,
                                   err_msg="Generic2dOscillator: W fixed-point off")

    def test_limit_cycle_with_positive_a(self):
        """a=2.0 → limit cycle; trajectory must not converge to a fixed point."""
        spec = SimulationSpec(
            model=generic_2d_oscillator,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"a": 2.0},
        )
        t, d = _run(spec, 500.0)
        last = d[int(0.8 * len(t)):]
        V_range = last[:, 0, 0].max() - last[:, 0, 0].min()
        assert V_range > 0.5, \
            f"Generic2dOscillator a=2: expected oscillation, got V range {V_range:.4f}"

    def test_coupling_shifts_trajectory(self):
        """Nonzero coupling (via coupling.b offset) shifts the trajectory."""
        spec_no_coup = SimulationSpec(
            model=generic_2d_oscillator,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=0.0, b=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
        )
        spec_with_coup = SimulationSpec(
            model=generic_2d_oscillator,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=0.0, b=1.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
        )
        _, d0 = _run(spec_no_coup, 2000.0)
        _, d1 = _run(spec_with_coup, 2000.0)
        assert not np.allclose(d0[-1], d1[-1]), \
            "Generic2dOscillator: coupling b offset had no effect on fixed point"

    def test_node_heterogeneous_I(self):
        """Per-node I parameter shifts individual node trajectories."""
        n = 3
        spec = SimulationSpec(
            model=generic_2d_oscillator,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((n, n)),
            node_params={"I": np.array([0.0, 0.5, 1.0])},
        )
        t, d = _run(spec, 2000.0)
        V_fps = d[-1, 0, :]   # (n_nodes,)
        # Different I → different fixed points
        assert not np.allclose(V_fps[0], V_fps[1]), \
            "Heterogeneous I: nodes 0 and 1 converged to same FP"
        assert not np.allclose(V_fps[1], V_fps[2]), \
            "Heterogeneous I: nodes 1 and 2 converged to same FP"


# ---------------------------------------------------------------------------
# LarterBreakspear — regime switch via d_V
# ---------------------------------------------------------------------------

class TestLarterBreakspearDynamics:
    def test_fixed_point_regime(self):
        """d_V=0.5 (<0.55): single fixed point, no oscillations."""
        spec = SimulationSpec(
            model=larter_breakspear,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"d_V": 0.5},
        )
        t, d = _run(spec, 500.0)
        last = d[int(0.8 * len(t)):]
        V_range = last[:, 0, 0].max() - last[:, 0, 0].min()
        assert V_range < 0.05, \
            f"LarterBreakspear d_V=0.5: expected fixed point, got V range {V_range:.4f}"

    def test_limit_cycle_regime(self):
        """d_V=0.57 (0.55–0.59): limit cycle — must show sustained oscillations."""
        spec = SimulationSpec(
            model=larter_breakspear,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"d_V": 0.57},
        )
        t, d = _run(spec, 2000.0)
        last = d[int(0.7 * len(t)):]
        V_range = last[:, 0, 0].max() - last[:, 0, 0].min()
        assert V_range > 0.1, \
            f"LarterBreakspear d_V=0.57: expected oscillation, got V range {V_range:.4f}"

    def test_state_stays_finite(self):
        """Default parameters: all state variables remain finite."""
        spec = SimulationSpec(
            model=larter_breakspear,
            integrator=IntegratorSpec(method="heun", dt=0.05),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((3, 3)),
        )
        t, d = _run(spec, 100.0)
        assert np.isfinite(d).all(), "LarterBreakspear: non-finite output"


# ---------------------------------------------------------------------------
# CoombesByrne2D — r must stay non-negative
# ---------------------------------------------------------------------------

class TestCoombesByrne2DDynamics:
    def test_r_stays_nonnegative(self):
        """r (firing rate) must remain ≥ 0 at all times."""
        spec = SimulationSpec(
            model=coombes_byrne_2d,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.05),
            monitors=(MonitorSpec("raw"),),
            weights=np.abs(np.random.default_rng(0).standard_normal((6, 6))) * 0.1,
        )
        t, d = _run(spec, 100.0)
        assert np.all(d[:, 0, :] >= 0.0), "CoombesByrne2D: r went negative"

    def test_positive_delta_keeps_r_positive(self):
        """For Delta > 0, the restoring force dr/dt|_{r=0} = Delta/pi > 0."""
        # This tests the physics: even if started close to 0, r stays positive
        spec = SimulationSpec(
            model=dataclasses.replace(
                coombes_byrne_2d,
                state_variables=(
                    dataclasses.replace(coombes_byrne_2d.state_variables[0], default_init=1e-6),
                    dataclasses.replace(coombes_byrne_2d.state_variables[1], default_init=0.0),
                ),
            ),
            integrator=IntegratorSpec(method="heun", dt=0.001),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
            node_params={"Delta": 1.0},
        )
        t, d = _run(spec, 5.0)
        assert np.all(d[:, 0, :] >= 0.0), \
            "CoombesByrne2D: r went negative even with Delta > 0"

    def test_coupling_enters_v_equation(self):
        """Coupling must affect V but not r directly."""
        n = 3
        rng = np.random.default_rng(1)
        W = np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(W, 0.0)

        spec_no_coup = SimulationSpec(
            model=coombes_byrne_2d,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=W,
        )
        spec_with_coup = SimulationSpec(
            model=coombes_byrne_2d,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.5),
            monitors=(MonitorSpec("raw"),),
            weights=W,
        )
        _, d0 = _run(spec_no_coup, 20.0)
        _, d1 = _run(spec_with_coup, 20.0)
        assert not np.allclose(d0[-1, 1, :], d1[-1, 1, :]), \
            "CoombesByrne2D: V unaffected by coupling"


# ---------------------------------------------------------------------------
# GastSD / GastSF — adaptation dynamics
# ---------------------------------------------------------------------------

class TestGastAdaptationDynamics:
    def test_gast_sd_r_nonnegative(self):
        """r ≥ 0 throughout simulation for GastSD."""
        spec = SimulationSpec(
            model=gast_sd,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((4, 4)),
        )
        t, d = _run(spec, 50.0)
        assert np.all(d[:, 0, :] >= 0.0), "GastSD: r went negative"

    def test_gast_sf_r_nonnegative(self):
        """r ≥ 0 throughout simulation for GastSF."""
        spec = SimulationSpec(
            model=gast_sf,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((4, 4)),
        )
        t, d = _run(spec, 50.0)
        assert np.all(d[:, 0, :] >= 0.0), "GastSF: r went negative"

    def test_gast_sd_vs_sf_differ(self):
        """SD and SF produce different trajectories (different V equations)."""
        def _run_model(model):
            spec = SimulationSpec(
                model=model,
                integrator=IntegratorSpec(method="heun", dt=0.01),
                coupling=CouplingSpec("linear", a=0.0),
                monitors=(MonitorSpec("raw"),),
                weights=np.zeros((1, 1)),
            )
            _, d = _run(spec, 50.0)
            return d

        d_sd = _run_model(gast_sd)
        d_sf = _run_model(gast_sf)
        assert not np.allclose(d_sd, d_sf, rtol=1e-3), \
            "GastSD and GastSF produce identical trajectories — V equations must differ"

    def test_adaptation_variable_responds_to_r(self):
        """B (adaptation derivative) must be non-zero when r > 0."""
        spec = SimulationSpec(
            model=gast_sd,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("linear", a=0.0),
            monitors=(MonitorSpec("raw"),),
            weights=np.zeros((1, 1)),
        )
        t, d = _run(spec, 50.0)
        # r (index 0) should be positive somewhere → B (index 3) should vary
        r_max = d[:, 0, 0].max()
        B_range = d[:, 3, 0].max() - d[:, 3, 0].min()
        if r_max > 1e-6:   # only check if r actually fires
            assert B_range > 1e-10, \
                "GastSD: B (adaptation derivative) not responding to r"

    @pytest.mark.parametrize("model", [gast_sd, gast_sf])
    def test_cr_cv_coupling_weights(self, model):
        """cr controls r-channel coupling; cv controls V-channel coupling."""
        n = 3
        rng = np.random.default_rng(2)
        W = np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(W, 0.0)

        def _spec(cr, cv):
            return SimulationSpec(
                model=model,
                integrator=IntegratorSpec(method="heun", dt=0.01),
                coupling=CouplingSpec("linear", a=0.1),
                monitors=(MonitorSpec("raw"),),
                weights=W,
                node_params={"cr": cr, "cv": cv},
            )

        _, d_r = _run(_spec(cr=1.0, cv=0.0), 20.0)
        _, d_v = _run(_spec(cr=0.0, cv=1.0), 20.0)
        _, d_0 = _run(_spec(cr=0.0, cv=0.0), 20.0)

        # With coupling: r-only and V-only should differ from no-coupling
        assert not np.allclose(d_r[-1], d_0[-1], rtol=1e-4), \
            f"{model.name}: cr=1 coupling had no effect"
        assert not np.allclose(d_v[-1], d_0[-1], rtol=1e-4), \
            f"{model.name}: cv=1 coupling had no effect"


# ---------------------------------------------------------------------------
# Multi-node with delays: new models must handle non-zero tract lengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,dt,coup_a", [
    (generic_2d_oscillator, 0.01, 0.05),
    (sup_hopf,              0.01, 0.1),
    (larter_breakspear,     0.1,  0.05),
    (coombes_byrne_2d,      0.01, 0.05),
    (gast_sd,               0.01, 0.02),
])
def test_with_delays(model, dt, coup_a):
    """Non-zero tract lengths must not cause crashes or non-finite output."""
    n = 4
    rng = np.random.default_rng(42)
    W = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    D = np.abs(rng.standard_normal((n, n))) * 10.0
    np.fill_diagonal(D, 0.0)

    spec = SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("raw"),),
        weights=W,
        tract_lengths=D,
        speed=4.0,
    )
    t, d = Simulator(spec, backend="numpy").run(20.0)["raw"]
    assert np.isfinite(d).all(), f"{model.name}: non-finite output with delays"
