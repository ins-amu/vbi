"""
M1 validation: Numba CPU backend for the new models.

Gold standard: NumPy baseline (validated in test_new_models_numpy.py).
All Numba results must match NumPy to rtol=1e-4 (deterministic).
"""
from vbi.simulator.spec.connectivity import Connectivity
import numpy as np
import pytest

from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
)
from vbi.simulator.spec.sweep import SweepSpec

from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator
from vbi.simulator.models.kuramoto import kuramoto
from vbi.simulator.models.sup_hopf import sup_hopf
from vbi.simulator.models.linear import linear
from vbi.simulator.models.larter_breakspear import larter_breakspear
from vbi.simulator.models.coombes_byrne_2d import coombes_byrne_2d
from vbi.simulator.models.gast_sd import gast_sd
from vbi.simulator.models.gast_sf import gast_sf

pytestmark = pytest.mark.slow


numba = pytest.importorskip("numba", reason="numba not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights(n, seed=0):
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(min=1e-8)
    return W


def _spec(model, n_nodes=4, dt=0.01, coup_a=0.05, method="heun",
          stochastic=False, node_params=None, tract_lengths=None):
    W = _weights(n_nodes)
    if tract_lengths is None:
        tract_lengths = np.zeros_like(W)
    noise_nsig = None
    if stochastic:
        n_noise = len(model.noise_indices)
        noise_nsig = np.full(n_noise, 1e-4)
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method=method, dt=dt,
                                  stochastic=stochastic, noise_nsig=noise_nsig),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("raw"),),
        connectivity=Connectivity(weights=W, tract_lengths=tract_lengths),

        node_params=node_params or {},
    )


def _np(spec, duration):
    return Simulator(spec, backend="numpy").run(duration)["raw"]


def _nb(spec, duration):
    return Simulator(spec, backend="numba").run(duration)["raw"]


# ---------------------------------------------------------------------------
# Parametrised deterministic match: Numba must reproduce NumPy to rtol=1e-4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,dt,coup_a,duration", [
    (generic_2d_oscillator, 0.01, 0.05,  50.0),
    (kuramoto,              0.01, 0.1,   30.0),
    (sup_hopf,              0.01, 0.1,   30.0),
    (linear,                0.01, 0.05,  30.0),
    (larter_breakspear,     0.1,  0.05,  50.0),
    (coombes_byrne_2d,      0.01, 0.05,  30.0),
    (gast_sd,               0.01, 0.02,  30.0),
    (gast_sf,               0.01, 0.02,  30.0),
])
class TestNumbaDeterministicMatchesNumPy:

    def test_raw_matches_numpy_small(self, model, dt, coup_a, duration):
        """Numba raw output must match NumPy to rtol=1e-4 (4 nodes)."""
        spec = _spec(model, n_nodes=4, dt=dt, coup_a=coup_a)
        t_np, d_np = _np(spec, duration)
        t_nb, d_nb = _nb(spec, duration)
        assert d_np.shape == d_nb.shape, \
            f"{model.name}: shape mismatch np={d_np.shape} nb={d_nb.shape}"
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg=f"{model.name}: Numba diverges from NumPy (n_nodes=4)",
        )

    def test_raw_matches_numpy_euler(self, model, dt, coup_a, duration):
        """Euler integrator: Numba must also match NumPy."""
        spec = _spec(model, n_nodes=4, dt=dt, coup_a=coup_a, method="euler")
        t_np, d_np = _np(spec, duration)
        t_nb, d_nb = _nb(spec, duration)
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg=f"{model.name}: Numba Euler diverges from NumPy",
        )

    def test_raw_matches_numpy_single_node(self, model, dt, coup_a, duration):
        """Single node (no coupling): Numba must match NumPy exactly."""
        spec = _spec(model, n_nodes=1, dt=dt, coup_a=0.0)
        t_np, d_np = _np(spec, duration)
        t_nb, d_nb = _nb(spec, duration)
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg=f"{model.name}: single-node Numba diverges from NumPy",
        )


# ---------------------------------------------------------------------------
# Deterministic match with delays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,dt,coup_a", [
    (generic_2d_oscillator, 0.01, 0.05),
    (sup_hopf,              0.01, 0.1),
    (larter_breakspear,     0.1,  0.05),
    (coombes_byrne_2d,      0.01, 0.05),
])
def test_with_delays_matches_numpy(model, dt, coup_a):
    """Non-zero tract lengths: Numba ring buffer must match NumPy."""
    rng = np.random.default_rng(7)
    n = 4
    D = np.abs(rng.standard_normal((n, n))) * 15.0
    np.fill_diagonal(D, 0.0)
    spec = _spec(model, n_nodes=n, dt=dt, coup_a=coup_a, tract_lengths=D)
    t_np, d_np = _np(spec, 20.0)
    t_nb, d_nb = _nb(spec, 20.0)
    np.testing.assert_allclose(
        d_nb, d_np, rtol=1e-4,
        err_msg=f"{model.name}: Numba delayed diverges from NumPy",
    )


# ---------------------------------------------------------------------------
# Stochastic: reproducibility and finite output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,dt", [
    (generic_2d_oscillator, 0.01),
    (sup_hopf,              0.01),
    (larter_breakspear,     0.1),
    (coombes_byrne_2d,      0.01),
    (gast_sd,               0.01),
    (gast_sf,               0.01),
])
class TestNumbaStochastic:

    def test_stochastic_runs_finite(self, model, dt):
        spec = _spec(model, n_nodes=4, dt=dt, stochastic=True)
        _, d = _nb(spec, 20.0)
        assert np.isfinite(d).all(), f"{model.name}: stochastic Numba output not finite"

    def test_same_seed_reproduces(self, model, dt):
        spec = _spec(model, n_nodes=4, dt=dt, stochastic=True)
        _, d1 = _nb(spec, 20.0)
        _, d2 = _nb(spec, 20.0)
        np.testing.assert_array_equal(d1, d2,
                                      err_msg=f"{model.name}: same seed must reproduce")


# ---------------------------------------------------------------------------
# Bounds: Numba must also enforce state variable bounds
# ---------------------------------------------------------------------------

def test_coombes_byrne_r_nonneg_numba():
    """Numba backend: r in CoombesByrne2D must stay >= 0."""
    spec = _spec(coombes_byrne_2d, n_nodes=6, coup_a=0.1)
    _, d = _nb(spec, 50.0)
    assert np.all(d[:, 0, :] >= 0.0), "Numba CoombesByrne2D: r went negative"


def test_gast_sd_r_nonneg_numba():
    """Numba backend: r in GastSD must stay >= 0."""
    spec = _spec(gast_sd, n_nodes=4, coup_a=0.05)
    _, d = _nb(spec, 50.0)
    assert np.all(d[:, 0, :] >= 0.0), "Numba GastSD: r went negative"


# ---------------------------------------------------------------------------
# Node-heterogeneous parameters via Numba
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,param,values,dt", [
    (generic_2d_oscillator, "I",   np.array([0.0, 0.5, 1.0]), 0.1),
    (kuramoto,              "omega", np.array([0.5, 1.0, 2.0]), 0.01),
    (sup_hopf,              "a",   np.array([-1.0, 0.0, 1.0]), 0.01),
    (linear,                "gamma", np.array([-5.0, -10.0, -20.0]), 0.01),
])
def test_heterogeneous_params_numba_matches_numpy(model, param, values, dt):
    """Per-node parameters must give same result in Numba and NumPy."""
    n = len(values)
    spec = SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("linear", a=0.0),
        monitors=(MonitorSpec("raw"),),
        connectivity=Connectivity(weights=np.zeros((n, n))),
        node_params={param: values},
    )
    _, d_np = _np(spec, 30.0)
    _, d_nb = _nb(spec, 30.0)
    np.testing.assert_allclose(
        d_nb, d_np, rtol=1e-4,
        err_msg=f"{model.name}: node-heterogeneous '{param}' Numba vs NumPy mismatch",
    )


# ---------------------------------------------------------------------------
# Kuramoto sinusoidal coupling (kind='kuramoto') - Numba vs NumPy
# ---------------------------------------------------------------------------

class TestKuramotoCouplingNumba:
    """Verify that the Numba backend reproduces NumPy for kind='kuramoto'."""

    def _kuramoto_spec(self, n, theta0, omega, G, dt=0.01,
                       tract_lengths=None, speed=1.0):
        import dataclasses
        W = _weights(n, seed=42)
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)
        return SimulationSpec(
            model=mdl,
            integrator=IntegratorSpec(method="heun", dt=dt),
            coupling=CouplingSpec("kuramoto"),
            monitors=(MonitorSpec("raw"),),
            connectivity=Connectivity(weights=W, tract_lengths=tract_lengths, speed=speed),


            node_params={"omega": omega, "G": G},
        )

    def test_deterministic_matches_numpy(self):
        """Zero-delay sinusoidal Kuramoto: Numba must match NumPy to rtol=1e-4."""
        n = 5
        rng = np.random.default_rng(3)
        theta0 = rng.uniform(-np.pi, np.pi, n)
        omega  = rng.uniform(0.5, 2.0, n)
        spec = self._kuramoto_spec(n, theta0, omega, G=0.5)
        _, d_np = _np(spec, 50.0)
        _, d_nb = _nb(spec, 50.0)
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg="Kuramoto sinusoidal: Numba diverges from NumPy")

    def test_euler_matches_numpy(self):
        """Euler integrator: Numba sinusoidal Kuramoto must also match NumPy."""
        import dataclasses
        n = 4
        rng = np.random.default_rng(5)
        theta0 = rng.uniform(-np.pi, np.pi, n)
        omega  = rng.uniform(0.5, 2.0, n)
        W = _weights(n, seed=42)
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)
        spec = SimulationSpec(
            model=mdl,
            integrator=IntegratorSpec(method="euler", dt=0.01),
            coupling=CouplingSpec("kuramoto"),
            monitors=(MonitorSpec("raw"),),
            connectivity=Connectivity(weights=W),
            node_params={"omega": omega, "G": 0.5},
        )
        _, d_np = _np(spec, 30.0)
        _, d_nb = _nb(spec, 30.0)
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg="Kuramoto sinusoidal Euler: Numba diverges from NumPy")

    def test_delayed_matches_numpy(self):
        """Delayed sinusoidal Kuramoto: ring-buffer handling must match NumPy."""
        import dataclasses
        n = 4
        rng = np.random.default_rng(11)
        theta0 = rng.uniform(-np.pi, np.pi, n)
        omega  = rng.uniform(0.5, 2.0, n)
        W = _weights(n, seed=42)
        tract = np.abs(rng.standard_normal((n, n))) * 10.0
        np.fill_diagonal(tract, 0.0)
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)
        spec = SimulationSpec(
            model=mdl,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("kuramoto"),
            monitors=(MonitorSpec("raw"),),
            connectivity=Connectivity(weights=W, tract_lengths=tract, speed=1.0),


            node_params={"omega": omega, "G": 0.5},
        )
        _, d_np = _np(spec, 30.0)
        _, d_nb = _nb(spec, 30.0)
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg="Delayed Kuramoto sinusoidal: Numba diverges from NumPy")

# ---------------------------------------------------------------------------
# Sweep: Numba sweep must match NumPy sweep for new models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,sweep_param,values,dt,coup_a", [
    (generic_2d_oscillator, "I",   np.linspace(0.0, 1.0, 5), 0.01, 0.05),
    (sup_hopf,              "a",   np.linspace(-1.0, 1.0, 5), 0.01, 0.1),
    (larter_breakspear,     "d_V", np.linspace(0.5, 0.65, 4), 0.1,  0.05),
    (coombes_byrne_2d,      "eta", np.linspace(0.5, 3.0, 4),  0.01, 0.05),
    (gast_sd,               "eta", np.linspace(-8.0, -4.0, 4), 0.01, 0.02),
])
def test_sweep_numba_matches_numpy(model, sweep_param, values, dt, coup_a):
    """Numba sweep must match NumPy sweep to rtol=1e-4."""
    n_nodes = 4
    spec = SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("raw"),),
        connectivity=Connectivity(weights=_weights(n_nodes)),
    )
    sweep_spec = SweepSpec(params={sweep_param: values})

    np_results = Sweeper(spec, sweep_spec, backend="numpy").run(20.0)
    nb_results = Sweeper(spec, sweep_spec, backend="numba").run(20.0)

    for i, val in enumerate(values):
        _, d_np = np_results[i]["raw"]
        _, d_nb = nb_results[i]["raw"]
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg=f"{model.name} sweep {sweep_param}={val:.3f}: Numba vs NumPy mismatch",
        )
