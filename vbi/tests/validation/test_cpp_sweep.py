"""
C++ sweeper tests.

1. Serial sweep matches individual CppSimulator runs (correctness).
2. Parallel sweep matches serial sweep (parallelism correctness).
3. Sweep works for JR and WilsonCowan (model coverage).
"""
import numpy as np
import pytest

from vbi.simulator.api import Simulator, Sweeper
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.backend.cpp.sweeper import CppSweeper

from vbi.simulator.models.mpr import mpr
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.simulator.models.wilson_cowan import wilson_cowan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights(n, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 0.5, (n, n))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    return W


def _base_spec(model, n_nodes=6, dt=0.1, coup_a=0.05):
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=coup_a, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_weights(n_nodes),
    )


# ---------------------------------------------------------------------------
# 1. Serial sweep correctness: each sweep point must match its own CppSimulator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,param_name,values,coup_a,dt", [
    (mpr,          "eta",  np.linspace(-5.0, -4.0, 4), 0.1,  0.01),
    (jansen_rit,   "mu",   np.linspace(0.15, 0.30, 4), 0.01, 0.1),
    (wilson_cowan, "P",    np.linspace(0.0,  0.3,  4), 0.05, 0.1),
])
def test_serial_sweep_matches_individual_runs(model, param_name, values, coup_a, dt):
    """Each serial sweep point must match an individual CppSimulator run."""
    n_nodes  = 6
    duration = 50.0
    spec = _base_spec(model, n_nodes=n_nodes, dt=dt, coup_a=coup_a)
    sweep_spec = SweepSpec(params={param_name: values})
    sweeper = CppSweeper(spec, sweep_spec)

    sweep_results = sweeper.run_serial(duration)   # list of monitor dicts

    for i, val in enumerate(values):
        patched = SimulationSpec(
            model=spec.model,
            integrator=spec.integrator,
            coupling=spec.coupling,
            monitors=spec.monitors,
            weights=spec.weights,
            node_params={param_name: float(val)},
        )
        sim = Simulator(patched, backend="cpp")
        _, d_ref  = sim.run(duration)["raw"]
        _, d_sweep = sweep_results[i]["raw"]

        np.testing.assert_allclose(
            d_sweep, d_ref, rtol=1e-10, atol=0.0,
            err_msg=f"{model.name}, {param_name}={val:.3f}: "
                    f"sweep[{i}] differs from individual run",
        )


# ---------------------------------------------------------------------------
# 2. Parallel sweep matches serial sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,param_name,values,coup_a,dt", [
    (mpr,          "eta",  np.linspace(-5.0, -4.0, 6), 0.1,  0.01),
    (jansen_rit,   "mu",   np.linspace(0.15, 0.30, 6), 0.01, 0.1),
    (wilson_cowan, "P",    np.linspace(0.0,  0.3,  6), 0.05, 0.1),
])
def test_parallel_matches_serial(model, param_name, values, coup_a, dt):
    """Parallel (4-thread) sweep must be bit-identical to serial sweep."""
    n_nodes  = 6
    duration = 50.0
    spec = _base_spec(model, n_nodes=n_nodes, dt=dt, coup_a=coup_a)
    sweep_spec = SweepSpec(params={param_name: values})

    sweeper = CppSweeper(spec, sweep_spec, n_workers=4)
    serial_results   = sweeper.run_serial(duration)
    parallel_results = sweeper.run_parallel(duration, n_workers=4)

    for i in range(len(values)):
        _, d_serial   = serial_results[i]["raw"]
        _, d_parallel = parallel_results[i]["raw"]
        np.testing.assert_array_equal(
            d_parallel, d_serial,
            err_msg=f"{model.name}, {param_name}={values[i]:.3f}: "
                    f"parallel[{i}] differs from serial[{i}]",
        )


# ---------------------------------------------------------------------------
# 3. Sweeper via api.Sweeper interface
# ---------------------------------------------------------------------------

def test_sweeper_api_cpp():
    """Sweeper(spec, sweep_spec, backend='cpp') returns correct number of results."""
    spec = _base_spec(mpr, n_nodes=5, dt=0.01)
    eta_values = np.linspace(-5.5, -4.0, 5)
    sweep_spec = SweepSpec(params={"eta": eta_values})

    sweeper = Sweeper(spec, sweep_spec, backend="cpp")
    results = sweeper.run(50.0)

    assert len(results) == len(eta_values), \
        f"Expected {len(eta_values)} results, got {len(results)}"
    for res in results:
        assert "raw" in res, "Monitor 'raw' missing from sweep result"
        _, d = res["raw"]
        assert np.all(np.isfinite(d)), "Non-finite values in sweep output"


# ---------------------------------------------------------------------------
# 4. 2D parameter grid sweep
# ---------------------------------------------------------------------------

def test_2d_grid_sweep_cpp():
    """2D grid sweep: G × eta with 3×3 = 9 runs."""
    spec = _base_spec(mpr, n_nodes=5, dt=0.01)
    sweep_spec = SweepSpec(params={
        "eta": np.array([-5.0, -4.5, -4.0]),
        "J":   np.array([12.0, 14.5, 17.0]),
    })
    assert sweep_spec.n_samples == 9

    sweeper = CppSweeper(spec, sweep_spec)
    results = sweeper.run_serial(30.0)
    assert len(results) == 9
    for res in results:
        _, d = res["raw"]
        assert np.all(np.isfinite(d))


# ---------------------------------------------------------------------------
# 5. Stochastic sweep: each run gets a unique noise seed
# ---------------------------------------------------------------------------

def test_stochastic_sweep_unique_trajectories():
    """Stochastic sweep: different runs must not be identical (unique seeds)."""
    spec = SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(
            method="heun", dt=0.01, stochastic=True,
            noise_nsig=np.array([1e-3, 1e-3]), noise_seed=0,
        ),
        coupling=CouplingSpec(kind="linear", a=0.1, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_weights(5),
    )
    # All eta identical — only noise differs between runs
    sweep_spec = SweepSpec(params={"eta": np.full(4, -4.6)})
    sweeper = CppSweeper(spec, sweep_spec)
    results = sweeper.run_serial(30.0)

    d0 = results[0]["raw"][1]
    for i in range(1, len(results)):
        di = results[i]["raw"][1]
        assert not np.array_equal(di, d0), \
            f"Stochastic run {i} is identical to run 0 — seeds not independent"
