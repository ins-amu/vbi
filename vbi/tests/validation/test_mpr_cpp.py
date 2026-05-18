"""
M2 validation: C++ backend must match the NumPy reference.

Tests
-----
test_cpp_matches_numpy        — deterministic single run, rtol=1e-4
test_cpp_stoch_runs           — stochastic run completes without error
test_cpp_cache_reuse          — second build_or_load returns in < 200 ms
"""
import time
import numpy as np
import pytest

from vbi.simulator.api import Simulator
from vbi.simulator.models.mpr import mpr
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.monitor import MonitorSpec
from vbi.simulator.backend.cpp.build import build_or_load


# ---------------------------------------------------------------------------
# Shared spec builder
# ---------------------------------------------------------------------------

def _build_spec(n_nodes=10, dt=0.01, stochastic=False, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 0.5, (n_nodes, n_nodes))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)

    integrator = IntegratorSpec(
        method="heun",
        dt=dt,
        stochastic=stochastic,
        noise_nsig=np.array([1e-4, 1e-4]) if stochastic else None,
        noise_seed=42,
    )
    return SimulationSpec(
        model=mpr,
        integrator=integrator,
        coupling=CouplingSpec(kind="linear", a=0.1, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=W,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cpp_matches_numpy():
    """C++ Heun deterministic must reproduce NumPy to rtol=1e-4."""
    spec     = _build_spec(n_nodes=10, dt=0.01, stochastic=False)
    duration = 100.0   # ms — keep short for CI

    sim_np  = Simulator(spec, backend="numpy")
    sim_cpp = Simulator(spec, backend="cpp")

    t_np,  d_np  = sim_np.run(duration)["raw"]
    t_cpp, d_cpp = sim_cpp.run(duration)["raw"]

    assert t_np.shape  == t_cpp.shape,  "time arrays differ in shape"
    assert d_np.shape  == d_cpp.shape,  "data arrays differ in shape"
    np.testing.assert_allclose(d_cpp, d_np, rtol=1e-4, atol=1e-8,
                               err_msg="C++ vs NumPy trajectories disagree")


def test_cpp_stoch_runs():
    """Stochastic C++ run completes and produces finite output."""
    spec = _build_spec(n_nodes=5, dt=0.01, stochastic=True)
    sim  = Simulator(spec, backend="cpp")
    _, d = sim.run(50.0)["raw"]
    assert np.all(np.isfinite(d)), "stochastic C++ output contains non-finite values"


def test_cpp_cache_reuse():
    """Second build_or_load for same spec returns in < 200 ms (cache hit)."""
    spec = _build_spec(n_nodes=6, dt=0.01)
    build_or_load(spec)          # warm up

    t0 = time.perf_counter()
    build_or_load(spec)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.2, f"Cache hit took {elapsed:.3f}s — expected < 0.2s"
