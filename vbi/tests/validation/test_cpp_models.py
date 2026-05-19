"""
C++ backend validation for all supported models.

Each test checks that the deterministic C++ trajectory matches NumPy
to rtol=1e-4, covering different model sizes and equation structures.
"""
import numpy as np
import pytest

from vbi.simulator.api import Simulator
from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.integrator import IntegratorSpec
from vbi.simulator.spec.coupling import CouplingSpec
from vbi.simulator.spec.monitor import MonitorSpec

from vbi.simulator.models.mpr import mpr
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.simulator.models.wilson_cowan import wilson_cowan
from vbi.simulator.models.reduced_wong_wang import reduced_wong_wang
from vbi.simulator.models.wong_wang_exc_inh import wong_wang_exc_inh


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_weights(n_nodes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 0.5, (n_nodes, n_nodes))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    return W


def _make_spec(model, n_nodes=8, dt=0.1, seed=0, coup_a=0.05):
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=coup_a, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_make_weights(n_nodes, seed),
    )


def _check_cpp_vs_numpy(model, n_nodes=8, duration=50.0, dt=0.1, coup_a=0.05):
    spec = _make_spec(model, n_nodes=n_nodes, dt=dt, coup_a=coup_a)

    sim_np  = Simulator(spec, backend="numpy")
    sim_cpp = Simulator(spec, backend="cpp")

    _, d_np  = sim_np.run(duration)["raw"]
    _, d_cpp = sim_cpp.run(duration)["raw"]

    assert d_np.shape == d_cpp.shape, \
        f"Shape mismatch: numpy={d_np.shape}, cpp={d_cpp.shape}"
    np.testing.assert_allclose(
        d_cpp, d_np, rtol=1e-4, atol=1e-8,
        err_msg=f"{model.name}: C++ vs NumPy trajectory disagrees",
    )


# ---------------------------------------------------------------------------
# MPR (already tested in test_mpr_cpp.py — repeated here for completeness)
# ---------------------------------------------------------------------------

def test_mpr_cpp_vs_numpy_small():
    _check_cpp_vs_numpy(mpr, n_nodes=5, duration=50.0, dt=0.01, coup_a=0.1)


def test_mpr_cpp_vs_numpy_medium():
    _check_cpp_vs_numpy(mpr, n_nodes=20, duration=100.0, dt=0.01, coup_a=0.1)


# ---------------------------------------------------------------------------
# Jansen-Rit (6 SVs, nonlinear sigmoid, 1 coupling var)
# ---------------------------------------------------------------------------

def test_jr_cpp_vs_numpy_small():
    """JR has 6 state variables and a sigmoid — tests exp() in dfun."""
    _check_cpp_vs_numpy(jansen_rit, n_nodes=6, duration=100.0, dt=0.1, coup_a=0.01)


def test_jr_cpp_vs_numpy_medium():
    _check_cpp_vs_numpy(jansen_rit, n_nodes=16, duration=200.0, dt=0.1, coup_a=0.01)


def test_jr_cpp_stoch():
    """JR stochastic run completes and returns finite output."""
    spec = SimulationSpec(
        model=jansen_rit,
        integrator=IntegratorSpec(
            method="heun", dt=0.1, stochastic=True,
            noise_nsig=np.array([1e-3]), noise_seed=7,
        ),
        coupling=CouplingSpec(kind="linear", a=0.01, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_make_weights(6),
    )
    sim = Simulator(spec, backend="cpp")
    _, d = sim.run(100.0)["raw"]
    assert np.all(np.isfinite(d)), "JR stochastic C++ output has non-finite values"


# ---------------------------------------------------------------------------
# Wilson-Cowan (2 SVs, bounded [0,1], complex sigmoid with shift)
# ---------------------------------------------------------------------------

def test_wc_cpp_vs_numpy_small():
    """WC has state bounds [0,1] and a shifted sigmoid — tests clamping."""
    _check_cpp_vs_numpy(wilson_cowan, n_nodes=6, duration=100.0, dt=0.1, coup_a=0.05)


def test_wc_cpp_vs_numpy_medium():
    _check_cpp_vs_numpy(wilson_cowan, n_nodes=20, duration=200.0, dt=0.1, coup_a=0.05)


def test_wc_cpp_stoch():
    """WC stochastic run completes with bounded output."""
    spec = SimulationSpec(
        model=wilson_cowan,
        integrator=IntegratorSpec(
            method="heun", dt=0.1, stochastic=True,
            noise_nsig=np.array([1e-4, 1e-4]), noise_seed=13,
        ),
        coupling=CouplingSpec(kind="linear", a=0.05, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_make_weights(8),
    )
    sim = Simulator(spec, backend="cpp")
    _, d = sim.run(100.0)["raw"]
    assert np.all(np.isfinite(d)), "WC stochastic C++ output has non-finite values"
    # WC state is bounded [0, 1]
    assert d.min() >= -1e-6, "WC state went below lower bound"
    assert d.max() <= 1.0 + 1e-6, "WC state exceeded upper bound"


# ---------------------------------------------------------------------------
# Euler integrator
# ---------------------------------------------------------------------------

def test_mpr_euler_cpp_vs_numpy():
    """C++ Euler must match NumPy Euler."""
    spec = SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="euler", dt=0.001, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=0.1, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_make_weights(8),
    )
    sim_np  = Simulator(spec, backend="numpy")
    sim_cpp = Simulator(spec, backend="cpp")
    _, d_np  = sim_np.run(20.0)["raw"]
    _, d_cpp = sim_cpp.run(20.0)["raw"]
    np.testing.assert_allclose(d_cpp, d_np, rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# ReducedWongWang (1 SV, bounded [0,1], piecewise-safe transfer function)
# ---------------------------------------------------------------------------

def test_rww_cpp_vs_numpy_small():
    """RWW: 1 SV with exp() in transfer function — tests C++ codegen for RWW."""
    _check_cpp_vs_numpy(reduced_wong_wang, n_nodes=6, duration=100.0, dt=0.1, coup_a=0.02)


def test_rww_cpp_vs_numpy_medium():
    _check_cpp_vs_numpy(reduced_wong_wang, n_nodes=20, duration=200.0, dt=0.1, coup_a=0.02)


def test_rww_cpp_stoch():
    """RWW stochastic run completes and stays in [0, 1]."""
    spec = SimulationSpec(
        model=reduced_wong_wang,
        integrator=IntegratorSpec(
            method="heun", dt=0.1, stochastic=True,
            noise_nsig=np.array([1e-4]), noise_seed=21,
        ),
        coupling=CouplingSpec(kind="linear", a=0.02, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_make_weights(6),
    )
    sim = Simulator(spec, backend="cpp")
    _, d = sim.run(100.0)["raw"]
    assert np.all(np.isfinite(d)), "RWW stochastic C++ output has non-finite values"
    assert d.min() >= -1e-6,       "RWW state went below lower bound"
    assert d.max() <= 1.0 + 1e-6,  "RWW state exceeded upper bound"


# ---------------------------------------------------------------------------
# WongWangExcInh (2 SVs, separate exc/inh populations, two transfer functions)
# ---------------------------------------------------------------------------

def test_wwex_cpp_vs_numpy_small():
    """WWEX: 2-population model with separate exc/inh dynamics."""
    _check_cpp_vs_numpy(wong_wang_exc_inh, n_nodes=6, duration=100.0, dt=0.1, coup_a=0.02)


def test_wwex_cpp_vs_numpy_medium():
    _check_cpp_vs_numpy(wong_wang_exc_inh, n_nodes=20, duration=200.0, dt=0.1, coup_a=0.02)


def test_wwex_cpp_stoch():
    """WWEX stochastic run completes with both SVs in [0, 1]."""
    spec = SimulationSpec(
        model=wong_wang_exc_inh,
        integrator=IntegratorSpec(
            method="heun", dt=0.1, stochastic=True,
            noise_nsig=np.array([1e-4, 1e-4]), noise_seed=33,
        ),
        coupling=CouplingSpec(kind="linear", a=0.02, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=_make_weights(6),
    )
    sim = Simulator(spec, backend="cpp")
    _, d = sim.run(100.0)["raw"]
    assert np.all(np.isfinite(d)), "WWEX stochastic C++ output has non-finite values"
    assert d.min() >= -1e-6,       "WWEX state went below lower bound"
    assert d.max() <= 1.0 + 1e-6,  "WWEX state exceeded upper bound"


# ---------------------------------------------------------------------------
# Delayed coupling (DDE)
# ---------------------------------------------------------------------------

def test_mpr_cpp_vs_numpy_with_delays():
    """C++ ring buffer must match NumPy DenseHistory with non-zero delays."""
    n_nodes = 8
    rng = np.random.default_rng(42)
    W = rng.uniform(0, 0.5, (n_nodes, n_nodes))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    # tract lengths ~ 1–10 mm at 4 mm/ms → 0.25–2.5 ms delay
    TL = rng.uniform(1.0, 10.0, (n_nodes, n_nodes))
    np.fill_diagonal(TL, 0.0)

    spec = SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=0.01, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=0.05, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=W,
        tract_lengths=TL,
        speed=4.0,
    )
    sim_np  = Simulator(spec, backend="numpy")
    sim_cpp = Simulator(spec, backend="cpp")
    _, d_np  = sim_np.run(50.0)["raw"]
    _, d_cpp = sim_cpp.run(50.0)["raw"]
    np.testing.assert_allclose(d_cpp, d_np, rtol=1e-4, atol=1e-8,
                               err_msg="Delayed MPR: C++ vs NumPy disagree")
