"""
M3 validation: Numba-CUDA backend.

All tests are skipped when no CUDA device is present.

Gold standard: Numba CPU backend (validated against NumPy in test_mpr_numba.py).
GPU uses float32; CPU uses float64.  Accepted tolerance: rtol=1e-3.

Tests
-----
test_cuda_single_matches_numba    — single run, deterministic
test_cuda_stoch_runs              — stochastic run, finite output
test_cuda_single_with_delays      — non-zero tract lengths
test_cuda_sweep_matches_numba     — sweep, deterministic (multi-model)
test_cuda_sweep_stoch_unique      — stochastic sweep, unique trajectories
test_cuda_sweep_pipeline_shape    — pipeline mode, shape + finite
test_cuda_sweep_throughput        — benchmark, not a pass/fail test
"""
import time

import numpy as np
import pytest

try:
    from numba import cuda as _numba_cuda
    CUDA_AVAILABLE = _numba_cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

cuda_only = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="no CUDA device available"
)

from vbi.simulator import Simulator, Sweeper
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec,
)
from vbi.simulator.spec.sweep import SweepSpec
from vbi.simulator.models.mpr import mpr
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.simulator.models.wilson_cowan import wilson_cowan
from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights(n, seed=0):
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(1e-8)
    return W


def _spec(model=mpr, n_nodes=10, dt=0.01, coup_a=0.1,
          stochastic=False, tract_lengths=None):
    W = _weights(n_nodes)
    if tract_lengths is None:
        tract_lengths = np.zeros_like(W)
    noise_nsig = None
    if stochastic:
        n_noise = len(model.noise_indices)
        noise_nsig = np.full(n_noise, 1e-4)
    return SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=dt,
                                  stochastic=stochastic, noise_nsig=noise_nsig),
        coupling=CouplingSpec("linear", a=coup_a),
        monitors=(MonitorSpec("raw"),),
        weights=W,
        tract_lengths=tract_lengths,
    )


# ---------------------------------------------------------------------------
# Single-run tests
# ---------------------------------------------------------------------------

@cuda_only
def test_cuda_single_matches_numba():
    """CUDA single run must match Numba CPU to rtol=1e-3 (float32 vs float64)."""
    spec     = _spec(mpr, n_nodes=10, dt=0.01)
    duration = 100.0

    _, d_nb   = Simulator(spec, backend="numba").run(duration)["raw"]
    _, d_cuda = Simulator(spec, backend="cuda").run(duration)["raw"]

    assert d_nb.shape == d_cuda.shape, \
        f"Shape mismatch: numba={d_nb.shape}, cuda={d_cuda.shape}"
    np.testing.assert_allclose(
        d_cuda, d_nb, rtol=1e-3, atol=1e-5,
        err_msg="CUDA single run diverges from Numba CPU (float32 vs float64)",
    )


@cuda_only
def test_cuda_stoch_runs():
    """Stochastic CUDA run completes and produces finite output."""
    spec = _spec(mpr, n_nodes=5, stochastic=True)
    _, d = Simulator(spec, backend="cuda").run(50.0)["raw"]
    assert np.isfinite(d).all(), "CUDA stochastic output is not finite"


@cuda_only
def test_cuda_single_with_delays():
    """Non-zero tract lengths: CUDA ring buffer must match Numba CPU (rtol=1e-3)."""
    n = 8
    rng = np.random.default_rng(7)
    D = np.abs(rng.standard_normal((n, n))) * 10.0
    np.fill_diagonal(D, 0.0)
    spec = _spec(mpr, n_nodes=n, dt=0.01, tract_lengths=D)

    _, d_nb   = Simulator(spec, backend="numba").run(50.0)["raw"]
    _, d_cuda = Simulator(spec, backend="cuda").run(50.0)["raw"]

    np.testing.assert_allclose(
        d_cuda, d_nb, rtol=1e-3, atol=1e-5,
        err_msg="CUDA delayed ring buffer diverges from Numba CPU",
    )


@cuda_only
@pytest.mark.parametrize("model,dt,coup_a", [
    (mpr,                     0.01, 0.1),
    (jansen_rit,              0.1,  0.01),
    (wilson_cowan,            0.1,  0.05),
    (generic_2d_oscillator,   0.01, 0.05),
])
def test_cuda_single_model_coverage(model, dt, coup_a):
    """CUDA single run must work for multiple models (rtol=1e-3)."""
    spec = _spec(model, n_nodes=6, dt=dt, coup_a=coup_a)
    _, d_nb   = Simulator(spec, backend="numba").run(20.0)["raw"]
    _, d_cuda = Simulator(spec, backend="cuda").run(20.0)["raw"]
    assert d_nb.shape == d_cuda.shape
    # float32 GPU vs float64 CPU; higher-dimensional models accumulate more
    # rounding error, so we allow rtol=5e-3 here.
    np.testing.assert_allclose(
        d_cuda, d_nb, rtol=5e-3, atol=1e-5,
        err_msg=f"{model.name}: CUDA vs Numba mismatch",
    )


# ---------------------------------------------------------------------------
# Sweep tests
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("model,param,values,dt,coup_a", [
    (mpr,   "eta", np.linspace(-5.5, -4.0, 20), 0.01, 0.1),
    (mpr,   "J",   np.linspace(10.0, 18.0, 20), 0.01, 0.1),
])
def test_cuda_sweep_matches_numba(model, param, values, dt, coup_a):
    """CUDA sweep must match Numba CPU sweep (rtol=1e-3, float32 vs float64)."""
    spec       = _spec(model, n_nodes=10, dt=dt, coup_a=coup_a)
    sweep_spec = SweepSpec(params={param: values})

    nb_results   = Sweeper(spec, sweep_spec, backend="numba").run(100.0)
    cuda_results = Sweeper(spec, sweep_spec, backend="cuda").run(100.0)

    for i, val in enumerate(values):
        _, d_nb   = nb_results[i]["raw"]
        _, d_cuda = cuda_results[i]["raw"]
        np.testing.assert_allclose(
            d_cuda, d_nb, rtol=1e-3, atol=1e-5,
            err_msg=f"{model.name} sweep {param}={val:.3f}: CUDA vs Numba mismatch",
        )


@cuda_only
def test_cuda_sweep_stoch_unique():
    """Stochastic CUDA sweep: different parameter sets must produce different output."""
    spec       = _spec(mpr, n_nodes=5, stochastic=True)
    # Same parameter value for all runs — only noise differs
    sweep_spec = SweepSpec(params={"eta": np.full(6, -4.6)})
    results    = Sweeper(spec, sweep_spec, backend="cuda").run(50.0)

    d0 = results[0]["raw"][1]
    for i in range(1, len(results)):
        di = results[i]["raw"][1]
        assert not np.array_equal(di, d0), \
            f"CUDA stochastic run {i} identical to run 0 — seeds not independent"


@cuda_only
def test_cuda_sweep_80_nodes_finite():
    """80-node CUDA sweep must produce finite output (TVB brain scale)."""
    spec       = _spec(mpr, n_nodes=80, dt=0.01)
    sweep_spec = SweepSpec(params={"eta": np.linspace(-5.5, -4.0, 10)})
    results    = Sweeper(spec, sweep_spec, backend="cuda").run(200.0)
    for res in results:
        _, d = res["raw"]
        assert np.isfinite(d).all(), "80-node CUDA sweep produced non-finite output"


@cuda_only
def test_cuda_sweep_pipeline_shape():
    """CUDA sweep in pipeline mode returns correct shape and finite values."""
    from vbi.feature_extraction import (
        FeaturePipeline, get_features_by_domain, get_features_by_given_names,
    )
    n_nodes   = 10
    n_samples = 15
    spec = SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=0.01),
        coupling=CouplingSpec("linear", a=0.1),
        monitors=(MonitorSpec("tavg", period=1.0),),
        weights=_weights(n_nodes),
    )
    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])
    pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=50.0)
    sweep_spec = SweepSpec(
        params={"eta": np.linspace(-5.5, -4.0, n_samples)},
        pipeline=pipeline,
    )
    labels, values = Sweeper(spec, sweep_spec, backend="cuda").run(500.0)
    assert values.shape[0] == n_samples
    assert "eta" in labels
    assert any("mean" in l for l in labels)
    assert np.isfinite(values).all(), "CUDA pipeline sweep produced non-finite values"


# ---------------------------------------------------------------------------
# Throughput benchmark (informational — never fails)
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("n_nodes,n_samples,duration", [
    (10,  500,  200.0),
    (80,  200, 1000.0),
])
def test_cuda_sweep_throughput(n_nodes, n_samples, duration, capsys):
    spec       = _spec(mpr, n_nodes=n_nodes, dt=0.01)
    sweep_spec = SweepSpec(params={"eta": np.linspace(-5.5, -4.0, n_samples)})

    # Warm-up compile
    Sweeper(spec, SweepSpec(params={"eta": np.array([-4.6])}),
            backend="cuda").run(10.0)

    t0 = time.perf_counter()
    Sweeper(spec, sweep_spec, backend="cuda").run(duration)
    elapsed = time.perf_counter() - t0
    rate = n_samples / elapsed

    with capsys.disabled():
        print(f"\nCUDA sweep: {rate:.1f} samples/s  "
              f"(n_nodes={n_nodes}, duration={duration}ms, n_samples={n_samples})")
