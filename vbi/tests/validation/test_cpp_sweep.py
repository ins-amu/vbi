"""
C++ sweeper tests.

1. Serial sweep matches individual CppSimulator runs (correctness).
2. Parallel sweep matches serial sweep (parallelism correctness).
3. Sweep works for JR and WilsonCowan (model coverage).
4. C++ sweep raw output matches Numba sweep (cross-backend validation).
5. Pipeline mode produces correct shapes and values.
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
from vbi.simulator.models.reduced_wong_wang import reduced_wong_wang
from vbi.simulator.models.wong_wang_exc_inh import wong_wang_exc_inh

pytestmark = pytest.mark.slow



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


# ---------------------------------------------------------------------------
# 6. batch_size: batched run must match unbatched run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_batch_size_serial_matches_unbatched(batch_size):
    """Serial run with batch_size must produce bit-identical results to no batching."""
    n_samples = 8
    spec = _base_spec(mpr, n_nodes=5, dt=0.01)
    sweep_spec = SweepSpec(params={"eta": np.linspace(-5.5, -4.0, n_samples)})
    sweeper = CppSweeper(spec, sweep_spec)

    ref     = sweeper.run_serial(30.0)                          # default (no batching)
    batched = sweeper.run_serial(30.0, batch_size=batch_size)

    for i in range(n_samples):
        _, d_ref = ref[i]["raw"]
        _, d_bat = batched[i]["raw"]
        np.testing.assert_array_equal(
            d_bat, d_ref,
            err_msg=f"batch_size={batch_size}: result[{i}] differs from unbatched",
        )


@pytest.mark.parametrize("batch_size", [2, 5])
def test_batch_size_parallel_matches_serial(batch_size):
    """Parallel batched run must match serial unbatched run."""
    n_samples = 6
    spec = _base_spec(mpr, n_nodes=5, dt=0.01)
    sweep_spec = SweepSpec(params={"eta": np.linspace(-5.5, -4.0, n_samples)})
    sweeper = CppSweeper(spec, sweep_spec, n_workers=2)

    serial   = sweeper.run_serial(30.0)
    parallel = sweeper.run_parallel(30.0, batch_size=batch_size)

    for i in range(n_samples):
        _, d_ser = serial[i]["raw"]
        _, d_par = parallel[i]["raw"]
        np.testing.assert_array_equal(
            d_par, d_ser,
            err_msg=f"batch_size={batch_size}: parallel result[{i}] differs from serial",
        )


# ---------------------------------------------------------------------------
# 7. Sweep over ReducedWongWang and WongWangExcInh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,param_name,values,coup_a,dt", [
    (reduced_wong_wang,  "w",   np.linspace(0.4, 0.8, 4), 0.02, 0.1),
    (wong_wang_exc_inh,  "w_p", np.linspace(0.7, 1.0, 4), 0.02, 0.1),
])
def test_rww_sweep_serial_finite(model, param_name, values, coup_a, dt):
    """RWW / WWEX serial sweep completes with finite output."""
    spec = _base_spec(model, n_nodes=5, dt=dt, coup_a=coup_a)
    sweeper = CppSweeper(spec, SweepSpec(params={param_name: values}))
    results = sweeper.run_serial(30.0)

    assert len(results) == len(values)
    for res in results:
        _, d = res["raw"]
        assert np.all(np.isfinite(d)), \
            f"{model.name} sweep produced non-finite output"


# ---------------------------------------------------------------------------
# 8. Cross-backend: C++ sweep must match Numba sweep (M2 "done when" criterion)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,param_name,values,coup_a,dt", [
    (mpr,          "eta", np.linspace(-5.0, -4.0, 5), 0.1,  0.01),
    (jansen_rit,   "mu",  np.linspace(0.15, 0.30, 4), 0.01, 0.1),
    (wilson_cowan, "P",   np.linspace(0.0,  0.3,  4), 0.05, 0.1),
])
def test_cpp_sweep_matches_numba_sweep(model, param_name, values, coup_a, dt):
    """C++ serial sweep raw output must match Numba sweep to rtol=1e-4."""
    pytest.importorskip("numba", reason="numba not installed")
    n_nodes  = 5
    duration = 30.0
    spec = _base_spec(model, n_nodes=n_nodes, dt=dt, coup_a=coup_a)
    sweep_spec = SweepSpec(params={param_name: values})

    cpp_results = CppSweeper(spec, sweep_spec).run_serial(duration)
    nb_results  = Sweeper(spec, sweep_spec, backend="numba").run(duration)

    for i, val in enumerate(values):
        _, d_cpp = cpp_results[i]["raw"]
        _, d_nb  = nb_results[i]["raw"]
        np.testing.assert_allclose(
            d_cpp, d_nb, rtol=1e-4, atol=1e-8,
            err_msg=f"{model.name} {param_name}={val:.3f}: "
                    f"C++ sweep[{i}] differs from Numba sweep[{i}]",
        )


# ---------------------------------------------------------------------------
# 9. Pipeline mode: C++ sweep with FeaturePipeline produces correct output
# ---------------------------------------------------------------------------

def test_cpp_sweep_pipeline_shape():
    """CppSweeper in pipeline mode returns (labels, values) with correct shape."""
    from vbi.feature_extraction import (
        FeaturePipeline, get_features_by_domain, get_features_by_given_names,
    )
    n_nodes  = 4
    n_samples = 6
    duration  = 100.0
    spec = SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=0.01, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=0.1, b=0.0),
        monitors=(MonitorSpec(kind="tavg", period=1.0),),
        weights=_weights(n_nodes),
    )
    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])
    pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=50.0)
    sweep_spec = SweepSpec(
        params={"eta": np.linspace(-5.5, -4.0, n_samples)},
        pipeline=pipeline,
    )
    sweeper = CppSweeper(spec, sweep_spec)
    labels, values = sweeper.run(duration)

    assert values.shape[0] == n_samples, \
        f"Expected {n_samples} rows, got {values.shape[0]}"
    assert "eta" in labels
    assert any("mean" in l for l in labels)
    assert np.all(np.isfinite(values)), "Pipeline output contains non-finite values"


def test_cpp_sweep_pipeline_matches_numpy_sweep():
    """C++ pipeline sweep feature values must match NumPy pipeline sweep."""
    from vbi.feature_extraction import (
        FeaturePipeline, get_features_by_domain, get_features_by_given_names,
    )
    n_nodes   = 4
    n_samples = 5
    duration  = 100.0
    spec = SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=0.01, stochastic=False),
        coupling=CouplingSpec(kind="linear", a=0.1, b=0.0),
        monitors=(MonitorSpec(kind="tavg", period=1.0),),
        weights=_weights(n_nodes),
    )
    cfg = get_features_by_domain("statistical")
    cfg = get_features_by_given_names(cfg, ["calc_mean", "calc_std"])
    pipeline = FeaturePipeline(cfg, signal="tavg", t_cut=50.0)
    sweep_spec = SweepSpec(
        params={"eta": np.linspace(-5.5, -4.0, n_samples)},
        pipeline=pipeline,
    )

    cpp_labels, cpp_vals = CppSweeper(spec, sweep_spec).run(duration)
    np_labels,  np_vals  = Sweeper(spec, sweep_spec, backend="numpy").run(duration)

    assert cpp_labels == np_labels, "Label mismatch between C++ and NumPy pipeline"
    np.testing.assert_allclose(
        cpp_vals, np_vals, rtol=1e-4, atol=1e-8,
        err_msg="C++ pipeline sweep features differ from NumPy pipeline sweep",
    )
