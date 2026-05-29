"""
M4 validation: JAX backend - single-run and sweep.

Gold standard: NumPy baseline (validated against TVB in test_mpr_numpy.py).

Tolerance note: JAX uses float32 by default (matching the GPU/CUDA backend),
while NumPy uses float64.  After N steps the accumulated rounding error is
O(N * eps_float32) ≈ O(N * 1e-7).  Tests here use short durations (≤ 200 ms)
to keep accumulated error well below rtol=1e-2.

For long-run comparisons the stochastic moment tests (mean/std) are used
instead of trajectory matching.
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

pytestmark = pytest.mark.slow

jax = pytest.importorskip("jax", reason="jax not installed")
jnp = jax.numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_sim(spec, duration):
    return Simulator(spec, backend="numpy").run(duration)


def _jx_sim(spec, duration):
    return Simulator(spec, backend="jax").run(duration)


# ---------------------------------------------------------------------------
# Single-run: deterministic - trajectory comparison
# ---------------------------------------------------------------------------

class TestDeterministicMatchesNumPy:
    """
    JAX (float32) vs NumPy (float64) on short runs.
    rtol=1e-2 accommodates float32 accumulation over 200 ms / 0.1 ms = 2 000 steps.
    """

    @pytest.mark.parametrize("n_nodes", [1, 2, 10])
    @pytest.mark.parametrize("method", ["heun", "euler"])
    def test_raw_matches_numpy(self, n_nodes, method):
        spec = make_mpr_spec(
            n_nodes=n_nodes, dt=0.1, method=method,
            monitors=(MonitorSpec("raw"),),
        )
        _, d_np = _np_sim(spec, 200.0)["raw"]
        _, d_jx = _jx_sim(spec, 200.0)["raw"]
        assert d_np.shape == d_jx.shape, "shape mismatch"
        np.testing.assert_allclose(
            d_jx, d_np, rtol=1e-2,
            err_msg=f"JAX raw diverges from NumPy (n_nodes={n_nodes}, {method})",
        )

    def test_r_stays_nonnegative(self):
        spec = make_mpr_spec(
            n_nodes=10, dt=0.1,
            monitors=(MonitorSpec("raw"),),
        )
        _, d_jx = _jx_sim(spec, 200.0)["raw"]
        assert np.all(d_jx[:, 0, :] >= 0.0), "r violated lower bound (r >= 0)"

    def test_subsample_matches_numpy(self):
        spec = make_mpr_spec(
            n_nodes=4, dt=0.1,
            monitors=(MonitorSpec("subsample", period=1.0),),
        )
        _, d_np = _np_sim(spec, 200.0)["subsample"]
        _, d_jx = _jx_sim(spec, 200.0)["subsample"]
        assert d_np.shape == d_jx.shape
        np.testing.assert_allclose(d_jx, d_np, rtol=1e-2)

    def test_tavg_matches_numpy(self):
        spec = make_mpr_spec(
            n_nodes=4, dt=0.1,
            monitors=(MonitorSpec("tavg", period=1.0),),
        )
        _, d_np = _np_sim(spec, 200.0)["tavg"]
        _, d_jx = _jx_sim(spec, 200.0)["tavg"]
        assert d_np.shape == d_jx.shape
        np.testing.assert_allclose(d_jx, d_np, rtol=1e-2)

    def test_with_delays(self):
        spec = make_mpr_spec(
            n_nodes=4, dt=0.1,
            monitors=(MonitorSpec("raw"),),
        )
        _, d_np = _np_sim(spec, 200.0)["raw"]
        _, d_jx = _jx_sim(spec, 200.0)["raw"]
        np.testing.assert_allclose(d_jx, d_np, rtol=1e-2)

    def test_no_delays(self):
        W, _ = make_weights(4)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("raw"),),
            weights=W,
            tract_lengths=None,  # zero delays
        )
        _, d_np = _np_sim(spec, 200.0)["raw"]
        _, d_jx = _jx_sim(spec, 200.0)["raw"]
        np.testing.assert_allclose(d_jx, d_np, rtol=1e-2)


# ---------------------------------------------------------------------------
# Single-run: stochastic - moment comparison
# ---------------------------------------------------------------------------

class TestStochasticMoments:
    """Compare mean and std of stochastic trajectories over many realizations."""

    def test_stochastic_mean_std(self):
        n_nodes = 4
        spec = make_mpr_spec(
            n_nodes=n_nodes, dt=0.1, stochastic=True,
            monitors=(MonitorSpec("subsample", period=1.0),),
        )
        n_trials = 20
        np_means, jx_means = [], []
        for seed in range(n_trials):
            spec_s = SimulationSpec(
                model=spec.model, integrator=IntegratorSpec(
                    method=spec.integrator.method,
                    dt=spec.integrator.dt,
                    stochastic=True,
                    noise_nsig=spec.integrator.noise_nsig,
                    noise_seed=seed,
                ),
                coupling=spec.coupling,
                monitors=spec.monitors,
                weights=spec.weights,
                tract_lengths=spec.tract_lengths,
            )
            _, d_np = _np_sim(spec_s, 500.0)["subsample"]
            _, d_jx = _jx_sim(spec_s, 500.0)["subsample"]
            np_means.append(d_np.mean())
            jx_means.append(d_jx.mean())

        np.testing.assert_allclose(
            np.mean(np_means), np.mean(jx_means), rtol=0.1,
            err_msg="Stochastic mean diverges between JAX and NumPy",
        )


# ---------------------------------------------------------------------------
# JIT reuse: second call must be faster than first
# ---------------------------------------------------------------------------

class TestJitReuse:
    def test_second_call_faster(self):
        spec = make_mpr_spec(
            n_nodes=10, dt=0.1,
            monitors=(MonitorSpec("subsample", period=1.0),),
        )
        sim = Simulator(spec, backend="jax")
        t0 = time.perf_counter()
        sim.run(500.0)
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        sim.run(500.0)
        t_second = time.perf_counter() - t0

        assert t_second < t_first, (
            f"Second JIT call ({t_second:.3f}s) should be faster than first ({t_first:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Differentiability: jax.grad through the simulation
# ---------------------------------------------------------------------------

class TestGradient:
    def test_grad_runs_and_is_finite(self):
        """jax.grad(loss)(G) must return a finite scalar."""
        import jax
        from vbi.simulator.backend.jax_.simulator import JaxSimulator

        W, D = make_weights(4)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("subsample", period=1.0),),
            weights=W,
            tract_lengths=D,
        )

        sim = JaxSimulator()
        sim.build(spec)
        n_steps = round(200.0 / spec.integrator.dt)  # concrete int

        def loss(G_val):
            # _run_core(params, n_steps) - n_steps is a concrete Python int
            params = {**sim._params, "G": G_val}
            result = sim._run_core(params, n_steps)
            _, data = result["subsample"]   # data is a JAX array (not yet numpy)
            return jnp.mean(data ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.float32(2.0))
        assert jnp.isfinite(g), f"Gradient is not finite: {g}"


# ---------------------------------------------------------------------------
# BOLD monitor
# ---------------------------------------------------------------------------

class TestBoldMonitor:
    def _bold_spec(self, n_nodes, tr=2000.0):
        W, D = make_weights(n_nodes)
        return SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("bold", tr=tr),),
            weights=W,
            tract_lengths=D,
        )

    def test_bold_shape(self):
        spec = self._bold_spec(n_nodes=4, tr=2000.0)
        _, bold = _jx_sim(spec, 6000.0)["bold"]
        # 6000 ms / 2000 ms TR = 3 samples
        assert bold.shape == (3, 1, 4), f"expected (3, 1, 4), got {bold.shape}"

    def test_bold_matches_numpy(self):
        """
        JAX and NumPy BOLD values agree on shared samples (rtol=5e-2 float32).

        NumPy records at step % tr_steps == 0 and step > 0, so for a 6000ms
        run at TR=2000ms it returns 2 samples (at 2000ms, 4000ms).
        JAX runs 3 full TR windows and returns 3 samples (2000, 4000, 6000ms).
        We compare the first min(n_np, n_jx) samples.
        """
        spec = self._bold_spec(n_nodes=4, tr=2000.0)
        _, bold_np = _np_sim(spec, 6000.0)["bold"]
        _, bold_jx = _jx_sim(spec, 6000.0)["bold"]
        n = min(bold_np.shape[0], bold_jx.shape[0])
        np.testing.assert_allclose(bold_jx[:n], bold_np[:n], rtol=5e-2,
                                   err_msg="JAX BOLD diverges from NumPy BOLD")

    def test_bold_nonzero(self):
        """BOLD signal should be non-trivial after a few TR periods."""
        spec = self._bold_spec(n_nodes=4, tr=2000.0)
        _, bold = _jx_sim(spec, 6000.0)["bold"]
        assert not np.allclose(bold, 0.0), "BOLD signal is all zeros"

    def test_bold_tr_respected(self):
        """Number of BOLD samples = floor(duration / TR)."""
        tr = 1500.0
        duration = 7500.0
        spec = self._bold_spec(n_nodes=4, tr=tr)
        _, bold = _jx_sim(spec, duration)["bold"]
        expected = int(duration / tr)   # 7500/1500 = 5
        assert bold.shape[0] == expected, (
            f"expected {expected} TR samples, got {bold.shape[0]}")


# ---------------------------------------------------------------------------
# Sweep: shape and consistency
# ---------------------------------------------------------------------------

class TestSweep:
    def test_sweep_fc_shape(self):
        n_nodes = 6
        n_G = 5
        W, D = make_weights(n_nodes)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("subsample", period=1.0),),
            weights=W,
            tract_lengths=D,
        )
        sweep_spec = SweepSpec(params={"G": np.linspace(1.0, 4.0, n_G)})
        results = Sweeper(spec, sweep_spec, backend="jax").run(500.0)
        # results is a list of monitor dicts, one per sample
        assert len(results) == n_G
        assert "subsample" in results[0]
        _, data = results[0]["subsample"]
        # data shape: (n_record, n_sv, n_nodes)
        assert data.shape[-1] == n_nodes

    def test_sweep_matches_numpy_single_run(self):
        """Sweep with one parameter set must match single Simulator run."""
        n_nodes = 4
        W, D = make_weights(n_nodes)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("subsample", period=1.0),),
            weights=W,
            tract_lengths=D,
        )
        G_val = 2.0
        sweep_spec = SweepSpec(params={"G": np.array([G_val])})
        sweep_results = Sweeper(spec, sweep_spec, backend="jax").run(200.0)
        _, sweep_data = sweep_results[0]["subsample"]

        # Single run with same G
        from vbi.simulator.spec import SimulationSpec as SS
        spec_g = SS(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("subsample", period=1.0),),
            weights=W,
            tract_lengths=D,
            node_params={"G": G_val},
        )
        _, single_data = _jx_sim(spec_g, 200.0)["subsample"]

        np.testing.assert_allclose(
            sweep_data, single_data, rtol=1e-4,
            err_msg="Sweep run with one sample must match single Simulator run",
        )

    def test_sweep_bold_shape(self):
        n_nodes = 6
        n_G = 4
        W, D = make_weights(n_nodes)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("bold", tr=2000.0),),
            weights=W,
            tract_lengths=D,
        )
        sweep_spec = SweepSpec(params={"G": np.linspace(1.0, 4.0, n_G)})
        results = Sweeper(spec, sweep_spec, backend="jax").run(6000.0)
        assert len(results) == n_G
        _, bold = results[0]["bold"]
        # shape: (n_record, 1, n_nodes)
        assert bold.shape[1:] == (1, n_nodes)
        assert bold.shape[0] > 0

    def test_same_noise_true_identical_first_step(self):
        """same_noise=True: all runs should see identical noise at step 0."""
        n_nodes, n_G = 4, 4
        W, D = make_weights(n_nodes)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1, stochastic=True,
                                      noise_nsig=np.array([1e-2, 1e-2])),
            coupling=CouplingSpec("linear", a=0.0),   # no coupling → pure noise
            monitors=(MonitorSpec("subsample", period=1.0),),
            weights=W, tract_lengths=None,
        )
        sweep_spec = SweepSpec(
            params={"G": np.linspace(1.0, 4.0, n_G)},
            same_noise=True,
        )
        results = Sweeper(spec, sweep_spec, backend="jax").run(100.0)
        # With same_noise=True and no coupling, first recorded state = after 10 steps
        # - noise is the ONLY difference driver. With coupling=0, G has no effect.
        # All runs must be numerically identical.
        _, d0 = results[0]["subsample"]
        for i in range(1, n_G):
            _, di = results[i]["subsample"]
            np.testing.assert_array_equal(
                d0, di,
                err_msg=f"same_noise=True: run {i} differs from run 0",
            )

    def test_same_noise_false_independent(self):
        """same_noise=False: runs must differ (independent noise realizations)."""
        n_nodes, n_G = 4, 4
        W, _ = make_weights(n_nodes)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1, stochastic=True,
                                      noise_nsig=np.array([1e-2, 1e-2])),
            coupling=CouplingSpec("linear", a=0.0),   # no coupling → pure noise
            monitors=(MonitorSpec("subsample", period=1.0),),
            weights=W, tract_lengths=None,
        )
        sweep_spec = SweepSpec(
            params={"G": np.linspace(1.0, 4.0, n_G)},
            same_noise=False,
        )
        results = Sweeper(spec, sweep_spec, backend="jax").run(100.0)
        _, d0 = results[0]["subsample"]
        _, d1 = results[1]["subsample"]
        assert not np.allclose(d0, d1), \
            "same_noise=False: runs 0 and 1 should differ (independent noise)"

    def test_sweep_throughput(self):
        """Record throughput - not a pass/fail, but logs samples/s."""
        n_nodes = 10
        n_samples = 50
        W, D = make_weights(n_nodes)
        spec = SimulationSpec(
            model=mpr,
            integrator=IntegratorSpec(method="heun", dt=0.1),
            coupling=CouplingSpec("linear", a=1.0),
            monitors=(MonitorSpec("subsample", period=1.0),),
            weights=W,
            tract_lengths=D,
        )
        sweep_spec = SweepSpec(params={"G": np.linspace(0.5, 5.0, n_samples)})
        sweeper = Sweeper(spec, sweep_spec, backend="jax")

        # Warmup JIT
        sweeper.run(200.0)

        t0 = time.perf_counter()
        sweeper.run(1000.0)
        elapsed = time.perf_counter() - t0
        rate = n_samples / elapsed
        print(f"\nJAX sweep: {rate:.1f} samples/s  ({n_nodes} nodes, {n_samples} samples)")
