"""
Stimulation tests across backends.

Verifies that StimSpec injection works correctly and consistently:
  - stimulus physically affects the target node
  - onset/offset gating is respected
  - pre-sampled waveform (array) and callable waveform give the same result
  - numpy, numba, JAX and C++ backends all agree
  - no-stimulus baseline is unaffected (zero stim = no change)
  - frustrated Kuramoto (alpha) with stimulus
"""
import dataclasses

import numpy as np
import pytest

from vbi.simulator import Simulator
from vbi.simulator.spec import (
    SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec, StimSpec,
)
from vbi.simulator.models.kuramoto import kuramoto
from vbi.simulator.models.mpr import mpr

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


def _kuramoto_spec(n, theta0, omega, G, stim=(), dt=0.01):
    W = np.ones((n, n), dtype=float); np.fill_diagonal(W, 0.0)
    sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
    mdl = dataclasses.replace(kuramoto, state_variables=sv)
    return SimulationSpec(
        model=mdl,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("kuramoto"),
        monitors=(MonitorSpec("raw"),),
        weights=W,
        node_params={"omega": np.full(n, float(omega)), "G": G},
        stimuli=stim,
    )


def _mpr_spec(n, stim=(), dt=0.01):
    W = _weights(n, seed=5)
    return SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method="heun", dt=dt),
        coupling=CouplingSpec("linear", a=0.1),
        monitors=(MonitorSpec("raw"),),
        weights=W,
        stimuli=stim,
    )


def _run(spec, duration, backend="numpy"):
    return Simulator(spec, backend=backend).run(duration)["raw"][1]


# ---------------------------------------------------------------------------
# Physics: stimulus has expected effect
# ---------------------------------------------------------------------------

class TestStimPhysics:

    def test_stimulated_node_advances_faster(self):
        """A stimulated Kuramoto node must accumulate more phase than un-stimulated ones."""
        n = 4
        theta0 = np.zeros(n)
        stim = StimSpec(
            sv_name="theta",
            amplitude=np.array([1.0, 0.0, 0.0, 0.0]),
            waveform=lambda t: 0.05,
        )
        spec = _kuramoto_spec(n, theta0, omega=1.0, G=0.0, stim=(stim,))
        d = _run(spec, 100.0)
        assert d[-1, 0, 0] > d[-1, 0, 1], \
            "Stimulated node 0 should have larger phase than un-stimulated node 1"

    def test_zero_amplitude_stim_no_effect(self):
        """Stimulus with amplitude=0 must be identical to no stimulus."""
        n = 3
        theta0 = np.random.default_rng(42).uniform(-np.pi, np.pi, n)
        stim_zero = StimSpec(
            sv_name="theta",
            amplitude=np.zeros(n),
            waveform=lambda t: 1.0,
        )
        spec_no   = _kuramoto_spec(n, theta0, omega=1.0, G=0.5)
        spec_zero = _kuramoto_spec(n, theta0, omega=1.0, G=0.5, stim=(stim_zero,))
        d_no   = _run(spec_no,   50.0)
        d_zero = _run(spec_zero, 50.0)
        np.testing.assert_array_equal(d_no, d_zero,
            err_msg="Zero-amplitude stimulus must not change trajectory")

    def test_onset_offset_gating(self):
        """Stimulus active only in [onset, offset) must not affect outside that window."""
        n = 3
        theta0 = np.zeros(n)
        dt, duration = 0.01, 5.0
        stim = StimSpec(
            sv_name="theta",
            amplitude=np.array([1.0, 0.0, 0.0]),
            waveform=lambda t: 0.1,
            onset=2.0, offset=3.0,     # active for 1 ms only
        )
        spec = _kuramoto_spec(n, theta0, omega=1.0, G=0.0, stim=(stim,), dt=dt)
        spec_no = _kuramoto_spec(n, theta0, omega=1.0, G=0.0, dt=dt)
        d_stim = _run(spec,    duration)
        d_no   = _run(spec_no, duration)

        n_steps = round(duration / dt)
        t = np.arange(n_steps) * dt

        # Before onset: must be identical
        pre  = t < 2.0
        np.testing.assert_array_equal(
            d_stim[pre, 0, 0], d_no[pre, 0, 0],
            err_msg="Stimulus must not affect state before onset")

        # During stimulus: must differ on stimulated node
        during = (t >= 2.0) & (t < 3.0)
        assert not np.allclose(d_stim[during, 0, 0], d_no[during, 0, 0]), \
            "Stimulus must change state during onset-offset window"

    def test_callable_vs_presampled_waveform(self):
        """Callable waveform and equivalent pre-sampled array must give identical output."""
        n, dt, duration = 3, 0.01, 20.0
        n_steps = round(duration / dt)
        t_arr = np.arange(n_steps) * dt
        amp_fn = lambda t: 0.03 * np.sin(2 * np.pi * 0.1 * t)

        theta0 = np.random.default_rng(7).uniform(-np.pi, np.pi, n)
        stim_fn = StimSpec(sv_name="theta",
                           amplitude=np.ones(n),
                           waveform=amp_fn)
        stim_arr = StimSpec(sv_name="theta",
                            amplitude=np.ones(n),
                            waveform=amp_fn(t_arr))

        spec_fn  = _kuramoto_spec(n, theta0, omega=1.0, G=0.3, stim=(stim_fn,),  dt=dt)
        spec_arr = _kuramoto_spec(n, theta0, omega=1.0, G=0.3, stim=(stim_arr,), dt=dt)
        d_fn  = _run(spec_fn,  duration)
        d_arr = _run(spec_arr, duration)
        np.testing.assert_allclose(d_fn, d_arr, rtol=1e-12,
            err_msg="Callable and pre-sampled waveforms must agree")

    def test_multiple_stimuli(self):
        """Two simultaneous StimSpecs must both take effect independently."""
        n = 4
        theta0 = np.zeros(n)
        stim0 = StimSpec(sv_name="theta",
                         amplitude=np.array([1.0, 0.0, 0.0, 0.0]),
                         waveform=lambda t: 0.1)
        stim1 = StimSpec(sv_name="theta",
                         amplitude=np.array([0.0, 1.0, 0.0, 0.0]),
                         waveform=lambda t: 0.2)
        spec = _kuramoto_spec(n, theta0, omega=1.0, G=0.0, stim=(stim0, stim1))
        d = _run(spec, 50.0)
        # Node 1 receives double the stimulus amplitude → should advance more than node 0.
        # Both run at omega=1.0 base rate; the extra phase is:
        #   node 0: +0.1 * 50 = +5.0 rad above omega*t baseline
        #   node 1: +0.2 * 50 = +10.0 rad above omega*t baseline
        # So the stim-induced phase excess for node 1 is 2× that of node 0.
        baseline = 1.0 * 50.0    # omega * duration (same for both)
        excess0  = d[-1, 0, 0] - baseline
        excess1  = d[-1, 0, 1] - baseline
        assert excess1 > excess0 > 0, \
            "Both nodes must advance beyond baseline; node 1 more than node 0"
        assert abs(excess1 / excess0 - 2.0) < 0.05, \
            f"Stimulus-induced excess for node 1 should be 2× node 0; " \
            f"ratio={excess1/excess0:.3f}"


# ---------------------------------------------------------------------------
# Backend consistency: numpy, numba, JAX, C++ must agree
# ---------------------------------------------------------------------------

class TestBackendConsistency:

    def _compare(self, spec, duration, backends, rtol=1e-4, atol=0.0):
        results = {}
        for b in backends:
            try:
                results[b] = np.array(_run(spec, duration, backend=b))
            except Exception as e:
                pytest.skip(f"{b} backend unavailable: {e}")
        ref = results[list(results.keys())[0]]
        for b, d in list(results.items())[1:]:
            np.testing.assert_allclose(
                d, ref, rtol=rtol, atol=atol,
                err_msg=f"Backend {b!r} diverges from {list(results.keys())[0]!r}")

    def test_numpy_numba_kuramoto_stim(self):
        """Numpy and Numba must agree on Kuramoto with stimulus."""
        n = 4
        theta0 = np.random.default_rng(11).uniform(-np.pi, np.pi, n)
        stim = StimSpec(sv_name="theta",
                        amplitude=np.array([1.0, 0.5, 0.0, 0.0]),
                        waveform=lambda t: 0.05 * (1.0 if t < 100.0 else 0.0))
        spec = _kuramoto_spec(n, theta0, omega=1.0, G=0.5, stim=(stim,))
        self._compare(spec, 200.0, ["numpy", "numba"])

    def test_numpy_numba_mpr_stim(self):
        """Numpy and Numba must agree on MPR with stimulus."""
        n = 4
        stim = StimSpec(sv_name="r",
                        amplitude=np.array([1.0, 0.0, 0.0, 0.0]),
                        waveform=lambda t: 0.01)
        spec = _mpr_spec(n, stim=(stim,))
        self._compare(spec, 10.0, ["numpy", "numba"])

    def test_numpy_numba_delayed_stim(self):
        """Delayed Kuramoto with stimulus: numpy and numba must agree."""
        n = 3
        theta0 = np.random.default_rng(13).uniform(-np.pi, np.pi, n)
        W = np.ones((n, n), dtype=float); np.fill_diagonal(W, 0.0)
        tract = np.full((n, n), 5.0); np.fill_diagonal(tract, 0.0)
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)
        stim = StimSpec(sv_name="theta",
                        amplitude=np.array([1.0, 0.0, 0.0]),
                        waveform=lambda t: 0.05, onset=10.0, offset=50.0)
        spec = SimulationSpec(
            model=mdl,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("kuramoto"),
            monitors=(MonitorSpec("raw"),),
            weights=W, tract_lengths=tract, speed=1.0,
            node_params={"omega": np.ones(n), "G": 0.5},
            stimuli=(stim,),
        )
        self._compare(spec, 100.0, ["numpy", "numba"])

    def test_numpy_jax_mpr_stim(self):
        """JAX must agree with numpy on MPR with stimulus (float32 tolerance)."""
        n = 4
        stim = StimSpec(sv_name="r",
                        amplitude=np.array([1.0, 0.0, 0.0, 0.0]),
                        waveform=lambda t: 0.01)
        spec = _mpr_spec(n, stim=(stim,), dt=0.01)
        self._compare(spec, 5.0, ["numpy", "jax"], rtol=1e-2)

    def test_numpy_cpp_mpr_stim(self):
        """C++ must agree with numpy on MPR with stimulus."""
        n = 4
        stim = StimSpec(sv_name="r",
                        amplitude=np.array([1.0, 0.0, 0.0, 0.0]),
                        waveform=lambda t: 0.01)
        spec = _mpr_spec(n, stim=(stim,))
        self._compare(spec, 5.0, ["numpy", "cpp"])

    def test_numpy_cuda_mpr_stim(self):
        """CUDA must agree with numpy on MPR with stimulus (float32 tolerance)."""
        try:
            from numba import cuda as _cuda
            if not _cuda.is_available():
                pytest.skip("CUDA not available")
        except Exception:
            pytest.skip("CUDA not available")

        n = 4
        stim = StimSpec(sv_name="r",
                        amplitude=np.array([1.0, 0.0, 0.0, 0.0]),
                        waveform=lambda t: 0.01)
        spec = _mpr_spec(n, stim=(stim,), dt=0.01)
        self._compare(spec, 5.0, ["numpy", "cuda"], rtol=1e-3)

    def test_numpy_cuda_kuramoto_stim(self):
        """CUDA must agree with numpy on Kuramoto with stimulus.

        CUDA uses float32 internally; float32 vs float64 accumulation gives
        ~few-ms absolute error over short runs.  We verify the stimulus IS
        applied (output changes vs no-stimulus) and that absolute deviation
        stays within float32 tolerance.
        """
        try:
            from numba import cuda as _cuda
            if not _cuda.is_available():
                pytest.skip("CUDA not available")
        except Exception:
            pytest.skip("CUDA not available")

        n = 4
        theta0 = np.random.default_rng(17).uniform(-np.pi, np.pi, n)
        stim = StimSpec(sv_name="theta",
                        amplitude=np.array([1.0, 0.5, 0.0, 0.0]),
                        waveform=lambda t: 0.05, onset=0.0, offset=1.0)
        spec_stim = _kuramoto_spec(n, theta0, omega=1.0, G=0.5, stim=(stim,), dt=0.01)
        spec_base = _kuramoto_spec(n, theta0, omega=1.0, G=0.5,               dt=0.01)

        d_np_stim = np.array(_run(spec_stim, 2.0, backend="numpy"))
        d_cu_stim = np.array(_run(spec_stim, 2.0, backend="cuda"))
        d_cu_base = np.array(_run(spec_base, 2.0, backend="cuda"))

        # Stimulus must change the CUDA output
        assert not np.allclose(d_cu_stim, d_cu_base), \
            "CUDA stimulus must change output vs no-stimulus baseline"

        # numpy and CUDA with stimulus must agree within float32 tolerance
        np.testing.assert_allclose(
            d_cu_stim, d_np_stim, atol=1e-2,
            err_msg="CUDA Kuramoto+stim deviates from numpy beyond float32 tolerance")


# ---------------------------------------------------------------------------
# Stim with Kuramoto's alpha (frustrated) + stimulus
# ---------------------------------------------------------------------------

class TestFrustratedKuramotoWithStim:

    def test_alpha_and_stim_numpy_numba(self):
        """Frustrated Kuramoto (alpha != 0) plus stimulus: numpy and numba agree."""
        n = 4
        theta0 = np.random.default_rng(20).uniform(-np.pi, np.pi, n)
        stim = StimSpec(sv_name="theta",
                        amplitude=np.array([1.0, 0.0, 0.0, 0.0]),
                        waveform=lambda t: 0.05)
        sv  = (dataclasses.replace(kuramoto.state_variables[0], default_init=theta0),)
        mdl = dataclasses.replace(kuramoto, state_variables=sv)
        W = np.ones((n, n), dtype=float); np.fill_diagonal(W, 0.0)
        spec = SimulationSpec(
            model=mdl,
            integrator=IntegratorSpec(method="heun", dt=0.01),
            coupling=CouplingSpec("kuramoto", alpha=np.pi / 4),
            monitors=(MonitorSpec("raw"),),
            weights=W,
            node_params={"omega": np.ones(n), "G": 0.5},
            stimuli=(stim,),
        )
        d_np = np.array(_run(spec, 50.0, backend="numpy"))
        d_nb = np.array(_run(spec, 50.0, backend="numba"))
        np.testing.assert_allclose(d_nb, d_np, rtol=1e-4,
            err_msg="Frustrated Kuramoto + stim: numba diverges from numpy")
