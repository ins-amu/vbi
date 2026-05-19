"""
M1 validation: Numba CPU backend — Jansen-Rit model.

Gold standard: NumPy baseline (validated against TVB in test_jansen_rit_numpy.py).
Deterministic Numba results must match NumPy to rtol=1e-4.
"""
import dataclasses

import numpy as np
import pytest

from vbi.simulator import Simulator
from vbi.simulator.spec import MonitorSpec, IntegratorSpec
from .test_jansen_rit_numpy import make_jr_spec

numba = pytest.importorskip("numba", reason="numba not installed")


def _np(spec, duration):
    return Simulator(spec, backend="numpy").run(duration)


def _nb(spec, duration):
    return Simulator(spec, backend="numba").run(duration)


def _rng_weights(n, seed=42):
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(min=1.0)
    return W


class TestJansenRitDeterministicMatchesNumPy:
    @pytest.mark.parametrize("n_nodes", [1, 2, 4])
    @pytest.mark.parametrize("method", ["heun", "euler"])
    def test_raw_matches_numpy(self, n_nodes, method):
        W = _rng_weights(n_nodes)
        spec = make_jr_spec(W, method=method)
        t_np, d_np = _np(spec, 20.0)["raw"]
        t_nb, d_nb = _nb(spec, 20.0)["raw"]
        assert d_np.shape == d_nb.shape
        np.testing.assert_allclose(
            d_nb, d_np, rtol=1e-4,
            err_msg=f"JR Numba diverges from NumPy (n_nodes={n_nodes}, {method})",
        )

    def test_with_delays(self):
        n = 3
        rng = np.random.default_rng(7)
        W = np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(W, 0.0)
        D = np.abs(rng.standard_normal((n, n))) * 20.0
        np.fill_diagonal(D, 0.0)
        spec = make_jr_spec(W, tract_lengths=D)
        t_np, d_np = _np(spec, 20.0)["raw"]
        t_nb, d_nb = _nb(spec, 20.0)["raw"]
        np.testing.assert_allclose(d_nb, d_np, rtol=1e-4)

    def test_state_is_finite(self):
        W = _rng_weights(4)
        spec = make_jr_spec(W)
        _, d = _nb(spec, 50.0)["raw"]
        assert np.isfinite(d).all()


class TestJansenRitStochastic:
    def _stoch_spec(self, n_nodes=2, seed=0):
        W = _rng_weights(n_nodes)
        spec = make_jr_spec(W)
        # JR has one noise variable: y4
        return dataclasses.replace(
            spec,
            integrator=dataclasses.replace(
                spec.integrator,
                stochastic=True,
                noise_nsig=np.array([1e-4]),
                noise_seed=seed,
            ),
        )

    def test_stochastic_runs_without_error(self):
        _, d = _nb(self._stoch_spec(), 20.0)["raw"]
        assert np.isfinite(d).all()

    def test_same_seed_reproduces(self):
        spec = self._stoch_spec(seed=7)
        _, d1 = _nb(spec, 20.0)["raw"]
        _, d2 = _nb(spec, 20.0)["raw"]
        np.testing.assert_array_equal(d1, d2, err_msg="Same seed must reproduce")

    def test_different_seeds_differ(self):
        _, d1 = _nb(self._stoch_spec(seed=0), 20.0)["raw"]
        _, d2 = _nb(self._stoch_spec(seed=1), 20.0)["raw"]
        assert not np.allclose(d1, d2), "Different seeds must produce different noise"
