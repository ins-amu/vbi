"""
Tests for JRSigmoidalCoupling (numpy phase).

Validates that the coupling computes  c_y1 = W @ S(y1 - y2)  exactly, that
the delayed path is consistent with the instant path at zero delay, and that
running the full JR model with jr_sigmoidal coupling produces different (and
correct) results compared to the old difference-then-sigmoid convention.
"""

import numpy as np
import pytest

from vbi.simulator import Simulator
from vbi.simulator.backend.numpy_.coupling import JRSigmoidalCoupling
from vbi.simulator.backend.numpy_.history import History
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.simulator.spec import (
    CouplingSpec, IntegratorSpec, MonitorSpec, SimulationSpec,
)


# ── Shared sigmoid reference ───────────────────────────────────────────────────

NU_MAX = 0.0025
R      = 0.56
V0     = 5.52


def jr_sigmoid(v, nu_max=NU_MAX, r=R, v0=V0):
    return 2.0 * nu_max / (1.0 + np.exp(r * (v0 - v)))


# ── Unit tests: coupling math ──────────────────────────────────────────────────

class TestJRSigmoidalCouplingInstant:
    def _make(self, W):
        return JRSigmoidalCoupling(W, nu_max=NU_MAX, r=R, v0=V0)

    def test_two_nodes_formula(self):
        W  = np.array([[0.0, 0.3], [0.2, 0.0]])
        y1 = np.array([1.0, 2.0])
        y2 = np.array([0.5, 1.5])
        expected = W @ jr_sigmoid(y1 - y2)

        coup = self._make(W)
        result = coup.compute_instant(np.stack([y1, y2]))

        np.testing.assert_allclose(result[0], expected, rtol=1e-12)
        np.testing.assert_array_equal(result[1], np.zeros(2))

    def test_three_nodes_formula(self):
        rng = np.random.default_rng(0)
        W  = np.abs(rng.standard_normal((3, 3)))
        np.fill_diagonal(W, 0.0)
        y1 = rng.standard_normal(3)
        y2 = rng.standard_normal(3)
        expected = W @ jr_sigmoid(y1 - y2)

        coup = self._make(W)
        result = coup.compute_instant(np.stack([y1, y2]))

        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_differs_from_w_at_sigmoid_of_sum(self):
        # The key correctness check: W @ S(x) ≠ S(W @ x) for nonlinear S.
        W  = np.array([[0.0, 0.5], [0.4, 0.0]])
        y1 = np.array([2.0, 3.0])
        y2 = np.array([1.0, 1.0])
        diff = y1 - y2

        correct   = W @ jr_sigmoid(diff)           # TVB/JR: W @ S(x)
        incorrect = jr_sigmoid(W @ diff)           # old convention: S(W @ x)

        assert not np.allclose(correct, incorrect), (
            "W@S(x) should differ from S(W@x); test is degenerate"
        )

        coup = self._make(W)
        result = coup.compute_instant(np.stack([y1, y2]))
        np.testing.assert_allclose(result[0], correct, rtol=1e-12)


class TestJRSigmoidalCouplingDelayed:
    def _make(self, W):
        return JRSigmoidalCoupling(W, nu_max=NU_MAX, r=R, v0=V0)

    def test_delayed_formula_hand_computed(self):
        # 3 nodes, heterogeneous delays.
        # Construct delayed_state[cvar, src, tgt] directly.
        n = 3
        rng = np.random.default_rng(7)
        W = np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(W, 0.0)

        # Fake delayed state: different y1, y2 per (src, tgt) pair.
        y1_del = rng.standard_normal((n, n))  # (src, tgt)
        y2_del = rng.standard_normal((n, n))
        delayed_state = np.stack([y1_del, y2_del])  # (2, src, tgt)

        # Reference: for each tgt, sum over src of W[tgt,src]*S(y1[src,tgt]-y2[src,tgt])
        diff = y1_del - y2_del                  # (src, tgt)
        sigm = jr_sigmoid(diff)                 # (src, tgt)
        expected = np.einsum('ts,st->t', W, sigm)  # (tgt,)

        coup = self._make(W)
        result = coup.compute(delayed_state)

        np.testing.assert_allclose(result[0], expected, rtol=1e-12)
        np.testing.assert_array_equal(result[1], np.zeros(n))

    def test_zero_delay_matches_instant(self):
        # With zero delays, compute() and compute_instant() must agree.
        n = 3
        rng = np.random.default_rng(42)
        W  = np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(W, 0.0)
        y1 = rng.standard_normal(n)
        y2 = rng.standard_normal(n)

        # Build a History with constant state and zero delays.
        history = History(horizon=2, n_cvar=2, n_nodes=n)
        history.initialize(np.stack([y1, y2]))
        history.write(np.stack([y1, y2]))

        delay_steps = np.zeros((n, n), dtype=np.int32)
        delayed_state = history.read_delayed(delay_steps)  # (2, n, n)

        coup = self._make(W)
        result_delayed = coup.compute(delayed_state)
        result_instant = coup.compute_instant(np.stack([y1, y2]))

        np.testing.assert_allclose(result_delayed[0], result_instant[0], rtol=1e-12)

    def test_per_node_sigmoid_params(self):
        # Heterogeneous nu_max / r / v0 per source node.
        n = 3
        rng = np.random.default_rng(5)
        W      = np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(W, 0.0)
        y1 = rng.standard_normal(n)
        y2 = rng.standard_normal(n)

        nu_max = np.array([0.002, 0.003, 0.0025])
        r_arr  = np.array([0.5, 0.6, 0.56])
        v0_arr = np.array([5.0, 5.5, 5.52])

        def S_node(v, i):
            return 2.0 * nu_max[i] / (1.0 + np.exp(r_arr[i] * (v0_arr[i] - v)))

        diff = y1 - y2
        expected = np.array([
            sum(W[tgt, src] * S_node(diff[src], src) for src in range(n))
            for tgt in range(n)
        ])

        coup = JRSigmoidalCoupling(W, nu_max=nu_max, r=r_arr, v0=v0_arr)
        result = coup.compute_instant(np.stack([y1, y2]))

        np.testing.assert_allclose(result[0], expected, rtol=1e-12)


# ── Integration test: full numpy JR simulation ─────────────────────────────────

def _jr_spec(n_nodes, coupling_kind, tract_lengths=None):
    rng = np.random.default_rng(0)
    W = np.abs(rng.standard_normal((n_nodes, n_nodes)))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True).clip(min=1.0)
    if tract_lengths is None:
        tract_lengths = np.zeros((n_nodes, n_nodes))
    return SimulationSpec(
        model=jansen_rit,
        integrator=IntegratorSpec(method="heun", dt=0.05),
        coupling=CouplingSpec(coupling_kind),
        monitors=(MonitorSpec("raw"),),
        weights=W,
        tract_lengths=tract_lengths,
        speed=4.0,
        node_params={"mu": np.full(n_nodes, 0.22), "G": 1.5},
    )


class TestJRSigmoidalSimulation:
    def test_runs_and_is_finite(self):
        spec = _jr_spec(3, "jr_sigmoidal")
        _, data = Simulator(spec, backend="numpy").run(20.0)["raw"]
        assert np.isfinite(data).all()

    def test_output_shape(self):
        spec = _jr_spec(4, "jr_sigmoidal")
        t, data = Simulator(spec, backend="numpy").run(10.0)["raw"]
        assert data.shape[1] == 6   # 6 state variables
        assert data.shape[2] == 4   # 4 nodes

    def test_differs_from_linear_coupling(self):
        # jr_sigmoidal applies a sigmoid per source node before weighting;
        # linear coupling scales the raw state without any nonlinearity.
        # The two must produce different trajectories for nonzero weights.
        spec_jr  = _jr_spec(2, "jr_sigmoidal")
        spec_lin = _jr_spec(2, "linear")
        _, data_jr  = Simulator(spec_jr,  backend="numpy").run(20.0)["raw"]
        _, data_lin = Simulator(spec_lin, backend="numpy").run(20.0)["raw"]
        assert not np.allclose(data_jr, data_lin), (
            "jr_sigmoidal and linear coupling must produce different trajectories"
        )

    def test_with_delays_is_finite(self):
        rng = np.random.default_rng(3)
        n = 3
        D = np.abs(rng.standard_normal((n, n))) * 10.0
        np.fill_diagonal(D, 0.0)
        spec = _jr_spec(n, "jr_sigmoidal", tract_lengths=D)
        _, data = Simulator(spec, backend="numpy").run(20.0)["raw"]
        assert np.isfinite(data).all()
