from __future__ import annotations
import numpy as np
from vbi.simulator.spec.coupling import CouplingSpec


class LinearCoupling:
    """
    c[cvar, tgt] = G * a * sum_src(weights[tgt, src] * x_delayed[cvar, src, tgt]) + b

    Parameters
    ----------
    spec : CouplingSpec
    weights : (n_nodes, n_nodes)   weights[tgt, src]
    G : float   global coupling strength (from model parameters)
    """

    def __init__(self, spec: CouplingSpec, weights: np.ndarray, G: float):
        self.a = spec.a
        self.b = spec.b
        self.weights = weights
        self.G = G

    def compute(self, delayed_state: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        delayed_state : (n_cvar, n_nodes, n_nodes)
            delayed_state[cvar, src, tgt] from History.read_delayed().

        Returns
        -------
        (n_cvar, n_nodes)   coupling input per (cvar, target node)
        """
        result = np.einsum('ts,cst->ct', self.weights, delayed_state)
        return self.G * self.a * result + self.b

    def compute_instant(self, cvar_state: np.ndarray) -> np.ndarray:
        """Fast path for zero-delay (ODE) case — skips the ring buffer.

        Parameters
        ----------
        cvar_state : (n_cvar, n_nodes)   current coupling-variable state.
        """
        # weights @ cvar_state.T → (n_nodes, n_cvar) → transpose → (n_cvar, n_nodes)
        result = (self.weights @ cvar_state.T).T
        return self.G * self.a * result + self.b


class SigmoidalCoupling:
    """
    c[cvar, tgt] = G * a * sum_src(w[tgt,src] * sigm(x_delayed[cvar,src,tgt])) + b
    sigm(x) = 1 / (1 + exp(-(x - midpoint) / sigma))
    """

    def __init__(self, spec: CouplingSpec, weights: np.ndarray, G: float):
        self.a = spec.a
        self.b = spec.b
        self.midpoint = spec.midpoint
        self.sigma = spec.sigma
        self.weights = weights
        self.G = G

    def compute(self, delayed_state: np.ndarray) -> np.ndarray:
        sigm = 1.0 / (1.0 + np.exp(-(delayed_state - self.midpoint) / self.sigma))
        result = np.einsum('ts,cst->ct', self.weights, sigm)
        return self.G * self.a * result + self.b

    def compute_instant(self, cvar_state: np.ndarray) -> np.ndarray:
        """Fast path for zero-delay (ODE) case."""
        sigm = 1.0 / (1.0 + np.exp(-(cvar_state - self.midpoint) / self.sigma))
        result = (self.weights @ sigm.T).T
        return self.G * self.a * result + self.b


def build_coupling(spec: CouplingSpec, weights: np.ndarray,
                   G: float) -> LinearCoupling | SigmoidalCoupling:
    if spec.kind == "linear":
        return LinearCoupling(spec, weights, G)
    if spec.kind == "sigmoidal":
        return SigmoidalCoupling(spec, weights, G)
    raise ValueError(f"Unknown coupling kind: {spec.kind!r}")
