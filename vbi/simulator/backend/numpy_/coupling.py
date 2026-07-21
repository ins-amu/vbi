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

    def compute(self, delayed_state: np.ndarray,
                current_state: np.ndarray | None = None) -> np.ndarray:
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
        """Fast path for zero-delay (ODE) case - skips the ring buffer.

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

    def compute(self, delayed_state: np.ndarray,
                current_state: np.ndarray | None = None) -> np.ndarray:
        sigm = 1.0 / (1.0 + np.exp(-(delayed_state - self.midpoint) / self.sigma))
        result = np.einsum('ts,cst->ct', self.weights, sigm)
        return self.G * self.a * result + self.b

    def compute_instant(self, cvar_state: np.ndarray) -> np.ndarray:
        """Fast path for zero-delay (ODE) case."""
        sigm = 1.0 / (1.0 + np.exp(-(cvar_state - self.midpoint) / self.sigma))
        result = (self.weights @ sigm.T).T
        return self.G * self.a * result + self.b



class JRSigmoidalCoupling:
    """
    c[0, tgt] = sum_src W[tgt, src] * S(y1[src] - y2[src])
    c[1, tgt] = 0

    where S(v) = 2*nu_max / (1 + exp(r*(v0 - v)))  is the JR sigmoid.

    Sigmoid is applied per source node before structural weighting, giving the
    TVB/JR-style firing-rate drive:  G * W @ S(y1 - y2).
    G is NOT applied here; it is multiplied in the model dfun (c_y1 is already
    the weighted incoming firing-rate drive).
    Requires exactly 2 coupling variables (y1, y2).
    """

    def __init__(self, weights: np.ndarray,
                 nu_max: float | np.ndarray,
                 r: float | np.ndarray,
                 v0: float | np.ndarray):
        self.weights = weights
        self.nu_max = np.asarray(nu_max, dtype=np.float64)
        self.r = np.asarray(r, dtype=np.float64)
        self.v0 = np.asarray(v0, dtype=np.float64)

    def _sigmoid(self, v: np.ndarray) -> np.ndarray:
        return 2.0 * self.nu_max / (1.0 + np.exp(self.r * (self.v0 - v)))

    def compute_instant(self, cvar_state: np.ndarray) -> np.ndarray:
        diff = cvar_state[0] - cvar_state[1]           # (n_nodes,)
        sigm = self._sigmoid(diff)                      # (n_nodes,)
        result = self.weights @ sigm                    # (n_nodes,)
        return np.stack([result, np.zeros_like(result)])

    def compute(self, delayed_state: np.ndarray,
                current_state: np.ndarray | None = None) -> np.ndarray:
        # delayed_state: (2, n_nodes, n_nodes)  [cvar, src, tgt]
        diff = delayed_state[0] - delayed_state[1]      # (n_src, n_tgt)
        nu_max = self.nu_max
        r = self.r
        v0 = self.v0
        if nu_max.ndim == 1:
            nu_max = nu_max[:, np.newaxis]
            r = r[:, np.newaxis]
            v0 = v0[:, np.newaxis]
        sigm = 2.0 * nu_max / (1.0 + np.exp(r * (v0 - diff)))   # (n_src, n_tgt)
        result = np.einsum('ts,st->t', self.weights, sigm)        # (n_tgt,)
        return np.stack([result, np.zeros_like(result)])


class KuramotoCoupling:
    """
    c[tgt] = (G/N) Σ_src W[tgt,src] sin(θ_src(t-τ) − θ_tgt(t) + alpha)

    alpha=0 gives the standard Kuramoto coupling; alpha≠0 adds frustration.
    """

    def __init__(self, weights: np.ndarray, G: float, alpha: float = 0.0):
        self.weights = weights
        self.G = G
        self.N = weights.shape[0]
        self.alpha = alpha

    def compute(self, delayed_state: np.ndarray,
                current_state: np.ndarray) -> np.ndarray:
        theta_src = delayed_state[0]
        theta_tgt = current_state[0]
        diff = theta_src - theta_tgt[np.newaxis, :] + self.alpha
        c = (self.G / self.N) * np.einsum('ts,st->t', self.weights, np.sin(diff))
        return c[np.newaxis, :]

    def compute_instant(self, cvar_state: np.ndarray) -> np.ndarray:
        theta = cvar_state[0]
        diff = theta[:, np.newaxis] - theta[np.newaxis, :] + self.alpha
        c = (self.G / self.N) * np.einsum('ts,st->t', self.weights, np.sin(diff))
        return c[np.newaxis, :]


def build_coupling(
    spec: CouplingSpec,
    weights: np.ndarray,
    G: float,
    model_params: dict | None = None,
) -> LinearCoupling | SigmoidalCoupling | KuramotoCoupling | JRSigmoidalCoupling:
    if spec.kind == "linear":
        return LinearCoupling(spec, weights, G)
    if spec.kind == "sigmoidal":
        return SigmoidalCoupling(spec, weights, G)
    if spec.kind == "kuramoto":
        return KuramotoCoupling(weights, G, alpha=spec.alpha)
    if spec.kind == "jr_sigmoidal":
        p = model_params or {}
        return JRSigmoidalCoupling(
            weights,
            nu_max=p.get("nu_max", 0.0025),
            r=p.get("r", 0.56),
            v0=p.get("v0", 5.52),
        )
    raise ValueError(f"Unknown coupling kind: {spec.kind!r}")
