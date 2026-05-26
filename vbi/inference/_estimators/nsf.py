"""
Neural Spline Flow (NSF) conditional density estimator.

Uses Rational-Quadratic (RQ) spline transforms inside an autoregressive
(MADE) backbone, following Durkan et al. (2019) "Neural Spline Flows".

The RQ-spline implementation is a numpy/autograd translation of the
reference nflows implementation (MIT licence).

Architecture
------------
- Same MADE conditioner as MAFEstimator (masked affine connections,
  optional ActNorm, random permutations, z-scoring).
- Forward transform per dimension: affine → RQ-spline instead of affine-only.
- Tail behaviour: linear (identity) outside ``[-tail_bound, tail_bound]``.
- MADE output per flow per dimension: K widths + K heights + (K-1) inner
  derivatives  →  (3K-1) * param_dim values per sample.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as _np               # plain numpy — used for non-differentiable ops
import autograd.numpy as anp
from autograd import grad
from tqdm.auto import trange

from .maf import MAFEstimator, _cosine_lr

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching nflows defaults)
# ---------------------------------------------------------------------------
_MIN_BIN_W  = 1e-3
_MIN_BIN_H  = 1e-3
_MIN_DERIV  = 1e-3

# Boundary-derivative initialiser constant: softplus(c) + _MIN_DERIV = 1.0
_BOUNDARY_D_CONST = float(_np.log(_np.exp(1.0 - _MIN_DERIV) - 1.0))


# ---------------------------------------------------------------------------
# Pure-function spline helpers
# ---------------------------------------------------------------------------

def _softmax(x):
    """Numerically stable softmax along the last axis (autograd-safe)."""
    shifted = x - anp.max(x, axis=-1, keepdims=True)
    e = anp.exp(shifted)
    return e / anp.sum(e, axis=-1, keepdims=True)


def _softplus(x):
    """Numerically stable softplus (autograd-safe)."""
    return anp.where(x > 20.0, x, anp.log1p(anp.exp(x)))


def _spline_params(unnorm_w, unnorm_h, unnorm_d_inner, B, K):
    """
    Convert raw MADE outputs to normalised spline parameters.

    Parameters
    ----------
    unnorm_w, unnorm_h : (N, K)   raw widths / heights
    unnorm_d_inner     : (N, K-1) raw inner derivatives
    B                  : float    tail bound
    K                  : int      number of bins

    Returns
    -------
    cumwidths  (N, K+1)   left/right edges of bins, from -B to B
    cumheights (N, K+1)   idem for output space
    widths     (N, K)     positive bin widths  (sum 2B per row)
    heights    (N, K)     positive bin heights (sum 2B per row)
    derivatives (N, K+1)  positive derivatives at K+1 knot points
    """
    N = unnorm_w.shape[0]

    # Widths: softmax → scale by 2B, with minimum per bin
    w = _softmax(unnorm_w)                              # (N, K), sums to 1
    w = _MIN_BIN_W + (1.0 - _MIN_BIN_W * K) * w        # min-width floor
    w = w * 2.0 * B                                     # scale to sum 2B

    # Heights: same
    h = _softmax(unnorm_h)
    h = _MIN_BIN_H + (1.0 - _MIN_BIN_H * K) * h
    h = h * 2.0 * B

    # Cumulative widths / heights: shape (N, K+1), starting at -B
    cumw = anp.concatenate(
        [-B * anp.ones((N, 1)), anp.cumsum(w, axis=-1) - B],
        axis=-1,
    )
    cumh = anp.concatenate(
        [-B * anp.ones((N, 1)), anp.cumsum(h, axis=-1) - B],
        axis=-1,
    )
    # Re-derive widths/heights from cumulative to avoid float drift
    widths  = cumw[:, 1:] - cumw[:, :-1]    # (N, K)
    heights = cumh[:, 1:] - cumh[:, :-1]    # (N, K)

    # Derivatives: pad boundary with constant → softplus + min
    pad = _BOUNDARY_D_CONST * anp.ones((N, 1))
    unnorm_d_all = anp.concatenate([pad, unnorm_d_inner, pad], axis=-1)  # (N, K+1)
    derivatives  = _MIN_DERIV + _softplus(unnorm_d_all)                  # (N, K+1)

    return cumw, cumh, widths, heights, derivatives


def _rq_forward_1d(x, unnorm_w, unnorm_h, unnorm_d, B, K):
    """
    Forward RQ-spline for one parameter dimension.

    Parameters
    ----------
    x        : (N,)  values in the parameter space (after z-scoring)
    unnorm_w : (N, K)
    unnorm_h : (N, K)
    unnorm_d : (N, K-1)  inner derivatives

    Returns
    -------
    y       : (N,)  transformed values
    log_det : (N,)  log |dy/dx|
    """
    N = x.shape[0]
    cumw, cumh, widths, heights, derivs = _spline_params(unnorm_w, unnorm_h, unnorm_d, B, K)

    # --- Bin selection (non-differentiable; pure numpy) ----------------------
    inside    = (_np.array(x) >= -B) & (_np.array(x) <= B)
    cumw_np   = _np.array(cumw)          # (N, K+1)
    x_np      = _np.array(x)            # (N,)
    # bin k is the last bin whose left edge <= x
    bin_idx   = _np.sum(cumw_np[:, :-1] <= x_np[:, None], axis=-1) - 1
    bin_idx   = _np.clip(bin_idx, 0, K - 1)   # (N,)  int
    n_arange  = _np.arange(N)

    # --- Gather per-bin values -----------------------------------------------
    cumw_k  = cumw[n_arange, bin_idx]          # (N,)
    w_k     = widths[n_arange, bin_idx]        # (N,)
    cumh_k  = cumh[n_arange, bin_idx]          # (N,)
    h_k     = heights[n_arange, bin_idx]       # (N,)
    d_k     = derivs[n_arange, bin_idx]        # (N,)
    d_k1    = derivs[n_arange, bin_idx + 1]    # (N,)
    s_k     = h_k / (w_k + 1e-10)             # slope = h/w

    # --- Forward RQ transform ------------------------------------------------
    theta     = (x - cumw_k) / (w_k + 1e-10)
    theta     = anp.clip(theta, 0.0, 1.0)
    theta1m   = theta * (1.0 - theta)

    denom     = s_k + (d_k1 + d_k - 2.0 * s_k) * theta1m
    numer     = h_k * (s_k * theta ** 2 + d_k * theta1m)
    y_spline  = cumh_k + numer / (denom + 1e-10)

    dn = s_k ** 2 * (d_k1 * theta ** 2 + 2.0 * s_k * theta1m + d_k * (1.0 - theta) ** 2)
    log_det_spline = anp.log(dn + 1e-10) - 2.0 * anp.log(denom + 1e-10)

    # --- Linear tails --------------------------------------------------------
    y_out       = anp.where(inside, y_spline,       x)
    log_det_out = anp.where(inside, log_det_spline, anp.zeros_like(log_det_spline))

    return y_out, log_det_out


def _rq_inverse_1d(y, unnorm_w, unnorm_h, unnorm_d, B, K):
    """
    Inverse RQ-spline: latent y → parameter x.

    Parameters
    ----------
    y : (N,)  latent values (from the base Gaussian)

    Returns
    -------
    x       : (N,)  parameter values
    log_det : (N,)  log |dy/dx| of the *forward* transform (for density)
    """
    N = y.shape[0]
    cumw, cumh, widths, heights, derivs = _spline_params(unnorm_w, unnorm_h, unnorm_d, B, K)

    # --- Bin selection on heights (inverse direction) ------------------------
    inside    = (_np.array(y) >= -B) & (_np.array(y) <= B)
    cumh_np   = _np.array(cumh)
    y_np      = _np.array(y)
    bin_idx   = _np.sum(cumh_np[:, :-1] <= y_np[:, None], axis=-1) - 1
    bin_idx   = _np.clip(bin_idx, 0, K - 1)
    n_arange  = _np.arange(N)

    # --- Gather ---------------------------------------------------------------
    cumw_k  = cumw[n_arange, bin_idx]
    w_k     = widths[n_arange, bin_idx]
    cumh_k  = cumh[n_arange, bin_idx]
    h_k     = heights[n_arange, bin_idx]
    d_k     = derivs[n_arange, bin_idx]
    d_k1    = derivs[n_arange, bin_idx + 1]
    s_k     = h_k / (w_k + 1e-10)

    # --- Solve quadratic for theta -------------------------------------------
    eta  = y - cumh_k
    a    =  h_k * (s_k - d_k) + eta * (d_k1 + d_k - 2.0 * s_k)
    b    =  h_k * d_k          - eta * (d_k1 + d_k - 2.0 * s_k)
    c    = -s_k * eta

    disc  = anp.maximum(b ** 2 - 4.0 * a * c, 0.0)
    # Numerically stable root (matches nflows)
    root  = (2.0 * c) / (-b - anp.sqrt(disc + 1e-10))
    root  = anp.clip(root, 0.0, 1.0)

    x_spline = root * w_k + cumw_k

    # Log |J| of the forward transform at root
    root1m = root * (1.0 - root)
    denom  = s_k + (d_k1 + d_k - 2.0 * s_k) * root1m
    dn     = s_k ** 2 * (d_k1 * root ** 2 + 2.0 * s_k * root1m + d_k * (1.0 - root) ** 2)
    log_det_spline = anp.log(dn + 1e-10) - 2.0 * anp.log(denom + 1e-10)

    # --- Linear tails --------------------------------------------------------
    x_out       = anp.where(inside, x_spline,       y)
    log_det_out = anp.where(inside, log_det_spline, anp.zeros_like(log_det_spline))

    return x_out, log_det_out


# ---------------------------------------------------------------------------
# NSFEstimator
# ---------------------------------------------------------------------------

@dataclass
class NSFEstimator(MAFEstimator):
    """
    Neural Spline Flow conditional density estimator.

    Inherits all training infrastructure from :class:`MAFEstimator`
    (Adam, mini-batch, validation split, early stopping, cosine LR,
    collapse monitoring).  Only the inner transform differs: RQ-spline
    instead of affine shift-and-scale.

    Parameters
    ----------
    num_bins : int
        Number of spline bins (K).  Higher = more expressive but slower.
        sbi default is 10; 8 is a good starting point.
    tail_bound : float
        Spline is defined on ``[-tail_bound, tail_bound]``; linear tails
        outside.  With z-scored theta (zero-mean, unit-var), ``5.0`` covers
        > 99.99 % of a Gaussian.

    All other parameters are inherited from :class:`MAFEstimator`.
    """

    num_bins:   int   = 8
    tail_bound: float = 5.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _out_per_dim(self) -> int:
        """MADE output values per parameter dimension per flow."""
        K = self.num_bins
        return K + K + (K - 1)   # widths + heights + inner_derivs

    # ------------------------------------------------------------------
    # Weight initialisation  (override: W2 is larger)
    # ------------------------------------------------------------------

    def _initialize_weights(self, rng) -> dict:
        weights = {}
        layers  = []
        D, C, H = self.param_dim, self._ctx_dim(), self.hidden_units
        O = self._out_per_dim() * D          # total MADE output size

        for k in range(self.n_flows):
            m_in     = anp.arange(1, D + 1)
            m_hidden = rng.randint(1, D + 1, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype("f")
            M2 = (m_hidden[None, :] < m_in[:, None]).astype("f")
            Mm = (m_hidden[None, :] <= m_hidden[:, None]).astype("f")
            perm     = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int)
            inv_perm[perm] = anp.arange(D)
            layers.append({"M1": M1, "M2": M2, "Mm": Mm,
                           "perm": perm, "inv_perm": inv_perm})

            w = 0.01
            weights[f"W1y_{k}"] = (rng.randn(H, D) * w).astype("f")
            weights[f"W1c_{k}"] = (rng.randn(H, C) * w).astype("f") if C > 0 else anp.zeros((H, C), "f")
            weights[f"b1_{k}"]  = anp.zeros(H, "f")
            for i in range(self.num_blocks - 1):
                weights[f"Wm_{k}_{i}"] = (rng.randn(H, H) * w).astype("f")
                weights[f"bm_{k}_{i}"] = anp.zeros(H, "f")

            # Output: O values per sample.  Use repeating mask (M2 block tiled).
            # Each of the _out_per_dim() outputs for dim d depends on dims < d.
            # We tile M2 along dim 0: O // D times, shape (O, H).
            reps  = self._out_per_dim()
            M2_O  = anp.tile(M2, (reps, 1))             # (O, H)
            W2c_O = anp.zeros((O, C), "f")
            weights[f"W2_{k}"]  = anp.zeros((O, H), "f")
            weights[f"W2c_{k}"] = W2c_O
            weights[f"b2_{k}"]  = anp.zeros(O, "f")
            # Store expanded mask for masked matmul
            weights[f"_M2O_{k}"] = M2_O.astype("f")

            if self.use_actnorm:
                weights[f"act_s_{k}"] = anp.ones(D,  "f")
                weights[f"act_b_{k}"] = anp.zeros(D, "f")

        self.model_constants = {"layers": layers}
        if self._emb is not None:
            weights.update(self._emb.init_weights(rng))
        return weights

    # ------------------------------------------------------------------
    # MADE forward — outputs spline params instead of (mu, log_sig)
    # ------------------------------------------------------------------

    def _made_nsf_forward(self, y, ctx, lc, k, weights):
        """
        MADE conditioner: maps (y, ctx) → raw spline parameters.

        Returns
        -------
        out : (N, _out_per_dim() * D)
        """
        M1, M2, Mm = lc["M1"], lc["M2"], lc["Mm"]
        W1y  = weights[f"W1y_{k}"]
        W1c  = weights[f"W1c_{k}"]
        b1   = weights[f"b1_{k}"]
        W2   = weights[f"W2_{k}"]
        W2c  = weights[f"W2c_{k}"]
        b2   = weights[f"b2_{k}"]
        M2O  = weights[f"_M2O_{k}"]

        h = self._act(anp.dot(y, (W1y * M1).T) +
                      (anp.dot(ctx, W1c.T) if self._ctx_dim() > 0 else 0.0) + b1)
        for i in range(self.num_blocks - 1):
            h = self._act(anp.dot(h, (weights[f"Wm_{k}_{i}"] * Mm).T) +
                          weights[f"bm_{k}_{i}"])
        out = anp.dot(h, (W2 * M2O).T)
        if self._ctx_dim() > 0:
            out = out + anp.dot(ctx, W2c.T)
        return out + b2   # (N, O)

    # ------------------------------------------------------------------
    # Log-probability (forward pass: theta → latent z)
    # ------------------------------------------------------------------

    def _get_log_prob(self, weights, features, params):
        if self._emb is not None:
            features = self._emb.forward(weights, features)
        x = self._z_x(features).astype("f") if self._ctx_dim() > 0 else features.astype("f")
        u = self._z_theta(params).astype("f")
        log_det = anp.zeros(u.shape[0], "f")

        K, D, B = self.num_bins, self.param_dim, self.tail_bound
        O = self._out_per_dim()

        for k, lc in enumerate(self.model_constants["layers"]):
            u = u[:, lc["perm"]]
            v, ld = self._apply_actnorm(u, k, weights, maybe_init=u)
            if self.use_actnorm:
                log_det = log_det + ld

            # MADE → raw spline params: (N, O*D)
            out = self._made_nsf_forward(v, x, lc, k, weights)

            # Split and reshape to (N, D, K) / (N, D, K-1)
            unnorm_w = out[:, 0:K*D].reshape(-1, D, K)
            unnorm_h = out[:, K*D:2*K*D].reshape(-1, D, K)
            unnorm_d = out[:, 2*K*D:].reshape(-1, D, K - 1)

            # Apply RQ spline per dimension, collect results
            u_parts  = []
            ld_parts = []
            for d in range(D):
                y_d, ld_d = _rq_forward_1d(
                    v[:, d],
                    unnorm_w[:, d, :], unnorm_h[:, d, :], unnorm_d[:, d, :],
                    B, K,
                )
                u_parts.append(y_d)
                ld_parts.append(ld_d)

            u        = anp.stack(u_parts,  axis=1)   # (N, D)
            log_det += anp.sum(anp.stack(ld_parts, axis=1), axis=1)

        base = -0.5 * anp.sum(u ** 2, axis=1) - 0.5 * D * anp.log(2.0 * anp.pi)
        return base + log_det

    def _loss_function(self, weights, features, params) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features, params) -> anp.ndarray:
        from .base import ConditionalDensityEstimator
        ConditionalDensityEstimator.log_prob(self, features, params)
        return self._get_log_prob(self.weights, features, params)

    # ------------------------------------------------------------------
    # Sampling (inverse pass: latent z → theta)
    # ------------------------------------------------------------------

    def sample(self, features, n_samples: int, rng) -> anp.ndarray:
        """Returns shape (n_conditions, n_samples, param_dim)."""
        from .base import ConditionalDensityEstimator
        ConditionalDensityEstimator.sample(self, features, n_samples, rng)

        features = anp.asarray(features, dtype="f")
        if features.ndim == 1 and self.feature_dim > 0:
            features = features.reshape(1, -1)
        if self._emb is not None:
            features = self._emb.forward(self.weights, features)
        n_cond = 1 if self.feature_dim == 0 else features.shape[0]

        x   = self._z_x(features).astype("f") if self._ctx_dim() > 0 else anp.zeros((n_cond, 0), "f")
        out = anp.zeros((n_cond, n_samples, self.param_dim), "f")

        K, D, B = self.num_bins, self.param_dim, self.tail_bound
        O = self._out_per_dim()

        for c in range(n_cond):
            y = rng.randn(n_samples, D).astype("f")

            for k, lc in reversed(list(enumerate(self.model_constants["layers"]))):
                ctx = x[c:c+1].repeat(n_samples, axis=0)   # (n_samples, C)
                v   = anp.zeros_like(y)

                # Autoregressive inverse: dimension by dimension
                for d_idx in range(D):
                    out_raw = self._made_nsf_forward(v, ctx, lc, k, self.weights)
                    # (n_samples, O*D)
                    uw = out_raw[:, 0:K*D].reshape(-1, D, K)
                    uh = out_raw[:, K*D:2*K*D].reshape(-1, D, K)
                    ud = out_raw[:, 2*K*D:].reshape(-1, D, K - 1)

                    x_d, _ = _rq_inverse_1d(
                        y[:, d_idx],
                        uw[:, d_idx, :], uh[:, d_idx, :], ud[:, d_idx, :],
                        B, K,
                    )
                    v = anp.concatenate(
                        [v[:, :d_idx],
                         x_d[:, None],
                         v[:, d_idx + 1:]], axis=1,
                    )

                # Inverse ActNorm
                if self.use_actnorm:
                    s = self.weights[f"act_s_{k}"]
                    b = self.weights[f"act_b_{k}"]
                    v = v / (s + 1e-12) - b

                y = v[:, lc["inv_perm"]]

            out[c] = self._inv_z_theta(y)

        return out
