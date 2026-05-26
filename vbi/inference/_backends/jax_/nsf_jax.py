"""
JAX Neural Spline Flow conditional density estimator.

Translation of nsf.py (autograd) to JAX.

Critical difference from the autograd version: the RQ-spline bin-selection
code used _np.array(x) conversions to run on plain numpy.  In JAX those
conversions would break jit compilation.  The JAX version uses only
jnp operations — bin selection via jnp.sum(cumw[:, :-1] <= x[:, None])
is a fully differentiable-compatible comparison that works inside jit.
"""
from __future__ import annotations

import logging
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

from .maf_jax import JaxMAFEstimator

log = logging.getLogger(__name__)

_MIN_BIN_W = 1e-3
_MIN_BIN_H = 1e-3
_MIN_DERIV = 1e-3
_BOUNDARY_D_CONST = float(np.log(np.exp(1.0 - _MIN_DERIV) - 1.0))


# ---------------------------------------------------------------------------
# Pure JAX spline helpers (no _np.array() conversions)
# ---------------------------------------------------------------------------

def _softmax_jax(x):
    shifted = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(shifted)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def _softplus_jax(x):
    return jnp.where(x > 20.0, x, jnp.log1p(jnp.exp(x)))


def _spline_params_jax(unnorm_w, unnorm_h, unnorm_d_inner, B, K):
    N = unnorm_w.shape[0]

    w = _softmax_jax(unnorm_w)
    w = _MIN_BIN_W + (1.0 - _MIN_BIN_W * K) * w
    w = w * 2.0 * B

    h = _softmax_jax(unnorm_h)
    h = _MIN_BIN_H + (1.0 - _MIN_BIN_H * K) * h
    h = h * 2.0 * B

    cumw = jnp.concatenate(
        [-B * jnp.ones((N, 1)), jnp.cumsum(w, axis=-1) - B], axis=-1)
    cumh = jnp.concatenate(
        [-B * jnp.ones((N, 1)), jnp.cumsum(h, axis=-1) - B], axis=-1)

    widths  = cumw[:, 1:] - cumw[:, :-1]
    heights = cumh[:, 1:] - cumh[:, :-1]

    pad          = _BOUNDARY_D_CONST * jnp.ones((N, 1))
    unnorm_d_all = jnp.concatenate([pad, unnorm_d_inner, pad], axis=-1)
    derivatives  = _MIN_DERIV + _softplus_jax(unnorm_d_all)

    return cumw, cumh, widths, heights, derivatives


def _rq_forward_1d_jax(x, unnorm_w, unnorm_h, unnorm_d, B, K):
    """
    Forward RQ-spline for one dimension — pure JAX, jit-safe.

    Bin selection uses jnp.sum(cumw[:, :-1] <= x[:, None]) instead of
    the _np.array conversion in the autograd version.
    """
    N = x.shape[0]
    cumw, cumh, widths, heights, derivs = _spline_params_jax(
        unnorm_w, unnorm_h, unnorm_d, B, K)

    inside  = (x >= -B) & (x <= B)

    # Bin selection: last bin whose left edge <= x
    bin_idx = jnp.sum(cumw[:, :-1] <= x[:, None], axis=-1) - 1
    bin_idx = jnp.clip(bin_idx, 0, K - 1).astype(jnp.int32)
    n_idx   = jnp.arange(N)

    cumw_k = cumw[n_idx, bin_idx]
    w_k    = widths[n_idx, bin_idx]
    cumh_k = cumh[n_idx, bin_idx]
    h_k    = heights[n_idx, bin_idx]
    d_k    = derivs[n_idx, bin_idx]
    d_k1   = derivs[n_idx, bin_idx + 1]
    s_k    = h_k / (w_k + 1e-10)

    theta    = (x - cumw_k) / (w_k + 1e-10)
    theta    = jnp.clip(theta, 0.0, 1.0)
    theta1m  = theta * (1.0 - theta)

    denom    = s_k + (d_k1 + d_k - 2.0 * s_k) * theta1m
    numer    = h_k * (s_k * theta ** 2 + d_k * theta1m)
    y_spline = cumh_k + numer / (denom + 1e-10)

    dn = s_k ** 2 * (d_k1 * theta ** 2 + 2.0 * s_k * theta1m + d_k * (1.0 - theta) ** 2)
    log_det_spline = jnp.log(dn + 1e-10) - 2.0 * jnp.log(denom + 1e-10)

    y_out       = jnp.where(inside, y_spline,       x)
    log_det_out = jnp.where(inside, log_det_spline, jnp.zeros_like(log_det_spline))
    return y_out, log_det_out


def _rq_inverse_1d_jax(y, unnorm_w, unnorm_h, unnorm_d, B, K):
    """
    Inverse RQ-spline — pure JAX, jit-safe.
    """
    N = y.shape[0]
    cumw, cumh, widths, heights, derivs = _spline_params_jax(
        unnorm_w, unnorm_h, unnorm_d, B, K)

    inside  = (y >= -B) & (y <= B)

    bin_idx = jnp.sum(cumh[:, :-1] <= y[:, None], axis=-1) - 1
    bin_idx = jnp.clip(bin_idx, 0, K - 1).astype(jnp.int32)
    n_idx   = jnp.arange(N)

    cumw_k = cumw[n_idx, bin_idx]
    w_k    = widths[n_idx, bin_idx]
    cumh_k = cumh[n_idx, bin_idx]
    h_k    = heights[n_idx, bin_idx]
    d_k    = derivs[n_idx, bin_idx]
    d_k1   = derivs[n_idx, bin_idx + 1]
    s_k    = h_k / (w_k + 1e-10)

    eta  = y - cumh_k
    a    =  h_k * (s_k - d_k) + eta * (d_k1 + d_k - 2.0 * s_k)
    b    =  h_k * d_k          - eta * (d_k1 + d_k - 2.0 * s_k)
    c    = -s_k * eta

    disc     = jnp.maximum(b ** 2 - 4.0 * a * c, 0.0)
    root     = (2.0 * c) / (-b - jnp.sqrt(disc + 1e-10))
    root     = jnp.clip(root, 0.0, 1.0)

    x_spline = root * w_k + cumw_k

    root1m = root * (1.0 - root)
    denom  = s_k + (d_k1 + d_k - 2.0 * s_k) * root1m
    dn     = s_k ** 2 * (d_k1 * root ** 2 + 2.0 * s_k * root1m + d_k * (1.0 - root) ** 2)
    log_det_spline = jnp.log(dn + 1e-10) - 2.0 * jnp.log(denom + 1e-10)

    x_out       = jnp.where(inside, x_spline,       y)
    log_det_out = jnp.where(inside, log_det_spline, jnp.zeros_like(log_det_spline))
    return x_out, log_det_out


# ---------------------------------------------------------------------------
# NSFEstimator (JAX)
# ---------------------------------------------------------------------------

@dataclass
class JaxNSFEstimator(JaxMAFEstimator):
    """
    Neural Spline Flow — JAX backend.

    Inherits all training infrastructure from JaxMAFEstimator.
    Only the inner transform differs: RQ-spline instead of affine.
    """

    num_bins:   int   = 8
    tail_bound: float = 5.0

    def _out_per_dim(self) -> int:
        K = self.num_bins
        return K + K + (K - 1)

    # ------------------------------------------------------------------
    # Weight initialisation (W2 is larger for spline output)
    # ------------------------------------------------------------------

    def _initialize_weights(self, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        weights = {}
        layers  = []
        D, C, H = self.param_dim, self._ctx_dim(), self.hidden_units
        O = self._out_per_dim() * D

        for k in range(self.n_flows):
            m_in     = np.arange(1, D + 1)
            m_hidden = rng.randint(1, D + 1, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype("f")
            M2 = (m_hidden[None, :] < m_in[:, None]).astype("f")
            Mm = (m_hidden[None, :] <= m_hidden[:, None]).astype("f")
            perm     = rng.permutation(D)
            inv_perm = np.empty(D, dtype=int)
            inv_perm[perm] = np.arange(D)
            layers.append({
                "M1": jnp.array(M1), "M2": jnp.array(M2), "Mm": jnp.array(Mm),
                "perm": jnp.array(perm), "inv_perm": jnp.array(inv_perm),
            })

            w = 0.01
            weights[f"W1y_{k}"] = (rng.randn(H, D) * w).astype("f")
            weights[f"W1c_{k}"] = (rng.randn(H, C) * w).astype("f") if C > 0 else np.zeros((H, C), "f")
            weights[f"b1_{k}"]  = np.zeros(H, "f")
            for i in range(self.num_blocks - 1):
                weights[f"Wm_{k}_{i}"] = (rng.randn(H, H) * w).astype("f")
                weights[f"bm_{k}_{i}"] = np.zeros(H, "f")

            reps  = self._out_per_dim()
            M2_O  = np.tile(np.array(M2), (reps, 1))
            weights[f"W2_{k}"]   = np.zeros((O, H), "f")
            weights[f"W2c_{k}"]  = np.zeros((O, C), "f")
            weights[f"b2_{k}"]   = np.zeros(O, "f")
            weights[f"_M2O_{k}"] = M2_O.astype("f")

            if self.use_actnorm:
                weights[f"act_s_{k}"] = np.ones(D,  "f")
                weights[f"act_b_{k}"] = np.zeros(D, "f")

        self.model_constants = {"layers": layers}
        if self._emb is not None:
            weights.update(self._emb.init_weights(np.random.RandomState(seed)))
        return weights

    # ------------------------------------------------------------------
    # MADE forward — returns raw spline parameters
    # ------------------------------------------------------------------

    def _made_nsf_forward(self, y, ctx, lc, k, weights):
        M1, M2, Mm = lc["M1"], lc["M2"], lc["Mm"]
        W1y  = weights[f"W1y_{k}"]
        W1c  = weights[f"W1c_{k}"]
        b1   = weights[f"b1_{k}"]
        W2   = weights[f"W2_{k}"]
        W2c  = weights[f"W2c_{k}"]
        b2   = weights[f"b2_{k}"]
        M2O  = weights[f"_M2O_{k}"]

        h = self._act(jnp.dot(y, (W1y * M1).T) +
                      (jnp.dot(ctx, W1c.T) if self._ctx_dim() > 0 else 0.0) + b1)
        for i in range(self.num_blocks - 1):
            h = self._act(jnp.dot(h, (weights[f"Wm_{k}_{i}"] * Mm).T) +
                          weights[f"bm_{k}_{i}"])
        out = jnp.dot(h, (W2 * M2O).T)
        if self._ctx_dim() > 0:
            out = out + jnp.dot(ctx, W2c.T)
        return out + b2

    def _forward_one_step(self, v, x, lc, k, w):
        """Propagate v through one NSF step (spline) for ActNorm data-dep init."""
        K, D, B = self.num_bins, self.param_dim, self.tail_bound
        out      = self._made_nsf_forward(v, x, lc, k, w)
        unnorm_w = out[:, 0:K*D].reshape(-1, D, K)
        unnorm_h = out[:, K*D:2*K*D].reshape(-1, D, K)
        unnorm_d = out[:, 2*K*D:].reshape(-1, D, K - 1)
        u_parts  = []
        for d in range(D):
            y_d, _ = _rq_forward_1d_jax(
                v[:, d], unnorm_w[:, d, :], unnorm_h[:, d, :], unnorm_d[:, d, :], B, K)
            u_parts.append(y_d)
        return jnp.stack(u_parts, axis=1)

    # ------------------------------------------------------------------
    # Log-probability (forward: theta → z)
    # ------------------------------------------------------------------

    def _get_log_prob(self, weights, features, params):
        if self._emb is not None:
            features = self._emb.forward(weights, features)
        x = self._z_x(features).astype("f") if self._ctx_dim() > 0 else features.astype("f")
        u = self._z_theta(params).astype("f")
        log_det = jnp.zeros(u.shape[0], "f")

        K, D, B = self.num_bins, self.param_dim, self.tail_bound
        O = self._out_per_dim()

        for k, lc in enumerate(self.model_constants["layers"]):
            u = u[:, lc["perm"]]
            v, ld = self._apply_actnorm(u, k, weights)
            if self.use_actnorm:
                log_det = log_det + ld

            out      = self._made_nsf_forward(v, x, lc, k, weights)
            unnorm_w = out[:, 0:K*D].reshape(-1, D, K)
            unnorm_h = out[:, K*D:2*K*D].reshape(-1, D, K)
            unnorm_d = out[:, 2*K*D:].reshape(-1, D, K - 1)

            u_parts  = []
            ld_parts = []
            for d in range(D):
                y_d, ld_d = _rq_forward_1d_jax(
                    v[:, d],
                    unnorm_w[:, d, :], unnorm_h[:, d, :], unnorm_d[:, d, :],
                    B, K,
                )
                u_parts.append(y_d)
                ld_parts.append(ld_d)

            u        = jnp.stack(u_parts,  axis=1)
            log_det += jnp.sum(jnp.stack(ld_parts, axis=1), axis=1)

        base = -0.5 * jnp.sum(u ** 2, axis=1) - 0.5 * D * jnp.log(2.0 * jnp.pi)
        return base + log_det

    def _loss_function(self, weights, features, params) -> float:
        return -jnp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features, params):
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        w = {k: jnp.array(v) for k, v in self.weights.items()}
        f = jnp.array(features, dtype="f")
        p = jnp.array(params,   dtype="f")
        return np.array(self._get_log_prob(w, f, p))

    # ------------------------------------------------------------------
    # Sampling (inverse: z → theta)
    # ------------------------------------------------------------------

    def sample(self, features, n_samples: int, rng):
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        key = self._to_jax_key(rng)

        f = jnp.array(features, dtype="f")
        if f.ndim == 1 and self.feature_dim > 0:
            f = f.reshape(1, -1)
        if self._emb is not None:
            w = {k: jnp.array(v) for k, v in self.weights.items()}
            f = self._emb.forward(w, f)
        n_cond = 1 if self.feature_dim == 0 else f.shape[0]

        x   = self._z_x(f).astype("f") if self._ctx_dim() > 0 else jnp.zeros((n_cond, 0), "f")
        out = np.zeros((n_cond, n_samples, self.param_dim), "f")
        w   = {k: jnp.array(v) for k, v in self.weights.items()}

        K, D, B = self.num_bins, self.param_dim, self.tail_bound
        O = self._out_per_dim()

        for c in range(n_cond):
            key, subkey = jax.random.split(key)
            y = jax.random.normal(subkey, (n_samples, D)).astype("f")

            for k_idx, lc in reversed(list(enumerate(self.model_constants["layers"]))):
                ctx = jnp.tile(x[c:c+1], (n_samples, 1))
                v_  = jnp.zeros_like(y)

                for d_idx in range(D):
                    out_raw  = self._made_nsf_forward(v_, ctx, lc, k_idx, w)
                    uw = out_raw[:, 0:K*D].reshape(-1, D, K)
                    uh = out_raw[:, K*D:2*K*D].reshape(-1, D, K)
                    ud = out_raw[:, 2*K*D:].reshape(-1, D, K - 1)

                    x_d, _ = _rq_inverse_1d_jax(
                        y[:, d_idx],
                        uw[:, d_idx, :], uh[:, d_idx, :], ud[:, d_idx, :],
                        B, K,
                    )
                    v_ = v_.at[:, d_idx].set(x_d)

                if self.use_actnorm:
                    s = w[f"act_s_{k_idx}"]
                    b = w[f"act_b_{k_idx}"]
                    v_ = v_ / (s + 1e-12) - b

                y = v_[:, lc["inv_perm"]]

            out[c] = np.array(self._inv_z_theta(y))

        return out
