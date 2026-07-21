"""JAX Mixture Density Network conditional density estimator."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .base_jax import JaxConditionalDensityEstimator


@dataclass
class JaxMDNEstimator(JaxConditionalDensityEstimator):
    """
    Mixture Density Network - JAX backend.

    Identical interface to ``MDNEstimator`` (autograd version).
    """

    param_dim:    int | None       = None
    feature_dim:  int | None       = None
    n_components: int              = 5
    hidden_sizes: tuple[int, ...] = (32, 32)

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)

    def _infer_dimensions(self, params, features):
        super()._infer_dimensions(params, features)
        self._offdiag_basis = self._create_offdiag_basis()

    def _create_offdiag_basis(self):
        n = self.param_dim * (self.param_dim - 1) // 2
        if n == 0:
            return None
        basis = np.zeros((n, self.param_dim, self.param_dim), dtype="f")
        rows, cols = np.triu_indices(self.param_dim, k=1)
        basis[np.arange(n), rows, cols] = 1
        return jnp.array(basis)

    def _initialize_weights(self, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        weights = {}
        in_size = self.feature_dim
        for i, out_size in enumerate(self.hidden_sizes):
            weights[f"W{i}"] = (rng.randn(in_size, out_size) * np.sqrt(2.0 / in_size)).astype("f")
            weights[f"b{i}"] = np.zeros(out_size, dtype="f")
            in_size = out_size

        last = self.hidden_sizes[-1] if self.hidden_sizes else self.feature_dim
        K, D = self.n_components, self.param_dim

        weights["W_alpha"]  = (rng.randn(last, K)      * 0.01).astype("f")
        weights["b_alpha"]  = np.zeros(K, dtype="f")
        weights["W_mu"]     = (rng.randn(last, K * D)  * 0.01).astype("f")
        weights["b_mu"]     = np.zeros(K * D, dtype="f")
        weights["W_L_diag"] = (rng.randn(last, K * D)  * 0.01).astype("f")
        weights["b_L_diag"] = np.zeros(K * D, dtype="f")

        n_off = D * (D - 1) // 2
        if n_off > 0:
            weights["W_L_offdiag"] = (rng.randn(last, K * n_off) * 0.01).astype("f")
            weights["b_L_offdiag"] = np.zeros(K * n_off, dtype="f")
        return weights

    def _forward_pass(self, weights, features):
        h = features
        for i in range(len(self.hidden_sizes)):
            h = jnp.tanh(h @ weights[f"W{i}"] + weights[f"b{i}"])

        K, D = self.n_components, self.param_dim
        log_alpha = h @ weights["W_alpha"] + weights["b_alpha"]
        alpha     = jnp.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        mu        = (h @ weights["W_mu"] + weights["b_mu"]).reshape(-1, K, D)

        L_log_diag = (h @ weights["W_L_diag"] + weights["b_L_diag"]).reshape(-1, K, D)
        L_diag     = jnp.einsum("nki,ij->nkij", jnp.exp(L_log_diag), jnp.eye(D, dtype="f"))

        n_off = D * (D - 1) // 2
        if n_off > 0:
            L_off_vals = (h @ weights["W_L_offdiag"] + weights["b_L_offdiag"]).reshape(-1, K, n_off)
            L_off      = jnp.einsum("nkl,lij->nkij", L_off_vals, self._offdiag_basis)
            L_prec     = L_diag + L_off
        else:
            L_prec = L_diag

        return alpha, mu, L_prec, L_log_diag

    def _get_log_prob(self, weights, features, params):
        """Per-sample log p(params | features), shape (B,). Used by APT loss."""
        return self._log_prob_core(weights, features, params)

    def _loss_function(self, weights, features, params) -> float:
        return -jnp.mean(self._get_log_prob(weights, features, params))

    def _log_prob_core(self, weights, features, params):
        """log_prob without re-embedding (features already embedded)."""
        alpha, mu, L_prec, L_log_diag = self._forward_pass(weights, features)
        delta      = params[:, jnp.newaxis, :] - mu
        z          = jnp.einsum("nkij,nkj->nki", L_prec, delta)
        quad       = -0.5 * jnp.sum(z ** 2, axis=2)
        log_det    = jnp.sum(L_log_diag, axis=2)
        log_prob_k = quad + log_det - 0.5 * self.param_dim * jnp.log(2 * math.pi)
        return logsumexp(jnp.log(alpha + 1e-9) + log_prob_k, axis=1)

    def log_prob(self, features, params):
        super().log_prob(features, params)
        w = {k: jnp.array(v) for k, v in self.weights.items()}
        f = jnp.array(features, dtype="f")
        p = jnp.array(params,   dtype="f")
        if self._emb is not None:
            f = self._emb.forward(w, f)
        return np.array(self._log_prob_core(w, f, p))

    def sample(self, features, n_samples: int, rng,
               log_prob_threshold=None, oversample_factor: int = 5):
        """
        Sample from p(params | features).

        Returns ndarray shape (n_conditions, n_samples, param_dim).
        """
        super().sample(features, n_samples, rng)
        key = self._to_jax_key(rng)

        w = {k: jnp.array(v) for k, v in self.weights.items()}
        f = jnp.array(features, dtype="f")
        if f.ndim == 1:
            f = f.reshape(1, -1)
        if self._emb is not None:
            f = self._emb.forward(w, f)
        n_cond = f.shape[0]
        n_cand = n_samples * oversample_factor

        alpha, mu, L_prec, _ = self._forward_pass(w, f)
        K = self.n_components

        log_alpha = jnp.log(alpha + 1e-9)
        key, subkey = jax.random.split(key)
        gumbel   = -jnp.log(-jnp.log(
            jax.random.uniform(subkey, (n_cond, n_cand, K), minval=1e-20)))
        comp_idx = jnp.argmax(log_alpha[:, jnp.newaxis, :] + gumbel, axis=2)

        ci          = jnp.arange(n_cond)[:, jnp.newaxis]
        chosen_mu   = mu[ci, comp_idx]                 # (n_cond, n_cand, D)
        chosen_Lp   = L_prec[ci, comp_idx]             # (n_cond, n_cand, D, D)

        try:
            L_cov = jnp.linalg.inv(chosen_Lp)
        except Exception:
            return np.full((n_cond, n_samples, self.param_dim), np.nan)

        key, subkey = jax.random.split(key)
        z       = jax.random.normal(subkey, (n_cond, n_cand, self.param_dim))
        samples = chosen_mu + jnp.einsum("ncsi,ncs->nci", L_cov, z)

        if log_prob_threshold is not None:
            flat_f = jnp.tile(f[:, jnp.newaxis, :], (1, n_cand, 1)).reshape(-1, self.feature_dim)
            flat_s = samples.reshape(-1, self.param_dim)
            lp     = self._log_prob_core(w, flat_f, flat_s)
            out    = []
            for i in range(n_cond):
                mask  = np.array(lp[i * n_cand:(i + 1) * n_cand]) > log_prob_threshold
                valid = np.array(flat_s[i * n_cand:(i + 1) * n_cand])[mask]
                if len(valid) == 0:
                    valid = np.tile(np.array(mu[i, 0]), (n_samples, 1))
                elif len(valid) < n_samples:
                    valid = np.concatenate([valid, np.tile(valid[-1:], (n_samples - len(valid), 1))])
                else:
                    valid = valid[:n_samples]
                out.append(valid)
            return np.stack(out)

        return np.array(samples[:, :n_samples, :])
