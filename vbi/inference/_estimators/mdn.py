"""Mixture Density Network conditional density estimator."""
import math
from dataclasses import dataclass

import autograd.numpy as anp
from autograd.scipy.special import logsumexp

from .base import ConditionalDensityEstimator


@dataclass
class MDNEstimator(ConditionalDensityEstimator):
    """
    Mixture Density Network for conditional density estimation.

    Models p(params | features) as a Gaussian mixture whose parameters
    are predicted by an MLP conditioned on features.

    Parameters
    ----------
    param_dim : int | None
    feature_dim : int | None
    n_components : int
        Number of Gaussian mixture components.
    hidden_sizes : tuple[int, ...]
        Hidden-layer widths of the MLP backbone.
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
        basis = anp.zeros((n, self.param_dim, self.param_dim), dtype="f")
        rows, cols = anp.triu_indices(self.param_dim, k=1)
        basis[anp.arange(n), rows, cols] = 1
        return basis

    def _initialize_weights(self, rng) -> dict:
        weights = {}
        in_size = self.feature_dim
        for i, out_size in enumerate(self.hidden_sizes):
            weights[f"W{i}"] = (rng.randn(in_size, out_size) * anp.sqrt(2.0 / in_size)).astype("f")
            weights[f"b{i}"] = anp.zeros(out_size, dtype="f")
            in_size = out_size

        last = self.hidden_sizes[-1] if self.hidden_sizes else self.feature_dim
        K, D = self.n_components, self.param_dim

        weights["W_alpha"]         = (rng.randn(last, K)         * 0.01).astype("f")
        weights["b_alpha"]         = anp.zeros(K, dtype="f")
        weights["W_mu"]            = (rng.randn(last, K * D)      * 0.01).astype("f")
        weights["b_mu"]            = anp.zeros(K * D, dtype="f")
        weights["W_L_diag"]        = (rng.randn(last, K * D)      * 0.01).astype("f")
        weights["b_L_diag"]        = anp.zeros(K * D, dtype="f")

        n_off = D * (D - 1) // 2
        if n_off > 0:
            weights["W_L_offdiag"] = (rng.randn(last, K * n_off) * 0.01).astype("f")
            weights["b_L_offdiag"] = anp.zeros(K * n_off, dtype="f")
        return weights

    def _forward_pass(self, weights, features):
        h = features
        for i in range(len(self.hidden_sizes)):
            h = anp.tanh(h @ weights[f"W{i}"] + weights[f"b{i}"])

        K, D = self.n_components, self.param_dim
        log_alpha = h @ weights["W_alpha"] + weights["b_alpha"]
        alpha     = anp.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        mu        = (h @ weights["W_mu"] + weights["b_mu"]).reshape(-1, K, D)

        L_log_diag = (h @ weights["W_L_diag"] + weights["b_L_diag"]).reshape(-1, K, D)
        L_diag     = anp.einsum("nki,ij->nkij", anp.exp(L_log_diag), anp.eye(D, dtype="f"))

        n_off = D * (D - 1) // 2
        if n_off > 0:
            L_off_vals = (h @ weights["W_L_offdiag"] + weights["b_L_offdiag"]).reshape(-1, K, n_off)
            L_off      = anp.einsum("nkl,lij->nkij", L_off_vals, self._offdiag_basis)
            L_prec     = L_diag + L_off
        else:
            L_prec = L_diag

        return alpha, mu, L_prec, L_log_diag

    def _get_log_prob(self, weights, features, params):
        """Per-sample log p(params | features), shape (B,). Used by APT loss."""
        alpha, mu, L_prec, L_log_diag = self._forward_pass(weights, features)
        delta      = params[:, anp.newaxis, :] - mu
        z          = anp.einsum("nkij,nkj->nki", L_prec, delta)
        quad       = -0.5 * anp.sum(z ** 2, axis=2)
        log_det    = anp.sum(L_log_diag, axis=2)
        log_prob_k = quad + log_det - 0.5 * self.param_dim * anp.log(2 * math.pi)
        return logsumexp(anp.log(alpha + 1e-9) + log_prob_k, axis=1)

    def _loss_function(self, weights, features, params) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def _log_prob_preembedded(self, features, params) -> anp.ndarray:
        """Compute log_prob assuming features are already embedded (no re-embedding)."""
        alpha, mu, L_prec, L_log_diag = self._forward_pass(self.weights, features)
        delta      = params[:, anp.newaxis, :] - mu
        z          = anp.einsum("nkij,nkj->nki", L_prec, delta)
        quad       = -0.5 * anp.sum(z ** 2, axis=2)
        log_det    = anp.sum(L_log_diag, axis=2)
        log_prob_k = quad + log_det - 0.5 * self.param_dim * anp.log(2 * math.pi)
        return logsumexp(anp.log(alpha + 1e-9) + log_prob_k, axis=1)

    def log_prob(self, features, params) -> anp.ndarray:
        super().log_prob(features, params)
        if self._emb is not None:
            features = self._emb.forward(self.weights, anp.asarray(features, dtype="f"))
        alpha, mu, L_prec, L_log_diag = self._forward_pass(self.weights, features)
        delta      = params[:, anp.newaxis, :] - mu
        z          = anp.einsum("nkij,nkj->nki", L_prec, delta)
        quad       = -0.5 * anp.sum(z ** 2, axis=2)
        log_det    = anp.sum(L_log_diag, axis=2)
        log_prob_k = quad + log_det - 0.5 * self.param_dim * anp.log(2 * math.pi)
        return logsumexp(anp.log(alpha + 1e-9) + log_prob_k, axis=1)

    def sample(self, features, n_samples: int, rng,
               log_prob_threshold=None, oversample_factor: int = 5) -> anp.ndarray:
        """
        Sample from p(params | features).

        Returns
        -------
        ndarray  shape (n_conditions, n_samples, param_dim)
        """
        super().sample(features, n_samples, rng)
        features = anp.asarray(features, dtype="f")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if self._emb is not None:
            features = self._emb.forward(self.weights, features)
        n_cond = features.shape[0]
        n_cand = n_samples * oversample_factor

        alpha, mu, L_prec, _ = self._forward_pass(self.weights, features)
        K = self.n_components

        log_alpha   = anp.log(alpha + 1e-9)
        gumbel      = -anp.log(-anp.log(rng.uniform(size=(n_cond, n_cand, K))))
        comp_idx    = anp.argmax(log_alpha[:, anp.newaxis, :] + gumbel, axis=2)

        ci          = anp.arange(n_cond)[:, anp.newaxis]
        chosen_mu   = mu[ci, comp_idx]
        chosen_Lp   = L_prec[ci, comp_idx]

        try:
            L_cov = anp.linalg.inv(chosen_Lp)
        except anp.linalg.LinAlgError:
            return anp.full((n_cond, n_samples, self.param_dim), anp.nan)

        z        = rng.randn(n_cond, n_cand, self.param_dim)
        samples  = chosen_mu + anp.einsum("ncsi,ncs->nci", L_cov, z)

        if log_prob_threshold is not None:
            flat_f  = anp.tile(features[:, anp.newaxis, :], (1, n_cand, 1)).reshape(-1, self.feature_dim)
            flat_s  = samples.reshape(-1, self.param_dim)
            # features already embedded above; skip re-embedding
            lp      = self._log_prob_preembedded(flat_f, flat_s)
            out     = []
            for i in range(n_cond):
                mask  = lp[i * n_cand:(i + 1) * n_cand] > log_prob_threshold
                valid = flat_s[i * n_cand:(i + 1) * n_cand][mask]
                if len(valid) == 0:
                    valid = anp.tile(mu[i, 0], (n_samples, 1))
                elif len(valid) < n_samples:
                    valid = anp.concatenate([valid, anp.tile(valid[-1:], (n_samples - len(valid), 1))])
                else:
                    valid = valid[:n_samples]
                out.append(valid)
            return anp.stack(out)

        return samples[:, :n_samples, :]
