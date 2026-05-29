"""
APT (Atomic Posterior Transform) loss for SNPE-C.

Used by SNPE when training with a non-prior proposal distribution
(multi-round sequential inference).  When num_atoms > 1 and the caller
supplies per-sample log-importance weights, the loss becomes:

    L_APT = E_i [ logsumexp_j(log_q(theta_j|x_i) + log_w_j)
                  - log_q(theta_i|x_i) - log_w_i ]

where the K atoms for sample i are drawn from the current mini-batch
(including i itself at position 0) and log_w_j = log_prior(theta_j) -
log_proposal(theta_j).

When all log_w are zero (round-1 / prior proposal) this reduces to a
contrastive normaliser estimate of the standard NLL, which converges to
the NLL as K → batch_size.

Reference: Papamakarios et al. "Normalizing Flows for Probabilistic
Modeling and Inference", 2021.  SNPE-C / APT variant.
"""
from __future__ import annotations

import numpy as np

try:
    import autograd.numpy as anp
    from autograd.scipy.special import logsumexp as _ag_logsumexp
    _HAS_AUTOGRAD = True
except ImportError:
    _HAS_AUTOGRAD = False

try:
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as _jax_logsumexp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _logsumexp_backend(arr, axis, backend: str):
    """Backend-aware logsumexp that stays in the computation graph."""
    if backend == "jax":
        return _jax_logsumexp(arr, axis=axis)
    return _ag_logsumexp(arr, axis=axis)


def make_apt_loss(estimator, num_atoms: int, seed: int = 0):
    """
    Build an APT loss function (same signature as ``_loss_function``).

    The returned callable has the signature::

        apt_loss(weights, features, params_aug) -> scalar

    where ``params_aug`` is ``params`` with ``log_importance_weight``
    appended as an extra last column.  The training caller is responsible
    for constructing ``params_aug``.

    Parameters
    ----------
    estimator : ConditionalDensityEstimator | JaxConditionalDensityEstimator
        The estimator whose ``_get_log_prob`` is used.
    num_atoms : int
        Number of contrastive atoms (K).  Must be >= 2.
    seed : int
        Seed for the atom-selection RNG (re-seeded each call for
        reproducibility; atom selection itself is pure numpy and does
        not affect the autograd graph).
    """
    if num_atoms < 2:
        raise ValueError(f"num_atoms must be >= 2 for APT, got {num_atoms}.")

    rng_state = [np.random.RandomState(seed)]

    # Detect backend from estimator type
    _is_jax = _HAS_JAX and hasattr(estimator, "_to_jax_key")

    def _atom_select(B: int, K: int) -> np.ndarray:
        """Return (B, K) int array; column 0 = self, others random from batch."""
        rng = rng_state[0]
        idx = np.zeros((B, K), dtype=np.intp)
        idx[:, 0] = np.arange(B)
        for i in range(B):
            pool = np.concatenate([np.arange(i, dtype=np.intp),
                                   np.arange(i + 1, B, dtype=np.intp)])
            idx[i, 1:] = rng.choice(pool, size=K - 1, replace=False)
        return idx

    def apt_loss(weights, features, params_aug):
        """APT loss; params_aug has log_w as last column."""
        if _is_jax:
            import jax.numpy as _jnp
            params    = _jnp.array(params_aug[:, :-1])
            log_w_raw = np.array(params_aug[:, -1])   # pure numpy (not in graph)
        else:
            params    = params_aug[:, :-1]
            log_w_raw = np.array(params_aug[:, -1])

        B = params.shape[0]
        K = min(num_atoms, B)

        atom_idx  = _atom_select(B, K)          # (B, K) numpy int
        flat_idx  = atom_idx.flatten()           # (B*K,)
        rep_idx   = np.repeat(np.arange(B), K)  # (B*K,) - which x_i each atom uses

        # Expand features (B*K, F) and params (B*K, D)
        # Features are pure data - numpy indexing is fine
        feat_exp = np.array(features)[rep_idx]   # (B*K, F) numpy

        # Apply embedding (if any) to the expanded features
        if estimator._emb is not None:
            feat_exp_t = (jnp.array(feat_exp) if _is_jax
                          else anp.asarray(feat_exp, dtype="f"))
            feat_exp_t = estimator._emb.forward(weights, feat_exp_t)
        else:
            feat_exp_t = (jnp.array(feat_exp, dtype="f") if _is_jax
                          else anp.asarray(feat_exp, dtype="f"))

        # Params for atoms - indexed with numpy (not optimised variable)
        params_np = np.array(params)
        params_exp_np = params_np[flat_idx]      # (B*K, D) numpy
        params_exp_t  = (jnp.array(params_exp_np, dtype="f") if _is_jax
                         else anp.asarray(params_exp_np, dtype="f"))

        # Per-sample log_prob for all atom pairs: (B*K,)
        log_q_flat = estimator._get_log_prob(weights, feat_exp_t, params_exp_t)

        # Reshape to (B, K)
        if _is_jax:
            log_q = log_q_flat.reshape(B, K)
        else:
            log_q = anp.reshape(log_q_flat, (B, K))

        # Importance weights for atoms: log_w[atom_idx] (B, K) numpy constant
        log_w_atoms = log_w_raw[flat_idx].reshape(B, K).astype("f")

        # APT loss: logsumexp(log_q + log_w, axis=1) - (log_q[:,0] + log_w[:,0])
        bkd = "jax" if _is_jax else "autograd"
        if _is_jax:
            combined = log_q + jnp.array(log_w_atoms)
        else:
            combined = log_q + anp.asarray(log_w_atoms)

        log_den  = _logsumexp_backend(combined, axis=1, backend=bkd)  # (B,)
        log_num  = log_q[:, 0] + (jnp.array(log_w_atoms[:, 0]) if _is_jax
                                   else anp.asarray(log_w_atoms[:, 0]))  # (B,)

        if _is_jax:
            return jnp.mean(log_den - log_num)
        return anp.mean(log_den - log_num)

    return apt_loss


def compute_log_importance_weights(
    rounds: list,
    prior,
) -> np.ndarray:
    """
    Compute per-sample log importance weights for all accumulated rounds.

    For round r with a non-None proposal q_r:
        log_w_i = log_prior(theta_i) - log_q_r(theta_i)
    For round 1 (proposal = None):
        log_w_i = 0

    Parameters
    ----------
    rounds : list of (theta, x, proposal) tuples
    prior  : prior object with .log_prob(theta) method, or None

    Returns
    -------
    log_w : ndarray (N,) float32
    """
    log_w_parts = []
    for theta_r, _, proposal_r in rounds:
        n = len(theta_r)
        if proposal_r is not None and prior is not None:
            lp = np.array(prior.log_prob(theta_r.astype(np.float64)), dtype="f")
            lq = np.array(proposal_r.log_prob(theta_r.astype(np.float64)), dtype="f")
            # Clip to avoid extreme weights (numerically stable)
            log_w = np.clip(lp - lq, -20.0, 20.0)
        else:
            log_w = np.zeros(n, dtype="f")
        log_w_parts.append(log_w)
    return np.concatenate(log_w_parts, axis=0)
