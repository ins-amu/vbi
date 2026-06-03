"""
JAX Masked Autoregressive Flow conditional density estimator.

Translation of maf.py (autograd) to JAX.

Key differences from the autograd version:
  - jax.numpy instead of autograd.numpy
  - jax.value_and_grad + jax.jit for the training step
  - ActNorm: no data-dependent init (act_s=1, act_b=0 at start; learned
    by gradient descent).  _actnorm_initialized is always True.
  - PRNG in sample(): numpy RandomState is converted to a JAX PRNGKey
  - weights dict holds numpy arrays at rest, converted to jnp inside jitted calls
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import trange

from .base_jax import JaxConditionalDensityEstimator

log = logging.getLogger(__name__)


def _cosine_lr(lr_max: float, lr_min: float, epoch: int, period: int) -> float:
    if epoch >= period:
        return lr_min
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1.0 + math.cos(math.pi * epoch / max(period - 1, 1))
    )


@dataclass
class JaxMAFEstimator(JaxConditionalDensityEstimator):
    """
    Masked Autoregressive Flow - JAX backend.

    Same interface as ``MAFEstimator`` (autograd version).
    """

    param_dim:     int | None    = None
    feature_dim:   int | None    = None
    n_flows:       int           = 4
    hidden_units:  int           = 64
    num_blocks:    int           = 2
    activation:    str           = "tanh"
    z_score_theta: bool          = True
    z_score_x:     bool          = True
    use_actnorm:   bool          = True
    embedding_dim: Optional[int] = None
    actnorm_eps:   float         = 1e-6

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self._dims_inferred      = False
        self.model_constants     = None
        self.theta_mean = self.theta_std = None
        self.x_mean     = self.x_std     = None
        self._use_pca   = False
        self._pca_components     = None
        self.val_loss_history: list[float] = []
        self.best_epoch: int     = -1
        self.best_val_loss: float | None = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _ctx_dim(self) -> int:
        if self.feature_dim == 0:
            return 0
        if self._use_pca and self.embedding_dim is not None:
            return self.embedding_dim
        return self.feature_dim

    def prepare_normalizers(self, features, params):
        n_extra = getattr(self, "_n_apt_extra_cols", 0)
        p = params[:, :params.shape[1] - n_extra] if n_extra > 0 else params
        if self.z_score_theta:
            self.theta_mean = jnp.array(np.mean(p,   axis=0))
            self.theta_std  = jnp.array(np.std(p,    axis=0) + 1e-8)
        else:
            self.theta_mean = jnp.zeros(self.param_dim)
            self.theta_std  = jnp.ones(self.param_dim)

        if self.feature_dim > 0:
            if self.z_score_x:
                self.x_mean = jnp.array(np.mean(features, axis=0))
                self.x_std  = jnp.array(np.std(features,  axis=0) + 1e-8)
            else:
                self.x_mean = jnp.zeros(self.feature_dim)
                self.x_std  = jnp.ones(self.feature_dim)

            if self.embedding_dim is not None and self.embedding_dim < self.feature_dim:
                X = (np.array(features) - np.array(self.x_mean)) / np.array(self.x_std)
                _, _, Vt = np.linalg.svd(X, full_matrices=False)
                self._pca_components = jnp.array(Vt[:self.embedding_dim, :].T)
                self._use_pca = True
            else:
                self._use_pca = False
                self._pca_components = None
        else:
            self.x_mean = jnp.zeros(0)
            self.x_std  = jnp.ones(0)

    def _act(self, x):
        if self.activation == "relu": return jnp.maximum(0.0, x)
        if self.activation == "elu":  return jnp.where(x > 0.0, x, jnp.exp(x) - 1.0)
        return jnp.tanh(x)

    def _z_theta(self, p):
        return (p - self.theta_mean) / self.theta_std

    def _inv_z_theta(self, z):
        return z * self.theta_std + self.theta_mean

    def _z_x(self, f):
        if self.feature_dim == 0:
            return f
        X = (f - self.x_mean) / self.x_std
        if self._use_pca:
            X = jnp.dot(X, self._pca_components)
        return X

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        weights = {}
        layers  = []
        D, C, H = self.param_dim, self._ctx_dim(), self.hidden_units

        for k in range(self.n_flows):
            m_in     = np.arange(1, D + 1)
            m_hidden = rng.randint(1, D + 1, size=H)
            M1       = (m_in[None, :] <= m_hidden[:, None]).astype("f")
            M2       = (m_hidden[None, :] < m_in[:, None]).astype("f")
            Mm       = (m_hidden[None, :] <= m_hidden[:, None]).astype("f")
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
            weights[f"W2_{k}"]  = np.zeros((2 * D, H), "f")
            weights[f"W2c_{k}"] = np.zeros((2 * D, C), "f")
            b2 = np.zeros(2 * D, "f"); b2[D:] = -2.0
            weights[f"b2_{k}"]  = b2

            if self.use_actnorm:
                weights[f"act_s_{k}"] = np.ones(D,  "f")
                weights[f"act_b_{k}"] = np.zeros(D, "f")

        self.model_constants = {"layers": layers}
        if self._emb is not None:
            weights.update(self._emb.init_weights(np.random.RandomState(seed)))
        return weights

    # ------------------------------------------------------------------
    # MADE forward and ActNorm
    # ------------------------------------------------------------------

    def _made_forward(self, y, ctx, lc, k, weights):
        M1, M2, Mm = lc["M1"], lc["M2"], lc["Mm"]
        W1y = weights[f"W1y_{k}"]
        W1c = weights[f"W1c_{k}"]
        b1  = weights[f"b1_{k}"]
        W2  = weights[f"W2_{k}"]
        W2c = weights[f"W2c_{k}"]
        b2  = weights[f"b2_{k}"]

        h = self._act(jnp.dot(y, (W1y * M1).T) +
                      (jnp.dot(ctx, W1c.T) if self._ctx_dim() > 0 else 0.0) + b1)
        for i in range(self.num_blocks - 1):
            h = self._act(jnp.dot(h, (weights[f"Wm_{k}_{i}"] * Mm).T) + weights[f"bm_{k}_{i}"])
        M2t = jnp.concatenate([M2, M2], axis=0)
        out = jnp.dot(h, (W2 * M2t).T)
        if self._ctx_dim() > 0:
            out = out + jnp.dot(ctx, W2c.T)
        out     = out + b2
        mu      = out[:, :self.param_dim]
        log_sig = jnp.clip(out[:, self.param_dim:], -7.0, 7.0)
        return mu, log_sig

    def _apply_actnorm(self, u, k, weights):
        """ActNorm forward: v = (u + b) * s, log|s| contribution to log_det."""
        if not self.use_actnorm:
            return u, 0.0
        s = weights[f"act_s_{k}"]
        b = weights[f"act_b_{k}"]
        return (u + b) * s, jnp.sum(jnp.log(jnp.abs(s) + 1e-12))

    def _forward_one_step(self, v, x, lc, k, w):
        """Propagate v through one flow step (affine); subclasses override for spline."""
        mu, log_sig = self._made_forward(v, x, lc, k, w)
        return (v - mu) * jnp.exp(-log_sig)

    def _init_actnorm_data_dependent(self, weights_np: dict, features, params) -> dict:
        """
        One-shot data-dependent ActNorm init, run once before the training loop.

        For each flow step k, sets act_b_k = -mean(u_k) and act_s_k = 1/std(u_k)
        so that the pre-ActNorm latent has zero mean and unit std.  This mirrors
        the autograd MAF's lazy first-forward init and prevents the exp(-2) log_sig
        bias from collapsing samples to the prior mean before training starts.
        Subclasses override _forward_one_step to plug in the correct transform.
        """
        if not self.use_actnorm:
            return weights_np

        N = min(500, len(params))
        f = jnp.array(features[:N], dtype="f")
        p = jnp.array(params[:N],   dtype="f")
        w = {k: jnp.array(v) for k, v in weights_np.items()}

        if self._emb is not None:
            f = self._emb.forward(w, f)
        x = self._z_x(f).astype("f") if self._ctx_dim() > 0 else f.astype("f")
        u = self._z_theta(p).astype("f")

        new_w = dict(weights_np)

        for k, lc in enumerate(self.model_constants["layers"]):
            u_perm = u[:, lc["perm"]]
            mean_u = np.array(jnp.mean(u_perm, axis=0))
            std_u  = np.array(jnp.std(u_perm,  axis=0)) + self.actnorm_eps
            new_w[f"act_b_{k}"] = (-mean_u).astype("f")
            new_w[f"act_s_{k}"] = (1.0 / std_u).astype("f")

            w[f"act_b_{k}"] = jnp.array(new_w[f"act_b_{k}"])
            w[f"act_s_{k}"] = jnp.array(new_w[f"act_s_{k}"])
            v_k, _ = self._apply_actnorm(u_perm, k, w)
            u = self._forward_one_step(v_k, x, lc, k, w)

        return new_w

    # ------------------------------------------------------------------
    # Log-probability
    # ------------------------------------------------------------------

    def _get_log_prob(self, weights, features, params):
        if self._emb is not None:
            features = self._emb.forward(weights, features)
        x = self._z_x(features).astype("f") if self._ctx_dim() > 0 else features.astype("f")
        u = self._z_theta(params).astype("f")
        log_det = jnp.zeros(u.shape[0], "f")

        for k, lc in enumerate(self.model_constants["layers"]):
            u = u[:, lc["perm"]]
            v, ld = self._apply_actnorm(u, k, weights)
            if self.use_actnorm:
                log_det = log_det + ld
            mu, log_sig = self._made_forward(v, x, lc, k, weights)
            u = (v - mu) * jnp.exp(-log_sig)
            log_det -= jnp.sum(log_sig, axis=1)

        base = -0.5 * jnp.sum(u ** 2, axis=1) - 0.5 * self.param_dim * jnp.log(2.0 * jnp.pi)
        return base + log_det

    def _loss_function(self, weights, features, params) -> float:
        return -jnp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features, params):
        super().log_prob(features, params)
        w = {k: jnp.array(v) for k, v in self.weights.items()}
        f = jnp.array(features, dtype="f")
        p = jnp.array(params,   dtype="f")
        return np.array(self._get_log_prob(w, f, p))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, features, n_samples: int, rng):
        """Returns shape (n_conditions, n_samples, param_dim)."""
        super().sample(features, n_samples, rng)
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

        for c in range(n_cond):
            key, subkey = jax.random.split(key)
            y = jax.random.normal(subkey, (n_samples, self.param_dim)).astype("f")

            for k_idx, lc in reversed(list(enumerate(self.model_constants["layers"]))):
                ctx = jnp.tile(x[c:c+1], (n_samples, 1))
                v_  = jnp.zeros_like(y)
                for i in range(self.param_dim):
                    mu, log_sig = self._made_forward(
                        v_, ctx, lc, k_idx, w)
                    v_ = v_.at[:, i].set(y[:, i] * jnp.exp(log_sig[:, i]) + mu[:, i])
                if self.use_actnorm:
                    s = w[f"act_s_{k_idx}"]
                    b = w[f"act_b_{k_idx}"]
                    v_ = v_ / (s + 1e-12) - b
                y = v_[:, lc["inv_perm"]]
            out[c] = np.array(self._inv_z_theta(y))
        return out

    # ------------------------------------------------------------------
    # Full training (overrides base - adds val split, LR schedule, etc.)
    # ------------------------------------------------------------------

    def train(
        self,
        params,
        features,
        n_iter: int = 2000,
        learning_rate: float = 1e-3,
        seed: int = 0,
        batch_size: int | None = None,
        verbose: bool = True,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        early_stopping_delta: float | None = 0.0,
        clip_max_norm: float | None = 5.0,
        resume_training: bool = False,
        lr_schedule: str | None = None,
        lr_min: float = 1e-5,
        lr_period: int = 500,
        monitor_collapse: bool = False,
        x_check=None,
        collapse_threshold: float = 0.05,
        check_every: int = 10,
        n_check: int = 200,
        loss_fn=None,
    ):
        params   = np.asarray(params, dtype="f")
        features = np.asarray(features, dtype="f")

        if not self._dims_inferred:
            self._infer_dimensions(params, features)
        if self._emb is not None:
            self.feature_dim = self._emb.output_dim

        if params.shape[0] != features.shape[0]:
            raise ValueError("params and features must have the same number of rows.")

        finite   = (np.all(np.isfinite(params),   axis=1) &
                    np.all(np.isfinite(features), axis=1))
        params   = params[finite]
        features = features[finite]
        if params.shape[0] == 0:
            raise ValueError("All data points were non-finite.")

        N      = params.shape[0]
        rng_np = np.random.RandomState(seed)

        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0, 1).")
        n_val     = int(N * validation_fraction)
        perm_idx  = rng_np.permutation(N)
        val_idx   = perm_idx[:n_val]   if n_val > 0 else np.array([], dtype=int)
        train_idx = perm_idx[n_val:]

        p_tr, f_tr = params[train_idx], features[train_idx]
        p_val, f_val = ((params[val_idx], features[val_idx])
                        if n_val > 0 else (None, None))

        # For APT: p_tr has an extra log_w column; strip it for init-only uses.
        # Mini-batches keep the full p_tr so apt_loss can read log_w from last col.
        _n_extra = getattr(self, "_n_apt_extra_cols", 0)
        p_tr_real = p_tr[:, :-_n_extra] if _n_extra else p_tr
        p_val_real = p_val[:, :-_n_extra] if (_n_extra and p_val is not None) else p_val

        if early_stopping_delta is None:
            early_stopping_delta = 1e-4 if lr_schedule == "cosine" else 0.0

        # Compute normalizers
        if self._emb is not None:
            _rng_norm = np.random.RandomState(seed)
            _init_emb_w = self._emb.init_weights(_rng_norm)
            f_tr_norm = self._emb.forward(_init_emb_w, f_tr)
        else:
            f_tr_norm = f_tr
        self.prepare_normalizers(f_tr_norm, p_tr_real)

        rng_w = np.random.RandomState(seed)
        if resume_training and self.weights is not None:
            weights_np = {k: np.array(v) for k, v in self.weights.items()}
        else:
            weights_np = self._initialize_weights(seed)
            weights_np = self._init_actnorm_data_dependent(weights_np, f_tr, p_tr_real)
        self.loss_history     = []
        self.val_loss_history = []

        # Convert to JAX
        weights = {k: jnp.array(v) for k, v in weights_np.items()}
        f_tr_j  = jnp.array(f_tr)
        p_tr_j  = jnp.array(p_tr)
        f_val_j = jnp.array(f_val) if f_val is not None else None
        p_val_j = jnp.array(p_val) if p_val is not None else None

        m = {k: jnp.zeros_like(v) for k, v in weights.items()}
        v = {k: jnp.zeros_like(v) for k, v in weights.items()}
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        # Compile the full Adam step into one jit call (default path).
        # APT: when loss_fn is provided (atom-sampling involves numpy → not jit-able),
        # fall back to eager grad + manual Adam so the APT loss can run per-step.
        if loss_fn is not None:
            _train_step = None
            _apt_grad = jax.value_and_grad(loss_fn)
        else:
            _train_step = self._make_train_step(clip_max_norm, beta1, beta2, eps_adam)
            _apt_grad   = None

        best_weights     = {k: np.array(v) for k, v in weights.items()}
        best_val         = np.inf if n_val > 0 else None
        epochs_no_improv = 0
        self.best_epoch    = -1
        self.best_val_loss = None

        _data_std = p_tr_real.std(axis=0) + 1e-8
        _stopped_for_collapse = False
        if monitor_collapse:
            _x_chk = (np.atleast_2d(np.asarray(x_check, dtype="f"))
                      if x_check is not None else f_tr[0:1].copy())
            _rng_chk = np.random.RandomState(seed + 999)
            _last_healthy_weights = {k: np.array(v) for k, v in weights.items()}
            _max_seen_std = None

        iterator = trange(n_iter, desc="Training", disable=not verbose)

        for epoch in iterator:
            lr_t = (_cosine_lr(learning_rate, lr_min, epoch, lr_period)
                    if lr_schedule == "cosine" else learning_rate)

            if batch_size is not None and batch_size < len(p_tr):
                idx  = rng_np.choice(len(p_tr), size=batch_size, replace=False)
                f_b  = f_tr_j[idx]
                p_b  = p_tr_j[idx]
            else:
                f_b, p_b = f_tr_j, p_tr_j

            if _train_step is not None:
                weights, m, v, train_loss = _train_step(
                    weights, m, v, f_b, p_b,
                    jnp.float32(lr_t), jnp.int32(epoch + 1),
                )
                train_loss_f = float(train_loss)
            else:
                # APT eager path: atom-sampling happens inside loss_fn
                train_loss_val, g_apt = _apt_grad(weights, f_b, p_b)
                train_loss_f = float(train_loss_val)
                # Manual Adam + grad clip
                total_sq = sum(float(jnp.sum(g_apt[k] ** 2)) for k in g_apt)
                norm = (total_sq + 1e-12) ** 0.5
                scale = (clip_max_norm / (norm + 1e-12)) if (clip_max_norm and norm > clip_max_norm) else 1.0
                new_weights = {}
                for key in weights:
                    gk = g_apt[key] * scale
                    m[key] = beta1 * m[key] + (1 - beta1) * gk
                    v[key] = beta2 * v[key] + (1 - beta2) * gk ** 2
                    m_hat  = m[key] / (1 - beta1 ** (epoch + 1))
                    v_hat  = v[key] / (1 - beta2 ** (epoch + 1))
                    new_weights[key] = weights[key] - lr_t * m_hat / (jnp.sqrt(v_hat) + eps_adam)
                weights = new_weights

            self.loss_history.append(train_loss_f)

            if not math.isfinite(train_loss_f):
                log.warning("Non-finite loss at epoch %d; restoring best weights.", epoch)
                weights = {k: jnp.array(best_weights[k]) for k in best_weights}
                break

            # Validation + early stopping
            active_loss_val = loss_fn if loss_fn is not None else self._loss_function
            if n_val > 0:
                val_loss  = float(active_loss_val(weights, f_val_j, p_val_j))
                self.val_loss_history.append(val_loss)
                improved  = (best_val - val_loss) > early_stopping_delta
                if improved:
                    best_val     = val_loss
                    best_weights = {k: np.array(v_) for k, v_ in weights.items()}
                    epochs_no_improv = 0
                    self.best_epoch    = epoch
                    self.best_val_loss = val_loss
                else:
                    epochs_no_improv += 1

                if verbose:
                    try:
                        iterator.set_postfix(
                            train=f"{train_loss_f:.4f}",
                            val=f"{val_loss:.4f}",
                            lr=f"{lr_t:.2e}",
                            patience=f"{epochs_no_improv}/{stop_after_epochs}",
                        )
                    except Exception:
                        pass

                if epochs_no_improv >= stop_after_epochs:
                    weights = {k: jnp.array(best_weights[k]) for k in best_weights}
                    break
            else:
                if verbose:
                    try:
                        iterator.set_postfix(train=f"{train_loss_f:.4f}", lr=f"{lr_t:.2e}")
                    except Exception:
                        pass

            # Collapse monitoring
            if monitor_collapse and (epoch + 1) % check_every == 0:
                try:
                    self.weights = {k: np.array(v_) for k, v_ in weights.items()}
                    chk     = self.sample(_x_chk, n_check, _rng_chk)
                    chk_std = chk[0].std(axis=0)

                    if _max_seen_std is None:
                        _max_seen_std = chk_std.copy()
                    else:
                        _max_seen_std = np.maximum(_max_seen_std, chk_std)

                    ref = _max_seen_std
                    if np.any(chk_std < collapse_threshold * ref):
                        log.info("Collapse detected at epoch %d. Restoring last healthy weights.", epoch)
                        _stopped_for_collapse = True
                        weights = {k: jnp.array(_last_healthy_weights[k]) for k in _last_healthy_weights}
                        break
                    else:
                        _last_healthy_weights = {k: np.array(v_) for k, v_ in weights.items()}
                except Exception as exc:
                    log.debug("Collapse monitor skipped at epoch %d: %s", epoch, exc)

        if n_val > 0 and not _stopped_for_collapse:
            self.weights = best_weights
        else:
            self.weights = {k: np.array(v_) for k, v_ in weights.items()}

    # ------------------------------------------------------------------
    # Jitted training step
    # ------------------------------------------------------------------

    def _make_train_step(self, clip_max_norm, beta1, beta2, eps_adam):
        """
        Return a jit-compiled function that does one full Adam step.

        Signature: (weights, m, v, features, params, lr, t) →
                   (new_weights, new_m, new_v, loss)

        Everything - forward pass, backward pass, gradient clipping, and Adam
        update - runs inside a single XLA kernel.  The caller only needs to
        extract the scalar ``loss`` (one sync point per step instead of ~3×N_keys).
        """
        loss_fn = self._loss_function

        @jax.jit
        def step(weights, m, v, features, params, lr, t):
            loss, g = jax.value_and_grad(loss_fn)(weights, features, params)

            # Gradient clipping - fully inside jit
            if clip_max_norm is not None:
                leaves   = jax.tree_util.tree_leaves(g)
                total_sq = sum(jnp.sum(gi ** 2) for gi in leaves)
                norm     = jnp.sqrt(total_sq + 1e-12)
                scale    = jnp.minimum(jnp.float32(1.0),
                                       jnp.float32(clip_max_norm) / (norm + 1e-12))
                g = jax.tree_util.tree_map(lambda gi: gi * scale, g)

            # Adam - fully inside jit
            m = jax.tree_util.tree_map(
                lambda mk, gk: beta1 * mk + (1.0 - beta1) * gk, m, g)
            v = jax.tree_util.tree_map(
                lambda vk, gk: beta2 * vk + (1.0 - beta2) * gk ** 2, v, g)
            t_f = t.astype(jnp.float32)
            m_hat = jax.tree_util.tree_map(
                lambda mk: mk / (1.0 - jnp.float32(beta1) ** t_f), m)
            v_hat = jax.tree_util.tree_map(
                lambda vk: vk / (1.0 - jnp.float32(beta2) ** t_f), v)
            new_w = jax.tree_util.tree_map(
                lambda wk, mhk, vhk: wk - lr * mhk / (jnp.sqrt(vhk) + eps_adam),
                weights, m_hat, v_hat,
            )
            return new_w, m, v, loss

        return step

    def reinitialize(self, seed: int = 0):
        weights = self._initialize_weights(seed)
        self.weights = weights
