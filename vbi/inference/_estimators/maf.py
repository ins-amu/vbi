"""Masked Autoregressive Flow conditional density estimator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import autograd.numpy as anp
from autograd import grad
from tqdm.auto import trange

from .base import ConditionalDensityEstimator


@dataclass
class MAFEstimator(ConditionalDensityEstimator):
    """
    Masked Autoregressive Flow for conditional density estimation.

    Implements MADE blocks with optional ActNorm, random permutations,
    z-score normalisation, and PCA embedding of features.

    Parameters
    ----------
    param_dim, feature_dim : int | None
    n_flows : int
        Number of MADE blocks (autoregressive transforms).
    hidden_units : int
        Hidden units per MADE block.
    activation : 'tanh' | 'relu' | 'elu'
    z_score_theta, z_score_x : bool
        Standardise parameters / features internally.
    use_actnorm : bool
        Insert data-dependent ActNorm between flows.
    embedding_dim : int | None
        If set, PCA-compress features to this dimension before conditioning.
    """

    param_dim:     int | None = None
    feature_dim:   int | None = None
    n_flows:       int        = 4
    hidden_units:  int        = 64
    activation:    str        = "tanh"
    z_score_theta: bool       = True
    z_score_x:     bool       = True
    use_actnorm:   bool       = True
    embedding_dim: Optional[int] = None
    actnorm_eps:   float      = 1e-6

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self._dims_inferred         = False
        self.model_constants        = None
        self._actnorm_initialized   = [False] * self.n_flows
        self.theta_mean = self.theta_std = None
        self.x_mean     = self.x_std     = None
        self._use_pca = False
        self._pca_components = None
        self.val_loss_history: list[float] = []
        self.best_epoch: int = -1
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
        if self.z_score_theta:
            self.theta_mean = anp.mean(params,   axis=0)
            self.theta_std  = anp.std(params,    axis=0) + 1e-8
        else:
            self.theta_mean = anp.zeros(self.param_dim)
            self.theta_std  = anp.ones(self.param_dim)

        if self.feature_dim > 0:
            if self.z_score_x:
                self.x_mean = anp.mean(features, axis=0)
                self.x_std  = anp.std(features,  axis=0) + 1e-8
            else:
                self.x_mean = anp.zeros(self.feature_dim)
                self.x_std  = anp.ones(self.feature_dim)

            if self.embedding_dim is not None and self.embedding_dim < self.feature_dim:
                X = (features - self.x_mean) / self.x_std
                _, _, Vt = anp.linalg.svd(X, full_matrices=False)
                self._pca_components = Vt[:self.embedding_dim, :].T
                self._use_pca = True
            else:
                self._use_pca = False
                self._pca_components = None
        else:
            self.x_mean = anp.zeros(0)
            self.x_std  = anp.ones(0)

    def _act(self, x):
        if self.activation == "relu": return anp.maximum(0.0, x)
        if self.activation == "elu":  return anp.where(x > 0.0, x, anp.exp(x) - 1.0)
        return anp.tanh(x)

    def _z_theta(self, p): return (p - self.theta_mean) / self.theta_std
    def _inv_z_theta(self, z): return z * self.theta_std + self.theta_mean

    def _z_x(self, f):
        if self.feature_dim == 0: return f
        X = (f - self.x_mean) / self.x_std
        if self._use_pca: X = anp.dot(X, self._pca_components)
        return X

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self, rng) -> dict:
        weights = {}
        layers  = []
        D, C, H = self.param_dim, self._ctx_dim(), self.hidden_units

        for k in range(self.n_flows):
            m_in     = anp.arange(1, D + 1)
            m_hidden = rng.randint(1, D + 1, size=H)
            M1       = (m_in[None, :] <= m_hidden[:, None]).astype("f")
            M2       = (m_hidden[None, :] < m_in[:, None]).astype("f")
            perm     = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int)
            inv_perm[perm] = anp.arange(D)
            layers.append({"M1": M1, "M2": M2, "perm": perm, "inv_perm": inv_perm})

            w = 0.01
            weights[f"W1y_{k}"] = (rng.randn(H, D) * w).astype("f")
            weights[f"W1c_{k}"] = (rng.randn(H, C) * w).astype("f") if C > 0 else anp.zeros((H, C), "f")
            weights[f"b1_{k}"]  = anp.zeros(H, "f")
            weights[f"W2_{k}"]  = anp.zeros((2 * D, H), "f")
            weights[f"W2c_{k}"] = anp.zeros((2 * D, C), "f")
            b2 = anp.zeros(2 * D, "f"); b2[D:] = -2.0
            weights[f"b2_{k}"]  = b2

            if self.use_actnorm:
                weights[f"act_s_{k}"] = anp.ones(D,  "f")
                weights[f"act_b_{k}"] = anp.zeros(D, "f")

        self.model_constants = {"layers": layers}
        return weights

    # ------------------------------------------------------------------
    # MADE forward and ActNorm
    # ------------------------------------------------------------------

    def _made_forward(self, y, ctx, lc, k, weights):
        M1, M2   = lc["M1"], lc["M2"]
        W1y, W1c = weights[f"W1y_{k}"], weights[f"W1c_{k}"]
        b1       = weights[f"b1_{k}"]
        W2, W2c  = weights[f"W2_{k}"],  weights[f"W2c_{k}"]
        b2       = weights[f"b2_{k}"]

        h   = self._act(anp.dot(y, (W1y * M1).T) +
                        (anp.dot(ctx, W1c.T) if self._ctx_dim() > 0 else 0.0) + b1)
        M2t = anp.concatenate([M2, M2], axis=0)
        out = anp.dot(h, (W2 * M2t).T)
        if self._ctx_dim() > 0: out = out + anp.dot(ctx, W2c.T)
        out = out + b2
        mu       = out[:, :self.param_dim]
        log_sig  = anp.clip(out[:, self.param_dim:], -7.0, 7.0)
        return mu, log_sig

    def _apply_actnorm(self, u, k, weights, maybe_init=None):
        if not self.use_actnorm: return u, 0.0
        s = weights[f"act_s_{k}"]
        b = weights[f"act_b_{k}"]
        if not self._actnorm_initialized[k] and maybe_init is not None:
            b = -anp.mean(maybe_init, axis=0)
            s = 1.0 / (anp.std(maybe_init, axis=0) + self.actnorm_eps)
            weights[f"act_s_{k}"] = s.astype("f")
            weights[f"act_b_{k}"] = b.astype("f")
            self._actnorm_initialized[k] = True
        return (u + b) * s, anp.sum(anp.log(anp.abs(s) + 1e-12))

    # ------------------------------------------------------------------
    # Log-probability
    # ------------------------------------------------------------------

    def _get_log_prob(self, weights, features, params):
        x = self._z_x(features).astype("f") if self._ctx_dim() > 0 else features.astype("f")
        u = self._z_theta(params).astype("f")
        log_det = anp.zeros(u.shape[0], "f")

        for k, lc in enumerate(self.model_constants["layers"]):
            u = u[:, lc["perm"]]
            v, ld = self._apply_actnorm(u, k, weights, maybe_init=u)
            if self.use_actnorm: log_det = log_det + ld
            mu, log_sig = self._made_forward(v, x, lc, k, weights)
            u = (v - mu) * anp.exp(-log_sig)
            log_det -= anp.sum(log_sig, axis=1)

        base = -0.5 * anp.sum(u ** 2, axis=1) - 0.5 * self.param_dim * anp.log(2.0 * anp.pi)
        return base + log_det

    def _loss_function(self, weights, features, params) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features, params) -> anp.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, features, n_samples: int, rng) -> anp.ndarray:
        """Returns shape (n_conditions, n_samples, param_dim)."""
        super().sample(features, n_samples, rng)
        features = anp.asarray(features, dtype="f")
        if features.ndim == 1 and self.feature_dim > 0:
            features = features.reshape(1, -1)
        n_cond = 1 if self.feature_dim == 0 else features.shape[0]

        x   = self._z_x(features).astype("f") if self._ctx_dim() > 0 else anp.zeros((n_cond, 0), "f")
        out = anp.zeros((n_cond, n_samples, self.param_dim), "f")

        for c in range(n_cond):
            y = rng.randn(n_samples, self.param_dim).astype("f")
            for k, lc in reversed(list(enumerate(self.model_constants["layers"]))):
                v = anp.zeros_like(y)
                for i in range(self.param_dim):
                    mu, log_sig = self._made_forward(
                        v, x[c:c+1].repeat(n_samples, axis=0), lc, k, self.weights)
                    v[:, i] = y[:, i] * anp.exp(log_sig[:, i]) + mu[:, i]
                if self.use_actnorm:
                    s = self.weights[f"act_s_{k}"]
                    b = self.weights[f"act_b_{k}"]
                    v = v / (s + 1e-12) - b
                y = v[:, lc["inv_perm"]]
            out[c] = self._inv_z_theta(y)
        return out

    # ------------------------------------------------------------------
    # Training (full override with validation split, early stopping, mini-batch)
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
        early_stopping_delta: float = 0.0,
        clip_max_norm: float | None = 5.0,
    ):
        """
        Train with Adam, optional mini-batches, train/val split, and
        early stopping.

        Parameters
        ----------
        batch_size : int | None
            Mini-batch size per gradient step.  None → full training set.
        verbose : bool
            Show tqdm progress bar.
        All other parameters mirror sbi's ``SNPE.train()`` kwarg names.
        """
        params   = anp.asarray(params)
        features = anp.asarray(features)

        if not self._dims_inferred:
            self._infer_dimensions(params, features)

        if params.shape[0] != features.shape[0]:
            raise ValueError("params and features must have the same number of rows.")

        finite   = (anp.all(anp.isfinite(params),   axis=1) &
                    anp.all(anp.isfinite(features), axis=1))
        params   = params[finite].astype("f")
        features = features[finite].astype("f")
        if params.shape[0] == 0:
            raise ValueError("All data points were non-finite.")

        N      = params.shape[0]
        rng_np = anp.random.RandomState(seed)

        # Train / validation split
        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0, 1).")
        n_val     = int(N * validation_fraction)
        perm      = rng_np.permutation(N)
        val_idx   = perm[:n_val]   if n_val > 0 else anp.array([], dtype=int)
        train_idx = perm[n_val:]

        p_tr, f_tr = params[train_idx], features[train_idx]
        p_val, f_val = ((params[val_idx], features[val_idx])
                        if n_val > 0 else (None, None))

        self.prepare_normalizers(f_tr, p_tr)

        rng_w = anp.random.RandomState(seed)
        self.weights = self._initialize_weights(rng_w)
        self.loss_history     = []
        self.val_loss_history = []
        self._actnorm_initialized = [False] * self.n_flows

        # ActNorm warmup
        if self.use_actnorm:
            k = min(512, len(p_tr))
            _ = self._get_log_prob(self.weights, f_tr[:k], p_tr[:k])

        m = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        v = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        gradient_func = grad(self._loss_function)
        iterator      = trange(n_iter, desc="Training", disable=not verbose)

        best_weights     = {k: w.copy() for k, w in self.weights.items()}
        best_val         = anp.inf if n_val > 0 else None
        epochs_no_improv = 0
        self.best_epoch    = -1
        self.best_val_loss = None

        for epoch in iterator:
            # Mini-batch selection on training set
            if batch_size is not None and batch_size < len(p_tr):
                idx    = rng_np.choice(len(p_tr), size=batch_size, replace=False)
                f_b, p_b = f_tr[idx], p_tr[idx]
            else:
                f_b, p_b = f_tr, p_tr

            g          = gradient_func(self.weights, f_b, p_b)
            train_loss = self._loss_function(self.weights, f_b, p_b)
            self.loss_history.append(float(train_loss))

            # Gradient clipping
            if clip_max_norm is not None:
                total_sq = sum(anp.sum(g[k] ** 2) for k in g)
                norm     = anp.sqrt(total_sq + 1e-12)
                if norm > clip_max_norm:
                    scale = clip_max_norm / (norm + 1e-12)
                    g = {k: g[k] * scale for k in g}

            # Adam update
            for key in self.weights:
                if not anp.all(anp.isfinite(g[key])):
                    self.weights = best_weights
                    return
                m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                v[key] = beta2 * v[key] + (1 - beta2) * g[key] ** 2
                m_hat  = m[key] / (1 - beta1 ** (epoch + 1))
                v_hat  = v[key] / (1 - beta2 ** (epoch + 1))
                self.weights[key] -= learning_rate * m_hat / (anp.sqrt(v_hat) + eps)

            # Validation + early stopping
            if n_val > 0:
                val_loss = self._loss_function(self.weights, f_val, p_val)
                self.val_loss_history.append(float(val_loss))
                improved = (best_val - val_loss) > early_stopping_delta
                if improved:
                    best_val     = float(val_loss)
                    best_weights = {k: w.copy() for k, w in self.weights.items()}
                    epochs_no_improv = 0
                    self.best_epoch    = epoch
                    self.best_val_loss = float(val_loss)
                else:
                    epochs_no_improv += 1

                if verbose:
                    try:
                        iterator.set_postfix(
                            train=f"{train_loss:.4f}",
                            val=f"{val_loss:.4f}",
                            patience=f"{epochs_no_improv}/{stop_after_epochs}",
                        )
                    except Exception:
                        pass

                if epochs_no_improv >= stop_after_epochs:
                    self.weights = best_weights
                    break
            else:
                if verbose:
                    try:
                        iterator.set_postfix(train=f"{train_loss:.4f}")
                    except Exception:
                        pass

        if n_val > 0:
            self.weights = best_weights

    def reinitialize(self, rng=None):
        if rng is None:
            rng = anp.random.RandomState(0)
        self.weights = self._initialize_weights(rng)
        self._actnorm_initialized = [False] * self.n_flows


# ---------------------------------------------------------------------------
# Legacy MAF (simpler, no ActNorm / z-scoring) — kept for backward compat
# ---------------------------------------------------------------------------

@dataclass
class MAFEstimator0(ConditionalDensityEstimator):
    """Original simpler MAF.  Deprecated — use MAFEstimator instead."""

    param_dim:    int = None
    feature_dim:  int = None
    n_flows:      int = 4
    hidden_units: int = 64

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self.model_constants = None

    def _initialize_weights(self, rng) -> dict:
        weights, layers = {}, []
        D, C, H = self.param_dim, self.feature_dim, self.hidden_units
        for k in range(self.n_flows):
            m_in     = anp.arange(1, D + 1)
            m_h      = rng.randint(1, D + 1, size=H)
            M1       = (m_in[None, :] <= m_h[:, None]).astype("f")
            M2       = (m_h[None, :] < m_in[:, None]).astype("f")
            perm     = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int); inv_perm[perm] = anp.arange(D)
            layers.append({"M1": M1, "M2": M2, "perm": perm, "inv_perm": inv_perm})
            weights[f"W1y_{k}"] = (rng.randn(H, D) * 0.01).astype("f")
            weights[f"W1c_{k}"] = (rng.randn(H, C) * 0.01).astype("f") if C > 0 else anp.zeros((H, C), "f")
            weights[f"b1_{k}"]  = anp.zeros(H, "f")
            weights[f"W2_{k}"]  = anp.zeros((2 * D, H), "f")
            weights[f"W2c_{k}"] = anp.zeros((2 * D, C), "f")
            weights[f"b2_{k}"]  = anp.zeros(2 * D, "f")
        self.model_constants = {"layers": layers}
        return weights

    def _made_forward(self, y, ctx, lc, k, weights):
        M1, M2 = lc["M1"], lc["M2"]
        h = anp.tanh(anp.dot(y, (weights[f"W1y_{k}"] * M1).T) +
                     (anp.dot(ctx, weights[f"W1c_{k}"].T) if self.feature_dim > 0 else 0.0) +
                     weights[f"b1_{k}"])
        M2t = anp.concatenate([M2, M2], axis=0)
        out = anp.dot(h, (weights[f"W2_{k}"] * M2t).T)
        if self.feature_dim > 0:
            out = out + anp.dot(ctx, weights[f"W2c_{k}"].T)
        out = out + weights[f"b2_{k}"]
        return out[:, :self.param_dim], anp.clip(out[:, self.param_dim:], -7.0, 7.0)

    def _get_log_prob(self, weights, features, params):
        u       = params
        log_det = anp.zeros(params.shape[0])
        for k, lc in enumerate(self.model_constants["layers"]):
            u = u[:, lc["perm"]]
            mu, alpha = self._made_forward(u, features, lc, k, weights)
            u = (u - mu) * anp.exp(-alpha)
            log_det -= anp.sum(alpha, axis=1)
        base = -0.5 * anp.sum(u ** 2, axis=1) - 0.5 * self.param_dim * anp.log(2.0 * anp.pi)
        return base + log_det

    def _loss_function(self, weights, features, params) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features, params) -> anp.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    def sample(self, features, n_samples: int, rng) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = anp.asarray(features, dtype="f")
        if features.ndim == 1: features = features.reshape(1, -1)
        n_cond = features.shape[0]
        if n_cond != n_samples:
            features = anp.repeat(features, n_samples, axis=0)
        z = rng.randn(n_samples, self.param_dim).astype("f")
        x = z
        for k, lc in reversed(list(enumerate(self.model_constants["layers"]))):
            y_perm = x
            u = anp.zeros_like(y_perm)
            for i in range(self.param_dim):
                mu, alpha = self._made_forward(u, features, lc, k, self.weights)
                u[:, i] = y_perm[:, i] * anp.exp(alpha[:, i]) + mu[:, i]
            x = u[:, lc["inv_perm"]]
        return x.reshape(features.shape[0] // n_samples, n_samples, self.param_dim)
