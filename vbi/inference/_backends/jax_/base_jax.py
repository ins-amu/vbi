"""
Abstract base for JAX-native conditional density estimators.

Drop-in replacement for the autograd ConditionalDensityEstimator.
Key differences:
  - weights dict holds jax.numpy arrays (not autograd arrays)
  - grad computed via jax.grad / jax.value_and_grad
  - training step is jax.jit-compiled for speed
  - PRNG: sample() accepts numpy RandomState (converted internally to JAX key)
"""
from __future__ import annotations

import abc
import logging

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import trange

log = logging.getLogger(__name__)


class JaxConditionalDensityEstimator(abc.ABC):
    """
    Abstract base for JAX-native conditional density estimators.

    Same public interface as ``ConditionalDensityEstimator`` so the
    ``Posterior`` and ``SNPE`` layers work identically.
    """

    def __init__(self, param_dim: int = None, feature_dim: int = None):
        self.param_dim   = param_dim
        self.feature_dim = feature_dim
        self._dims_inferred = False
        self.weights: dict | None = None
        self.loss_history: list[float] = []
        self._emb = None

    # ------------------------------------------------------------------
    # Embedding support (mirrors base.py)
    # ------------------------------------------------------------------

    def set_embedding(self, emb) -> None:
        self._emb = emb

    def _wrapped_loss(self, weights, features, params) -> float:
        if self._emb is not None:
            features = self._emb.forward(weights, features)
        return self._loss_function(weights, features, params)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _initialize_weights(self, seed: int) -> dict:
        pass

    @abc.abstractmethod
    def _loss_function(self, weights: dict, features, params) -> float:
        pass

    @abc.abstractmethod
    def sample(self, features, n_samples: int, rng):
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    @abc.abstractmethod
    def log_prob(self, features, params):
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    # ------------------------------------------------------------------
    # Dimension inference
    # ------------------------------------------------------------------

    def _infer_dimensions(self, params, features):
        params   = np.asarray(params)
        features = np.asarray(features)
        if params.ndim   == 1: params   = params.reshape(-1, 1)
        if features.ndim == 1: features = features.reshape(-1, 1)

        n_extra = getattr(self, "_n_apt_extra_cols", 0)
        self.param_dim   = params.shape[1] - n_extra
        self.feature_dim = features.shape[1]
        self._dims_inferred = True

        log.debug("Inferred dimensions: param_dim=%d  feature_dim=%d",
                  self.param_dim, self.feature_dim)

        if self.param_dim <= 0:
            raise ValueError(f"param_dim must be positive, got {self.param_dim}")

    # ------------------------------------------------------------------
    # PRNG helper: accept numpy RandomState or int seed → JAX key
    # ------------------------------------------------------------------

    @staticmethod
    def _to_jax_key(rng) -> jax.Array:
        if isinstance(rng, np.random.RandomState):
            seed = int(rng.randint(0, 2**31))
        elif isinstance(rng, (int, np.integer)):
            seed = int(rng)
        else:
            # Assume already a JAX key array
            return rng
        return jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Training loop — used by MDNEstimator; MAFEstimator overrides fully
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
        patience: int | None = None,
        min_delta: float = 0.0,
        window_size: int = 1,
        resume_training: bool = False,
        loss_fn=None,
    ):
        """
        Adam training loop (JAX).

        Mirrors ``ConditionalDensityEstimator.train`` but uses
        ``jax.value_and_grad`` compiled with ``jax.jit``.
        """
        params   = np.asarray(params, dtype="f")
        features = np.asarray(features, dtype="f")

        if not self._dims_inferred:
            self._infer_dimensions(params, features)

        if params.shape[0] != features.shape[0]:
            raise ValueError("params and features must have the same number of rows.")

        finite   = (np.all(np.isfinite(params),   axis=1) &
                    np.all(np.isfinite(features), axis=1))
        params   = params[finite]
        features = features[finite]
        if params.shape[0] == 0:
            raise ValueError("All data points were non-finite.")

        rng_np = np.random.RandomState(seed)

        if self._emb is not None:
            self.feature_dim = self._emb.output_dim

        if resume_training and self.weights is not None:
            weights = {k: jnp.array(v) for k, v in self.weights.items()}
        else:
            weights = self._initialize_weights(seed)
            weights = {k: jnp.array(v) for k, v in weights.items()}
            if self._emb is not None:
                emb_w = {k: jnp.array(v)
                         for k, v in self._emb.init_weights(rng_np).items()}
                weights.update(emb_w)

        features_j = jnp.array(features)
        params_j   = jnp.array(params)

        self.loss_history = []
        m = {k: jnp.zeros_like(v) for k, v in weights.items()}
        v = {k: jnp.zeros_like(v) for k, v in weights.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_loss   = np.inf
        counter     = 0
        loss_window: list[float] = []

        # APT: if loss_fn is provided it replaces _wrapped_loss; cannot be jit-compiled
        # because it contains numpy-level atom sampling. Fall back to eager mode.
        if loss_fn is not None:
            loss_and_grad = jax.value_and_grad(loss_fn)
        else:
            loss_and_grad = jax.jit(jax.value_and_grad(self._wrapped_loss))

        iterator = trange(n_iter, desc="Training", disable=not verbose)

        for i in iterator:
            if batch_size is not None and batch_size < len(params):
                idx  = rng_np.choice(len(params), size=batch_size, replace=False)
                f_b  = features_j[idx]
                p_b  = params_j[idx]
            else:
                f_b, p_b = features_j, params_j

            loss_val, g = loss_and_grad(weights, f_b, p_b)
            loss_f      = float(loss_val)
            self.loss_history.append(loss_f)

            if not np.isfinite(loss_f):
                log.warning("Non-finite loss at iteration %d — stopping.", i)
                break

            loss_window.append(loss_f)
            if len(loss_window) >= window_size:
                avg = sum(loss_window) / len(loss_window)
                loss_window = []
                if avg < best_loss - min_delta:
                    best_loss, counter = avg, 0
                else:
                    counter += 1
                    if patience is not None and counter >= patience:
                        log.info("Early stopping at iteration %d.", i)
                        break

            if verbose:
                iterator.set_postfix(loss=f"{loss_f:.4f}")

            # Adam update (pure pytree ops — no in-place mutation)
            new_weights = {}
            for key in weights:
                if not np.all(np.isfinite(np.array(g[key]))):
                    log.warning("Non-finite gradient for '%s' at iter %d.", key, i)
                    self.weights = {k: np.array(v) for k, v in weights.items()}
                    return
                m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                v[key] = beta2 * v[key] + (1 - beta2) * g[key] ** 2
                m_hat  = m[key] / (1 - beta1 ** (i + 1))
                v_hat  = v[key] / (1 - beta2 ** (i + 1))
                new_weights[key] = weights[key] - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            weights = new_weights

        self.weights = {k: np.array(v) for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Serialisation (identical to base.py — weights are numpy on disk)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {f"w__{k}": np.array(v) for k, v in self.weights.items()}
        data["__param_dim"]   = np.array(self.param_dim)
        data["__feature_dim"] = np.array(self.feature_dim)
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str):
        d   = np.load(path, allow_pickle=False)
        obj = cls.__new__(cls)
        obj.weights      = {k[3:]: d[k] for k in d if k.startswith("w__")}
        obj.param_dim    = int(d["__param_dim"])
        obj.feature_dim  = int(d["__feature_dim"])
        obj._dims_inferred = True
        obj.loss_history = []
        obj._emb = None
        return obj
