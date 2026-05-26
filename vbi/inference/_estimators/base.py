"""
Abstract base class for conditional density estimators.

Training uses the Adam optimizer with optional mini-batch sampling,
early stopping on a held-out validation set, and gradient clipping.
"""
import abc
import math
import logging

import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp
from tqdm.auto import trange

log = logging.getLogger(__name__)


class ConditionalDensityEstimator(abc.ABC):
    """
    Abstract base class for conditional density estimators.

    Provides a unified training loop (Adam + mini-batch + early stopping)
    and a standardised API for ``train``, ``sample``, and ``log_prob``.

    Parameters
    ----------
    param_dim : int | None
        Dimensionality of the target (parameters). Inferred from data if None.
    feature_dim : int | None
        Dimensionality of the condition (features). Inferred from data if None.
    """

    def __init__(self, param_dim: int = None, feature_dim: int = None):
        self.param_dim = param_dim
        self.feature_dim = feature_dim
        self._dims_inferred = False
        self.weights = None
        self.loss_history: list[float] = []
        self._emb = None  # optional EmbeddingNet, set via set_embedding()

    # ------------------------------------------------------------------
    # Embedding support
    # ------------------------------------------------------------------

    def set_embedding(self, emb) -> None:
        """Attach a learned EmbeddingNet trained jointly with this estimator."""
        self._emb = emb

    def _wrapped_loss(self, weights, features, params) -> float:
        """Apply embedding (if any) then delegate to _loss_function."""
        if self._emb is not None:
            features = self._emb.forward(weights, features)
        return self._loss_function(weights, features, params)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _initialize_weights(self, rng) -> dict:
        pass

    @abc.abstractmethod
    def _loss_function(self, weights: dict, features, params) -> float:
        pass

    @abc.abstractmethod
    def sample(self, features, n_samples: int, rng) -> anp.ndarray:
        if self.weights is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

    @abc.abstractmethod
    def log_prob(self, features, params) -> anp.ndarray:
        if self.weights is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

    # ------------------------------------------------------------------
    # Dimension inference
    # ------------------------------------------------------------------

    def _infer_dimensions(self, params, features):
        params   = anp.asarray(params)
        features = anp.asarray(features)
        if params.ndim   == 1: params   = params.reshape(-1, 1)
        if features.ndim == 1: features = features.reshape(-1, 1)

        self.param_dim   = params.shape[1]
        self.feature_dim = features.shape[1]
        self._dims_inferred = True

        log.debug("Inferred dimensions: param_dim=%d  feature_dim=%d",
                  self.param_dim, self.feature_dim)

        if self.param_dim <= 0:
            raise ValueError(f"param_dim must be positive, got {self.param_dim}")

    # ------------------------------------------------------------------
    # Training loop (used by MDNEstimator; MAFEstimator overrides fully)
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
    ):
        """
        Train with Adam.  Mini-batch training is enabled when ``batch_size``
        is set and smaller than the training set size.

        Parameters
        ----------
        params : array (N, param_dim)
        features : array (N, feature_dim)
        n_iter : int
            Maximum number of gradient steps.
        learning_rate : float
        seed : int
        batch_size : int | None
            Mini-batch size.  None → full-batch (original behaviour).
        verbose : bool
            Show tqdm progress bar.
        patience : int | None
            Early-stopping patience counted in windows (epochs).  None → disabled.
        min_delta : float
            Minimum loss improvement to count as progress.
        window_size : int
            Number of consecutive gradient steps averaged together before
            checking patience.  Set to ceil(N / batch_size) so that one window
            = one epoch and patience counts epochs, not raw steps.
        """
        params   = anp.asarray(params)
        features = anp.asarray(features)

        if not self._dims_inferred:
            self._infer_dimensions(params, features)

        if params.shape[0] != features.shape[0]:
            raise ValueError("params and features must have the same number of rows.")

        finite = (anp.all(anp.isfinite(params),   axis=1) &
                  anp.all(anp.isfinite(features), axis=1))
        params   = params[finite].astype("f")
        features = features[finite].astype("f")
        if params.shape[0] == 0:
            raise ValueError("All data points were non-finite.")

        rng = anp.random.RandomState(seed)

        if self._emb is not None:
            self.feature_dim = self._emb.output_dim

        if resume_training and self.weights is not None:
            pass  # warm start: keep existing weights
        else:
            self.weights = self._initialize_weights(rng)
            if self._emb is not None:
                self.weights.update(self._emb.init_weights(rng))

        self.loss_history = []

        m = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        v = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_loss    = anp.inf
        counter      = 0
        loss_window: list[float] = []

        gradient_func = grad(self._wrapped_loss)
        iterator = trange(n_iter, desc="Training", disable=not verbose)

        for i in iterator:
            # Mini-batch selection
            if batch_size is not None and batch_size < len(params):
                idx = rng.choice(len(params), size=batch_size, replace=False)
                f_batch, p_batch = features[idx], params[idx]
            else:
                f_batch, p_batch = features, params

            g    = gradient_func(self.weights, f_batch, p_batch)
            loss = self._wrapped_loss(self.weights, f_batch, p_batch)
            self.loss_history.append(float(loss))

            if not anp.isfinite(loss):
                log.warning("Non-finite loss at iteration %d — stopping.", i)
                break

            # Accumulate into rolling window; check patience only at window boundaries
            loss_window.append(float(loss))
            if len(loss_window) >= window_size:
                avg_loss = sum(loss_window) / len(loss_window)
                loss_window = []
                if avg_loss < best_loss - min_delta:
                    best_loss, counter = avg_loss, 0
                else:
                    counter += 1
                    if patience is not None and counter >= patience:
                        log.info("Early stopping at iteration %d.", i)
                        break

            if verbose:
                iterator.set_postfix(loss=f"{loss:.4f}")

            # Adam update
            for key in self.weights:
                if not anp.all(anp.isfinite(g[key])):
                    log.warning("Non-finite gradient for '%s' at iter %d.", key, i)
                    return
                m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                v[key] = beta2 * v[key] + (1 - beta2) * g[key] ** 2
                m_hat  = m[key] / (1 - beta1 ** (i + 1))
                v_hat  = v[key] / (1 - beta2 ** (i + 1))
                self.weights[key] -= learning_rate * m_hat / (anp.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save weights and config to a .npz file."""
        import numpy as np
        data = {f"w__{k}": v for k, v in self.weights.items()}
        data["__param_dim"]   = np.array(self.param_dim)
        data["__feature_dim"] = np.array(self.feature_dim)
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str):
        """Load a previously saved estimator (weights only; architecture must match)."""
        import numpy as np
        d   = np.load(path, allow_pickle=False)
        obj = cls.__new__(cls)
        obj.weights      = {k[3:]: d[k] for k in d if k.startswith("w__")}
        obj.param_dim    = int(d["__param_dim"])
        obj.feature_dim  = int(d["__feature_dim"])
        obj._dims_inferred = True
        obj.loss_history = []
        return obj
