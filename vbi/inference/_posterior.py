"""
Posterior object with sbi-compatible API.

``posterior.sample((n,), x=x_obs)`` and ``posterior.log_prob(theta, x=x_obs)``
match sbi's ``NeuralPosterior`` interface exactly (numpy arrays instead of tensors).
"""
from __future__ import annotations

import numpy as np


class Posterior:
    """
    Approximate posterior p(theta | x) wrapping a trained density estimator.

    The API mirrors ``sbi.inference.posteriors.NeuralPosterior``:

    Examples
    --------
    >>> posterior = inference.build_posterior(estimator)
    >>> samples   = posterior.sample((1000,), x=x_obs)
    >>> log_probs = posterior.log_prob(theta, x=x_obs)
    """

    def __init__(self, estimator, prior=None, default_x=None):
        self._estimator  = estimator
        self._prior      = prior
        self._default_x  = None if default_x is None else np.asarray(default_x)

    # ------------------------------------------------------------------
    # sbi-compatible public API
    # ------------------------------------------------------------------

    def set_default_x(self, x) -> "Posterior":
        """Store a default observation used when x is not supplied to sample/log_prob."""
        self._default_x = np.asarray(x)
        return self

    def sample(
        self,
        sample_shape,
        x=None,
        seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Draw samples from p(theta | x).

        Parameters
        ----------
        sample_shape : tuple[int, ...]  e.g. (1000,)
        x : array-like | None
            Observation to condition on.  Falls back to ``default_x``.
        seed : int | None

        Returns
        -------
        ndarray  shape (n_samples, param_dim)
            When a single observation is given the leading n_conditions dim
            is squeezed out to match the sbi convention.
        """
        x   = self._resolve_x(x)
        n   = int(np.prod(sample_shape))
        rng = np.random.RandomState(seed)

        x_2d    = np.atleast_2d(np.asarray(x, dtype="f"))
        samples = self._estimator.sample(x_2d, n_samples=n, rng=rng)
        # samples: (n_conditions, n_samples, param_dim)
        # Squeeze n_conditions=1 to match sbi's (n_samples, param_dim)
        if samples.shape[0] == 1:
            return np.array(samples[0])
        return np.array(samples)

    def log_prob(
        self,
        theta,
        x=None,
        norm_posterior: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Evaluate log p(theta | x).

        **Argument order matches sbi**: ``theta`` first, ``x`` as keyword.
        (This is the reverse of the low-level ``estimator.log_prob(features, params)``.)

        Parameters
        ----------
        theta : array (n, param_dim)
        x     : array | None
        norm_posterior : bool
            Ignored (kept for sbi compat); normalisation is implicit in the
            flow/mixture density.

        Returns
        -------
        ndarray  (n,)
        """
        x     = self._resolve_x(x)
        theta = np.atleast_2d(np.asarray(theta, dtype="f"))
        x_2d  = np.atleast_2d(np.asarray(x, dtype="f"))

        # Broadcast x to match theta count if needed
        if x_2d.shape[0] == 1 and theta.shape[0] > 1:
            x_2d = np.repeat(x_2d, theta.shape[0], axis=0)

        # Low-level API: log_prob(features, params) — note order swap here
        log_p = np.array(self._estimator.log_prob(x_2d, theta))

        if self._prior is not None:
            log_p = log_p + np.array(self._prior.log_prob(theta))

        return log_p

    def map(
        self,
        x=None,
        num_iter: int = 1000,
        learning_rate: float = 0.01,
        num_init_samples: int = 100,
        seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Maximum a posteriori estimate via gradient ascent on log_prob.

        Returns
        -------
        ndarray  (param_dim,)
        """
        from autograd import grad as _grad
        import autograd.numpy as anp

        x = self._resolve_x(x)

        # Initialise at the sample mean
        init = self.sample((num_init_samples,), x=x, seed=seed)
        theta = anp.mean(init, axis=0, keepdims=True).astype("f")

        x_2d = np.atleast_2d(np.asarray(x, dtype="f"))

        def neg_log_prob(th):
            x_rep = np.repeat(x_2d, th.shape[0], axis=0) if x_2d.shape[0] == 1 else x_2d
            return -anp.mean(self._estimator.log_prob(x_rep, th))

        grad_fn = _grad(neg_log_prob)
        for _ in range(num_iter):
            theta = theta - learning_rate * grad_fn(theta)

        return np.array(theta[0])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_x(self, x):
        if x is not None:
            return x
        if self._default_x is not None:
            return self._default_x
        raise ValueError(
            "No observation provided. Pass x= or call set_default_x(x_obs) first."
        )

    def __repr__(self):
        return (f"Posterior(estimator={type(self._estimator).__name__}, "
                f"prior={type(self._prior).__name__ if self._prior else None})")
