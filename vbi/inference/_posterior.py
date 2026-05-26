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

    def __init__(self, estimator, prior=None, default_x=None, sample_with: str = "direct"):
        self._estimator  = estimator
        self._prior      = prior
        self._default_x  = None if default_x is None else np.asarray(default_x)
        if sample_with not in ("direct", "rejection"):
            raise ValueError(f"sample_with={sample_with!r} not supported. Choose 'direct' or 'rejection'.")
        self._sample_with = sample_with

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
        reject_outside_prior: bool = False,
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
        reject_outside_prior : bool
            If True, discard samples whose prior log_prob is -inf (i.e. outside
            the prior support) and redraw until ``n`` valid samples are collected.
            Requires a prior to have been passed at construction time.
            Automatically enabled when ``sample_with='rejection'``.

        Returns
        -------
        ndarray  shape (n_samples, param_dim)
            When a single observation is given the leading n_conditions dim
            is squeezed out to match the sbi convention.
        """
        x   = self._resolve_x(x)
        n   = int(np.prod(sample_shape))
        rng = np.random.RandomState(seed)

        x_2d = np.atleast_2d(np.asarray(x, dtype="f"))

        use_rejection = reject_outside_prior or (self._sample_with == "rejection")
        if use_rejection:
            return self._sample_with_rejection(n, x_2d, rng)

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

    def sample_batched(
        self,
        sample_shape,
        x,
        max_sampling_batch_size: int = 10_000,
        seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Draw samples for a batch of observations.

        Parameters
        ----------
        sample_shape : tuple[int, ...]  e.g. (1000,)
        x : array (batch_size, feature_dim)
        max_sampling_batch_size : int
            Accepted for sbi compatibility; ignored (numpy loops directly).
        seed : int | None

        Returns
        -------
        ndarray  (batch_size, n_samples, param_dim)
        """
        x   = np.atleast_2d(np.asarray(x, dtype="f"))
        n   = int(np.prod(sample_shape))
        rng = np.random.RandomState(seed)

        raw = self._estimator.sample(x, n_samples=n, rng=rng)
        # raw: (batch_size, n_samples, param_dim)
        return np.array(raw)

    def log_prob_batched(
        self,
        theta,
        x,
        leading_is_sample: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Log prob for matched batches of theta and x.

        Both theta and x share the same leading batch dimension; row i uses
        ``(theta[i], x[i])``.  This is the natural use-case for SBC and
        coverage tests.

        Parameters
        ----------
        theta : array (batch_size, param_dim)
        x     : array (batch_size, feature_dim)
        leading_is_sample : bool
            Accepted for sbi compatibility; ignored.

        Returns
        -------
        ndarray  (batch_size,)
        """
        return self.log_prob(theta, x=x, **kwargs)

    def leakage_correction(
        self,
        x=None,
        num_rejection_samples: int = 10_000,
        seed: int | None = None,
    ) -> float:
        """
        Estimate the fraction of posterior mass inside the prior support.

        A value of 1.0 means no leakage; 0.5 means half the flow's mass falls
        outside the prior.

        Parameters
        ----------
        x : array-like | None
            Observation. Falls back to default_x.
        num_rejection_samples : int
            Direct samples used to estimate acceptance fraction.

        Returns
        -------
        float   acceptance_fraction ∈ (0, 1]
        """
        if self._prior is None:
            raise ValueError("leakage_correction requires a prior.")
        x    = self._resolve_x(x)
        rng  = np.random.RandomState(seed)
        x_2d = np.atleast_2d(np.asarray(x, dtype="f"))
        raw  = self._estimator.sample(x_2d, n_samples=num_rejection_samples, rng=rng)
        if raw.shape[0] == 1:
            raw = raw[0]
        log_p             = self._prior.log_prob(raw.astype(np.float64))
        acceptance_frac   = float(np.mean(np.isfinite(log_p)))
        return acceptance_frac

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_with_rejection(
        self, n: int, x_2d: np.ndarray, rng
    ) -> np.ndarray:
        """Oversample-and-filter rejection sampling against prior support."""
        if self._prior is None:
            raise ValueError(
                "reject_outside_prior=True requires a prior. "
                "Pass prior= to SNPE() or Posterior()."
            )
        collected: list[np.ndarray] = []
        total_accepted = 0
        total_drawn    = 0
        oversample     = 4

        while total_accepted < n:
            n_draw = max((n - total_accepted) * oversample, 128)
            raw    = self._estimator.sample(x_2d, n_samples=int(n_draw), rng=rng)
            if raw.shape[0] == 1:
                raw = raw[0]  # (n_draw, param_dim)
            log_p  = self._prior.log_prob(raw.astype(np.float64))
            inside = raw[np.isfinite(log_p)]
            collected.append(inside)
            total_accepted += len(inside)
            total_drawn    += int(n_draw)

            if total_drawn > n * 10_000:
                raise RuntimeError(
                    f"Rejection sampling failed: drew {total_drawn} samples but "
                    f"only accepted {total_accepted} / {n} within the prior support. "
                    "Check that the prior and estimator are compatible, or use "
                    "sample_with='direct'."
                )
            if total_accepted > 0:
                accept_rate = total_accepted / total_drawn
                oversample  = max(4, int(2.0 / max(accept_rate, 1e-6)))

        return np.concatenate(collected, axis=0)[:n]

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
