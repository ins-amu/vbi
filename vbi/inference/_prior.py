"""
Prior distributions for SBI — torch-free, numpy-native.

All priors expose the same interface as sbi's prior objects:
    prior.sample((n,))          → (n, d) ndarray
    prior.log_prob(theta)       → (n,)   ndarray
    prior.dim                   → int
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def _resolve_param_names(param_names, dim: int) -> list[str]:
    if param_names is not None:
        names = list(param_names)
        if len(names) != dim:
            raise ValueError(
                f"param_names has {len(names)} entries but prior has dim={dim}."
            )
        return names
    return [f"p{i}" for i in range(dim)]


class BoxUniform:
    """
    Independent uniform distribution on a hyperrectangle.

    Equivalent to ``sbi.utils.BoxUniform`` but accepts numpy arrays
    instead of torch tensors.

    Parameters
    ----------
    low  : array-like  shape (d,)
    high : array-like  shape (d,)

    Examples
    --------
    >>> prior = BoxUniform(low=np.array([0., -5.]), high=np.array([2., 0.]))
    >>> prior.sample((1000,)).shape
    (1000, 2)
    """

    def __init__(self, low, high, param_names=None):
        self.low  = np.asarray(low,  dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same shape.")
        if np.any(self.low >= self.high):
            raise ValueError("low must be strictly less than high for every dimension.")
        self._log_volume = np.sum(np.log(self.high - self.low))
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def event_shape(self):
        return self.low.shape

    @property
    def dim(self) -> int:
        return int(self.low.size)

    @property
    def _resolved_param_names(self) -> list[str]:
        return _resolve_param_names(self.param_names, self.dim)

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        """
        Draw i.i.d. samples.

        Parameters
        ----------
        sample_shape : tuple[int, ...]  e.g. (1000,)
        seed : int | None

        Returns
        -------
        ndarray  shape (*sample_shape, d)
        """
        rng  = np.random.default_rng(seed)
        n    = int(np.prod(sample_shape))
        u    = rng.uniform(size=(n, self.dim))
        samp = u * (self.high - self.low) + self.low
        return samp.reshape(*sample_shape, self.dim)

    def log_prob(self, theta) -> np.ndarray:
        """
        Log probability: 0 inside the box, -inf outside.

        Parameters
        ----------
        theta : array (n, d) or (d,)

        Returns
        -------
        ndarray  (n,)
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        inside = np.all((theta >= self.low) & (theta <= self.high), axis=1)
        lp     = np.where(inside, -self._log_volume, -np.inf)
        return lp

    def __repr__(self):
        return f"BoxUniform(low={self.low}, high={self.high})"


class Gaussian:
    """
    Independent (diagonal-covariance) Gaussian prior.

    Parameters
    ----------
    mean : array-like  shape (d,)
    std  : array-like  shape (d,)  — standard deviations (not variances)
    """

    def __init__(self, mean, std, param_names=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std  = np.asarray(std,  dtype=np.float64)
        if np.any(self.std <= 0):
            raise ValueError("std must be strictly positive.")
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    @property
    def _resolved_param_names(self) -> list[str]:
        return _resolve_param_names(self.param_names, self.dim)

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        n    = int(np.prod(sample_shape))
        samp = rng.normal(loc=self.mean, scale=self.std, size=(n, self.dim))
        return samp.reshape(*sample_shape, self.dim)

    def log_prob(self, theta) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        z     = (theta - self.mean) / self.std
        lp    = -0.5 * np.sum(z ** 2, axis=1) - np.sum(np.log(self.std)) \
                - 0.5 * self.dim * np.log(2 * np.pi)
        return lp

    def __repr__(self):
        return f"Gaussian(mean={self.mean}, std={self.std})"


class CustomPrior:
    """
    User-defined prior wrapping arbitrary sample and log_prob functions.

    Parameters
    ----------
    sample_fn   : callable  (sample_shape) → ndarray
    log_prob_fn : callable  (theta: ndarray) → ndarray
    dim         : int  dimensionality of the parameter space
    """

    def __init__(self, sample_fn, log_prob_fn, dim: int, param_names=None):
        self._sample_fn   = sample_fn
        self._log_prob_fn = log_prob_fn
        self._dim         = dim
        self.param_names  = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def _resolved_param_names(self) -> list[str]:
        return _resolve_param_names(self.param_names, self.dim)

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        return np.asarray(self._sample_fn(sample_shape))

    def log_prob(self, theta) -> np.ndarray:
        return np.asarray(self._log_prob_fn(theta))

    def __repr__(self):
        return f"CustomPrior(dim={self._dim})"


# ---------------------------------------------------------------------------
# MultivariateNormal — full covariance Gaussian
# ---------------------------------------------------------------------------

class MultivariateNormal:
    """
    Multivariate Gaussian with a full (dense) covariance matrix.

    Parameters
    ----------
    mean : array-like  (d,)
    cov  : array-like  (d, d)  — positive-definite covariance matrix
    """

    def __init__(self, mean, cov, param_names=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov  = np.asarray(cov,  dtype=np.float64)
        if self.mean.ndim != 1:
            raise ValueError("mean must be 1-D.")
        d = self.mean.size
        if self.cov.shape != (d, d):
            raise ValueError(f"cov must be ({d}, {d}).")
        self._L       = np.linalg.cholesky(self.cov)
        self._L_inv   = np.linalg.inv(self._L)
        self._log_det = 2.0 * np.sum(np.log(np.diag(self._L)))
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    @property
    def _resolved_param_names(self) -> list[str]:
        return _resolve_param_names(self.param_names, self.dim)

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        n    = int(np.prod(sample_shape))
        z    = rng.standard_normal((n, self.dim))
        samp = z @ self._L.T + self.mean
        return samp.reshape(*sample_shape, self.dim)

    def log_prob(self, theta) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        diff  = theta - self.mean                         # (n, d)
        z     = diff @ self._L_inv.T                      # (n, d): L_inv @ diff_i per row
        quad  = np.sum(z ** 2, axis=1)                    # (n,)
        return -0.5 * (self.dim * np.log(2.0 * np.pi) + self._log_det + quad)

    def __repr__(self):
        return f"MultivariateNormal(mean={self.mean}, cov=...)"


# ---------------------------------------------------------------------------
# LogNormal — log-normal marginals (independent per dimension)
# ---------------------------------------------------------------------------

class LogNormal:
    """
    Independent log-normal distribution per parameter dimension.

    X ~ LogNormal(mean, std)  ⟺  log(X) ~ N(mean, std²)

    Parameters
    ----------
    mean : array-like  (d,)  — mean of log(X)
    std  : array-like  (d,)  — std of log(X)  (must be > 0)
    """

    def __init__(self, mean, std, param_names=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std  = np.asarray(std,  dtype=np.float64)
        if np.any(self.std <= 0):
            raise ValueError("std must be strictly positive.")
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    @property
    def _resolved_param_names(self) -> list[str]:
        return _resolve_param_names(self.param_names, self.dim)

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        n    = int(np.prod(sample_shape))
        samp = rng.lognormal(mean=self.mean, sigma=self.std, size=(n, self.dim))
        return samp.reshape(*sample_shape, self.dim)

    def log_prob(self, theta) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        # -inf for non-positive values
        with np.errstate(divide="ignore", invalid="ignore"):
            log_x  = np.where(theta > 0, np.log(np.maximum(theta, 1e-300)), np.nan)
            z      = (log_x - self.mean) / self.std
            lp     = np.where(
                np.all(theta > 0, axis=1, keepdims=True),
                -0.5 * z ** 2 - np.log(self.std) - log_x - 0.5 * np.log(2.0 * np.pi),
                -np.inf,
            )
        return np.sum(lp, axis=1)

    def __repr__(self):
        return f"LogNormal(mean={self.mean}, std={self.std})"


# ---------------------------------------------------------------------------
# Gamma — positive-valued parameters (independent per dimension)
# ---------------------------------------------------------------------------

class Gamma:
    """
    Independent Gamma distribution per parameter dimension.

    Parameters
    ----------
    concentration : array-like  (d,)  — shape parameter α > 0
    rate          : array-like  (d,)  — rate parameter β > 0  (scale = 1/rate)
    """

    def __init__(self, concentration, rate, param_names=None):
        self.concentration = np.asarray(concentration, dtype=np.float64)
        self.rate          = np.asarray(rate,          dtype=np.float64)
        if np.any(self.concentration <= 0):
            raise ValueError("concentration must be strictly positive.")
        if np.any(self.rate <= 0):
            raise ValueError("rate must be strictly positive.")
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return int(self.concentration.size)

    @property
    def _resolved_param_names(self) -> list[str]:
        return _resolve_param_names(self.param_names, self.dim)

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        n    = int(np.prod(sample_shape))
        samp = rng.gamma(shape=self.concentration, scale=1.0 / self.rate, size=(n, self.dim))
        return samp.reshape(*sample_shape, self.dim)

    def log_prob(self, theta) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        a, b  = self.concentration, self.rate
        with np.errstate(divide="ignore", invalid="ignore"):
            per_dim = np.where(
                theta > 0,
                (a - 1.0) * np.log(np.maximum(theta, 1e-300))
                - b * theta
                + a * np.log(b)
                - gammaln(a),
                -np.inf,
            )
        return np.sum(per_dim, axis=1)

    def __repr__(self):
        return f"Gamma(concentration={self.concentration}, rate={self.rate})"


# ---------------------------------------------------------------------------
# Beta — [0, 1]-valued parameters (independent per dimension)
# ---------------------------------------------------------------------------

class Beta:
    """
    Independent Beta distribution per parameter dimension.

    Parameters
    ----------
    alpha : array-like  (d,)  — α > 0
    beta  : array-like  (d,)  — β > 0
    """

    def __init__(self, alpha, beta, param_names=None):
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.beta  = np.asarray(beta,  dtype=np.float64)
        if np.any(self.alpha <= 0) or np.any(self.beta <= 0):
            raise ValueError("alpha and beta must be strictly positive.")
        self._log_norm = gammaln(self.alpha) + gammaln(self.beta) - gammaln(self.alpha + self.beta)
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return int(self.alpha.size)

    @property
    def _resolved_param_names(self) -> list[str]:
        return _resolve_param_names(self.param_names, self.dim)

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        n    = int(np.prod(sample_shape))
        samp = rng.beta(a=self.alpha, b=self.beta, size=(n, self.dim))
        return samp.reshape(*sample_shape, self.dim)

    def log_prob(self, theta) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        a, b  = self.alpha, self.beta
        in_support = np.all((theta > 0) & (theta < 1), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_safe = np.clip(theta, 1e-300, 1.0 - 1e-300)
            per_dim = (
                (a - 1.0) * np.log(t_safe)
                + (b - 1.0) * np.log(1.0 - t_safe)
                - self._log_norm
            )
        lp = np.sum(per_dim, axis=1)
        return np.where(in_support, lp, -np.inf)

    def __repr__(self):
        return f"Beta(alpha={self.alpha}, beta={self.beta})"


# ---------------------------------------------------------------------------
# MultipleIndependent — product of independent priors (mixed types)
# ---------------------------------------------------------------------------

class MultipleIndependent:
    """
    Product of independent prior distributions, one per parameter block.

    Useful for mixing prior types across parameter dimensions, e.g. a
    uniform parameter alongside a log-normal one.

    Parameters
    ----------
    priors : list of prior objects
        Each must expose ``.sample``, ``.log_prob``, and ``.dim``.

    Examples
    --------
    >>> prior = MultipleIndependent([
    ...     BoxUniform(low=np.array([0.]), high=np.array([1.])),
    ...     Gamma(concentration=np.array([2.]), rate=np.array([1.])),
    ... ])
    >>> prior.dim   # 2
    """

    def __init__(self, priors, param_names=None):
        self._priors = list(priors)
        if not self._priors:
            raise ValueError("priors list must not be empty.")
        self._dims      = [p.dim for p in self._priors]
        self._total_dim = sum(self._dims)
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return self._total_dim

    @property
    def _resolved_param_names(self) -> list[str]:
        if self.param_names is not None:
            return _resolve_param_names(self.param_names, self.dim)
        names = []
        for p in self._priors:
            names.extend(p._resolved_param_names)
        return names

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        n    = int(np.prod(sample_shape))
        parts = [
            p.sample((n,), seed=int(rng.integers(0, 2 ** 31)))
            for p in self._priors
        ]
        samp = np.concatenate(parts, axis=1)
        return samp.reshape(*sample_shape, self._total_dim)

    def log_prob(self, theta) -> np.ndarray:
        theta  = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lp     = np.zeros(theta.shape[0])
        offset = 0
        for p, d in zip(self._priors, self._dims):
            lp    += p.log_prob(theta[:, offset: offset + d])
            offset += d
        return lp

    def __repr__(self):
        names = ", ".join(type(p).__name__ for p in self._priors)
        return f"MultipleIndependent([{names}])"


# ---------------------------------------------------------------------------
# RestrictedPrior — truncated prior via constraint function
# ---------------------------------------------------------------------------

class RestrictedPrior:
    """
    Truncated prior: samples from ``base_prior`` that satisfy ``constraint_fn``
    are kept; the rest are discarded via rejection sampling.

    Parameters
    ----------
    base_prior    : any prior object
    constraint_fn : callable  (theta: ndarray (n, d)) → bool array (n,)
                    True = accept, False = reject.

    Examples
    --------
    >>> # Restrict a 2-D Gaussian to the unit ball
    >>> prior = RestrictedPrior(
    ...     Gaussian(mean=np.zeros(2), std=np.ones(2)),
    ...     constraint_fn=lambda t: np.sum(t ** 2, axis=1) <= 1.0,
    ... )
    """

    def __init__(self, base_prior, constraint_fn, param_names=None):
        self._base       = base_prior
        self._constraint = constraint_fn
        self.param_names = param_names
        if param_names is not None:
            _resolve_param_names(param_names, self.dim)

    @property
    def dim(self) -> int:
        return self._base.dim

    @property
    def _resolved_param_names(self) -> list[str]:
        if self.param_names is not None:
            return _resolve_param_names(self.param_names, self.dim)
        return self._base._resolved_param_names

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        rng        = np.random.default_rng(seed)
        n          = int(np.prod(sample_shape))
        collected: list[np.ndarray] = []
        accepted   = 0
        drawn      = 0
        oversample = 4

        while accepted < n:
            n_draw = max((n - accepted) * oversample, 32)
            cands  = self._base.sample(
                (int(n_draw),), seed=int(rng.integers(0, 2 ** 31))
            )
            mask   = np.asarray(self._constraint(cands), dtype=bool)
            ok     = cands[mask]
            collected.append(ok)
            accepted += len(ok)
            drawn    += int(n_draw)

            if drawn > n * 100_000:
                raise RuntimeError(
                    f"RestrictedPrior: acceptance rate is too low — drew {drawn} "
                    f"samples but only accepted {accepted} / {n}."
                )
            if accepted > 0:
                oversample = max(4, int(2.0 / max(accepted / drawn, 1e-9)))

        samp = np.concatenate(collected, axis=0)[:n]
        return samp.reshape(*sample_shape, self.dim)

    def log_prob(self, theta) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lp    = self._base.log_prob(theta)
        mask  = np.asarray(self._constraint(theta), dtype=bool)
        return np.where(mask, lp, -np.inf)

    def __repr__(self):
        return f"RestrictedPrior(base={self._base!r})"
