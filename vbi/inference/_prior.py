"""
Prior distributions for SBI — torch-free, numpy-native.

All priors expose the same interface as sbi's prior objects:
    prior.sample((n,))          → (n, d) ndarray
    prior.log_prob(theta)       → (n,)   ndarray
    prior.support               → {'low': array, 'high': array} or None
"""
from __future__ import annotations

import numpy as np


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

    def __init__(self, low, high):
        self.low  = np.asarray(low,  dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same shape.")
        if np.any(self.low >= self.high):
            raise ValueError("low must be strictly less than high for every dimension.")
        self._log_volume = np.sum(np.log(self.high - self.low))

    @property
    def event_shape(self):
        return self.low.shape

    @property
    def dim(self) -> int:
        return int(self.low.size)

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

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std  = np.asarray(std,  dtype=np.float64)
        if np.any(self.std <= 0):
            raise ValueError("std must be strictly positive.")

    @property
    def dim(self) -> int:
        return int(self.mean.size)

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

    def __init__(self, sample_fn, log_prob_fn, dim: int):
        self._sample_fn   = sample_fn
        self._log_prob_fn = log_prob_fn
        self._dim         = dim

    @property
    def dim(self) -> int:
        return self._dim

    def sample(self, sample_shape, seed: int | None = None) -> np.ndarray:
        return np.asarray(self._sample_fn(sample_shape))

    def log_prob(self, theta) -> np.ndarray:
        return np.asarray(self._log_prob_fn(theta))

    def __repr__(self):
        return f"CustomPrior(dim={self._dim})"
