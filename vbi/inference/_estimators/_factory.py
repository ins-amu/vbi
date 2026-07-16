"""
Backend-selecting factory functions for density-estimator architectures.

``MAFEstimator``/``MDNEstimator``/``NSFEstimator`` (numpy/autograd) and their
JAX counterparts (``JaxMAFEstimator``/...) expose identical constructor
fields - only the backend differs.  These factories pick the right class for
a given ``backend`` and construct it, so callers don't need to know which
module a backend's class lives in::

    from vbi.inference import MAF
    est = MAF(n_flows=5, hidden_units=64, backend="jax")
"""
from __future__ import annotations

from .base import ConditionalDensityEstimator


def _build(key: str, backend: str, kwargs: dict) -> ConditionalDensityEstimator:
    from .._backends import resolve_backend, get_estimator_map

    cls = get_estimator_map(resolve_backend(backend))[key]
    return cls(**kwargs)


def MAF(*, backend: str = "auto", **kwargs) -> ConditionalDensityEstimator:
    """Build a Masked Autoregressive Flow estimator on the given backend ('auto' | 'numpy' | 'jax')."""
    return _build("maf", backend, kwargs)


def MDN(*, backend: str = "auto", **kwargs) -> ConditionalDensityEstimator:
    """Build a Mixture Density Network estimator on the given backend ('auto' | 'numpy' | 'jax')."""
    return _build("mdn", backend, kwargs)


def NSF(*, backend: str = "auto", **kwargs) -> ConditionalDensityEstimator:
    """Build a Neural Spline Flow estimator on the given backend ('auto' | 'numpy' | 'jax')."""
    return _build("nsf", backend, kwargs)
