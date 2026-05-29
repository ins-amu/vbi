"""JAX computation backend for vbi.inference estimators."""
from .mdn_jax import JaxMDNEstimator
from .maf_jax import JaxMAFEstimator
from .nsf_jax import JaxNSFEstimator

__all__ = ["JaxMDNEstimator", "JaxMAFEstimator", "JaxNSFEstimator"]
