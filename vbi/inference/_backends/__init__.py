"""Backend registry for vbi.inference estimators."""

AVAILABLE_BACKENDS = ["numpy"]

try:
    import numba  # noqa: F401
    AVAILABLE_BACKENDS.append("numba")
except ImportError:
    pass

_JAX_AVAILABLE = False
try:
    import jax  # noqa: F401
    _JAX_AVAILABLE = True
    AVAILABLE_BACKENDS.append("jax")
except ImportError:
    pass


def resolve_backend(requested: str) -> str:
    """Return the concrete backend name for ``requested``."""
    if requested == "auto":
        return "jax" if _JAX_AVAILABLE else "numpy"
    if requested not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Backend {requested!r} not available. "
            f"Available: {AVAILABLE_BACKENDS}. "
            f"Install the required package (e.g. pip install jax)."
        )
    return requested


def get_estimator_map(backend: str) -> dict:
    """
    Return the density-estimator class map for the given backend.

    Keys are the sbi-compatible density_estimator strings
    ('maf', 'mdn', 'nsf').  Values are the corresponding classes.
    """
    if backend == "jax":
        from .jax_ import JaxMAFEstimator, JaxMDNEstimator, JaxNSFEstimator
        return {
            "maf": JaxMAFEstimator,
            "mdn": JaxMDNEstimator,
            "nsf": JaxNSFEstimator,
        }

    # numpy (default) and any future numba backend fall through here
    from .._estimators import MAFEstimator, MDNEstimator, NSFEstimator
    return {
        "maf": MAFEstimator,
        "mdn": MDNEstimator,
        "nsf": NSFEstimator,
    }
