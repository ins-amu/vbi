# Backend registry — placeholder for numba_ and jax_ backends (MI-numba, MI1)
AVAILABLE_BACKENDS = ["numpy"]

try:
    import numba  # noqa: F401
    AVAILABLE_BACKENDS.append("numba")
except ImportError:
    pass

try:
    import jax  # noqa: F401
    AVAILABLE_BACKENDS.append("jax")
except ImportError:
    pass


def resolve_backend(requested: str) -> str:
    """Return the best available backend matching the request."""
    if requested == "auto":
        return "jax" if "jax" in AVAILABLE_BACKENDS else "numpy"
    if requested not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Backend {requested!r} not available.  "
            f"Available: {AVAILABLE_BACKENDS}.  "
            f"Install the required package (e.g. pip install jax)."
        )
    return requested
