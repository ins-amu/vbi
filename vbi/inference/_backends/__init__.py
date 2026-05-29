"""Backend registry for vbi.inference estimators."""
import importlib.util
import os

AVAILABLE_BACKENDS = ["numpy"]

try:
    import numba  # noqa: F401
    AVAILABLE_BACKENDS.append("numba")
except ImportError:
    pass

# Detect JAX without importing it - import is deferred until first actual use
# so that set_jax_device() can still influence JAX_PLATFORMS.
_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
if _JAX_AVAILABLE:
    AVAILABLE_BACKENDS.append("jax")

# The JAX platform to use.  Initialised from the environment (if the user
# already set JAX_PLATFORMS) or defaults to "cpu" (safe on machines where
# CUDA is present but cuDNN is absent or version-mismatched).
_jax_device: str = os.environ.get("JAX_PLATFORMS", "cpu")


def set_jax_device(device: str) -> None:
    """
    Select the JAX compute device.

    Call this **before** the first JAX estimator is created.  Once JAX has
    been imported (which happens the first time you call ``SNPE.train()`` or
    ``get_estimator_map('jax')``) the platform is fixed and this call has no
    effect.

    Parameters
    ----------
    device : str
        ``'cpu'``  - default; safe on any machine.
        ``'gpu'`` / ``'cuda'`` - requires a CUDA-capable GPU with cuDNN.
        ``'tpu'``  - requires a TPU runtime.

    Examples
    --------
    >>> from vbi.inference import set_jax_device, SNPE
    >>> set_jax_device('gpu')          # must come before SNPE.train()
    >>> inf = SNPE(prior=prior, density_estimator='maf', backend='jax')
    """
    global _jax_device
    _jax_device = device
    os.environ["JAX_PLATFORMS"] = device


def _apply_jax_platform() -> None:
    """Set JAX_PLATFORMS right before the first real JAX import."""
    os.environ.setdefault("JAX_PLATFORMS", _jax_device)


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
        _apply_jax_platform()     # must run before 'import jax' in jax_ modules
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
