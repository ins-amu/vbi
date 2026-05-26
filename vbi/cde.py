"""
vbi.cde is deprecated, use vbi.inference instead.

This module is kept for backward compatibility.  All classes have been moved
to ``vbi.inference`` with an sbi-compatible API.

Migration::

    # Old
    from vbi.cde import MDNEstimator, MAFEstimator

    # New (low-level, same classes)
    from vbi.inference import MDNEstimator, MAFEstimator

    # New (sbi-compatible high-level API — recommended)
    from vbi.inference import SNPE, BoxUniform
"""
import warnings

warnings.warn(
    "vbi.cde is deprecated and will be removed in a future release.  "
    "Use 'from vbi.inference import ...' instead.  "
    "The sbi-compatible SNPE interface is available at vbi.inference.SNPE.",
    DeprecationWarning,
    stacklevel=2,
)

from vbi.inference._estimators.base import ConditionalDensityEstimator
from vbi.inference._estimators.mdn  import MDNEstimator
from vbi.inference._estimators.maf  import MAFEstimator, MAFEstimator0

__all__ = [
    "ConditionalDensityEstimator",
    "MDNEstimator",
    "MAFEstimator",
    "MAFEstimator0",
]
