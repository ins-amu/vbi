"""
Compatibility module for backward compatibility.

This module is deprecated. Please use 'vbi.sbi_inference' instead.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "Importing from 'vbi.inference' is deprecated and will be removed in a future version. "
    "Please use 'from vbi.sbi_inference import Inference' instead. "
    "The old import path 'from vbi.inference import Inference' will continue to work "
    "for now but is discouraged.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location for backward compatibility
try:
    from .sbi_inference import Inference
except ImportError as e:
    # If sbi_inference import fails, provide the same error message as before
    raise ImportError(
        "Inference functionality requires additional dependencies. "
        "Install with: pip install vbi[inference]"
    ) from e

# Re-export everything for backward compatibility
__all__ = ['Inference']
