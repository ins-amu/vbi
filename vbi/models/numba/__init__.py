"""Numba JIT-compiled brain-model backend for VBI.

Exports the most commonly used Numba-based simulation classes.
All models support CPU execution; GPU acceleration is not available
in this backend (use ``vbi.models.cupy`` for GPU).
"""
from .mpr import MPR_sde
from .ww import WW_sde

__all__ = ["MPR_sde", "WW_sde"]