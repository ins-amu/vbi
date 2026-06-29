"""Brain-model sub-package for VBI.

Re-exports commonly used model classes from the Numba and CuPy backends.
Available backends:

- ``vbi.models.numba``  — Numba JIT-compiled CPU models
- ``vbi.models.cupy``   — GPU-accelerated models (requires CuPy)
- ``vbi.models.cpp``    — C++/SWIG models
- ``vbi.models.jax``    — JAX models
- ``vbi.models.pytorch``— PyTorch models
"""
