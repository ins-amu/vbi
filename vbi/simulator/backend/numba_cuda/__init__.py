"""
VBI Numba-CUDA backend.

Primary entry points (registered in api.py):
    CudaSimulator  — single-run (batch-size 1, wraps sweeper)
    CudaSweeperGPU — parallel sweep; each GPU thread runs one simulation

Design decisions vs TVB's nb_hybrid_cuda_sweep.py
--------------------------------------------------
* 1 GPU thread = 1 complete simulation   (same as TVB's simple kernel)
* float32 on GPU (4× memory vs float64)  (same as TVB)
* post-corrector bounds clamping only    (differs from TVB; see notes/tvb_heun_issue.md)
* generic dfun via @cuda.jit(device=True) code-gen (TVB inlines model-specific code)
* dense connectivity (CSR optimisation deferred to later milestone)

GPU memory layout
-----------------
state  : (n_samples, n_sv, n_nodes) float32
buf    : (n_samples, n_cvar, n_nodes, horizon) float32  — ring buffer
ts_out : (n_samples, n_record, n_sv, n_nodes) float32  — time series output
params : (n_samples, n_params) float32  — one row per sweep point

Prerequisites
-------------
    pip install numba[cuda]
    # CUDA toolkit must be installed; CUDA_HOME or PATH must include nvcc.
"""

try:
    from numba import cuda as _cuda
    CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

from .simulator import CudaSimulator
from .sweeper import CudaSweeperGPU
