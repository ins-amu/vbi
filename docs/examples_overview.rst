.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/custom.css">

Examples & Tutorials
====================

This section provides comprehensive examples and tutorials for using VBI across different computational backends and inference methods.

Simulation Examples
-------------------

Learn how to simulate brain models using different computational backends:

**Getting Started**
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/intro
   examples/intro_feature
   examples/distributions

**NumPy/Numba Backend (CPU)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CPU-based simulations using numba for acceleration with SBI inference:

.. toctree::
   :maxdepth: 1

   examples/damp_oscillator
   examples/jansen_rit_sde_numba
   examples/sl_sdde_numba
   examples/vep_sde_numba
   examples/mpr_sde_numba
   examples/rww_sde_numba
   examples/wilson_cowan_sde_numba
   examples/ww_full_sde_numba

**NumPy/Numba Backend with CDE (CPU)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CPU-based simulations using numba for acceleration with Conditional Density Estimation:

.. toctree::
   :maxdepth: 1

   examples/jansen_rit_sde_numba_cde
   examples/vep_sde_numba_cde
   examples/vep_sde_numba_cde_84
   examples/mpr_sde_numba_cde
   examples/wilson_cowan_sde_numba_cde
   examples/ww_full_sde_numba_cde

**CuPy Backend (GPU)**
^^^^^^^^^^^^^^^^^^^^^^

GPU-accelerated simulations using CuPy:

.. toctree::
   :maxdepth: 1

   examples/mpr_sde_cupy
   examples/jansen_rit_sde_cupy
   examples/ghb_sde_cupy
   examples/wilson_cowan_cupy
   examples/ww_full_sde_cupy

**PyTorch Backend (GPU/CPU)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch-based simulations with automatic differentiation:

.. toctree::
   :maxdepth: 1

   examples/rww_sde_pytorch

**C++ Backend (High Performance)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High-performance simulations using C++ backend with SBI and CDE inference:

.. toctree::
   :maxdepth: 1

   examples/damp_oscillator_cde
   examples/vep_sde
   examples/mpr_sde_cpp
   examples/jansen_rit_sde_cpp

**Specialized Examples**
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/mpr_tvbk
   examples/rww_sde_torch_kong

Quick Start Guide
-----------------

1. **Begin with** :doc:`examples/intro` - Basic VBI functionality
2. **Learn feature extraction** with :doc:`examples/intro_feature`
3. **Choose your backend**:
   
   - For **CPU**: Try numba examples (fast, no GPU required)
   - For **GPU**: Try CuPy examples (fastest, requires CUDA)
   - For **high performance**: Try C++ examples (optimized, good for production)

4. **Move to inference**: See :doc:`inference_examples` for parameter estimation

Computational Backend Comparison
---------------------------------

.. list-table:: Backend Performance Comparison
   :header-rows: 1
   :class: color-caption

   * - **Backend**
     - **Speed**
     - **Requirements**
     - **Best For**
   * - NumPy/Numba
     - Fast
     - CPU only
     - Development, prototyping
   * - CuPy
     - Fastest
     - CUDA GPU
     - Large simulations, production
   * - C++
     - Very Fast
     - C++ compiler
     - High performance, deployment

Installation Notes
------------------

- **Light version** (``pip install vbi``): Includes numba examples
- **Light + GPU version** (``pip install vbi[light-gpu]``): Includes CuPy support
- **Full version** (``pip install vbi[inference]``): Includes all backends
- **GPU version** (``pip install vbi[inference-gpu]``): Includes CuPy support
