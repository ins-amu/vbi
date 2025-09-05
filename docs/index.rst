.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/custom.css">


Virtual Brain Inference (VBI)
##############################


.. image:: _static/vbi_log.png
   :alt: VBI Logo
   :width: 200px
   :align: center


The **Virtual Brain Inference (VBI)** toolkit is an open-source, flexible solution tailored for probabilistic inference on virtual brain models. It integrates computational models with personalized anatomical data to deepen the understanding of brain dynamics and neurological processes. VBI supports **fast simulations**, comprehensive **feature extraction**, and employs **deep neural density estimators** to handle various neuroimaging data types. Its goal is to bridge the gap in solving the inverse problem of identifying control parameters that best explain observed data, thereby making these models applicable for clinical settings. VBI leverages high-performance computing through GPU acceleration and C++ code to ensure efficiency in processing.


Workflow
========

.. image:: _static/Fig1.png
   :alt: VBI Logo
   :width: 800px

Installation
============

**Prerequisites:**

First, create and activate a conda environment (Python 3.10+ recommended):

.. code-block:: bash

    conda env create --name vbi python=3.10
    conda activate vbi

**Installation Options:**

VBI offers flexible installation options tailored for different use cases and hardware configurations:

**Light Version - CPU Simulation Only**

For basic brain simulation with minimal dependencies (numba + C++):

.. code-block:: bash

    pip install vbi

*Includes:* Brain simulation models, feature extraction, visualization  
*Best for:* Users who only need simulation capabilities, minimal dependencies

**Light Version with GPU Acceleration**

Adds CuPy for GPU-accelerated simulations:

.. code-block:: bash

    pip install vbi[light-gpu]

*Includes:* Everything in light + CuPy for GPU-accelerated simulations  
*Best for:* GPU users who want fast simulations but don't need inference  
*Requirements:* NVIDIA GPU with CUDA support

**Parameter Inference (CPU)**

Adds PyTorch and SBI for Bayesian parameter inference:

.. code-block:: bash

    pip install vbi[inference]

*Includes:* Everything in light + PyTorch (CPU) + SBI for Bayesian inference  
*Best for:* Users who need parameter estimation but don't have GPU

**Parameter Inference with GPU**

Full functionality with GPU acceleration for both simulation and inference:

.. code-block:: bash

    pip install vbi[inference-gpu]

*Includes:* Full functionality with GPU acceleration for both simulation and inference  
*Best for:* GPU users who need both fast simulation and parameter inference  
*Requirements:* NVIDIA GPU with CUDA support

**Complete Installation**

All features including documentation tools and development dependencies:

.. code-block:: bash

    pip install vbi[all]

*Includes:* All above + documentation tools, development dependencies  
*Best for:* Developers, researchers who want all functionality

**Combining Options:**

You can combine multiple options as needed:

.. code-block:: bash

    # Inference + GPU + Development tools
    pip install vbi[inference-gpu,dev]
    
    # Light GPU + Documentation building
    pip install vbi[light-gpu,docs]

**Hardware-Specific Recommendations:**

**CPU-Only Systems:**

.. code-block:: bash

    pip install vbi[inference]        # For inference work
    pip install vbi                   # For simulation only

**Systems with NVIDIA GPU:**

.. code-block:: bash

    pip install vbi[inference-gpu]    # For GPU inference
    pip install vbi[light-gpu]        # For GPU simulation only

**Use Case Guide:**

- **Researchers doing parameter inference:** ``pip install vbi[inference-gpu]``
- **Students learning brain modeling:** ``pip install vbi``
- **HPC users with GPU clusters:** ``pip install vbi[light-gpu]``
- **Developers contributing to VBI:** ``pip install vbi[all,dev,docs]``
- **Classroom/workshop environments:** ``pip install vbi[inference]``

**From Source (Latest Development Version):**

.. code-block:: bash

    git clone https://github.com/ins-amu/vbi.git
    cd vbi
    pip install .

**For developers:**

.. code-block:: bash

    pip install -e .[all,dev,docs]

**Optional: Skip C++ Compilation**

If you encounter compilation issues, you can skip C++ components during installation:

.. code-block:: bash

    SKIP_CPP=1 pip install -e . 

Using Docker
============

To use the Docker image, you can pull it from the GitHub Container Registry and run it as follows:

.. code-block:: bash

   
    # Get it without building anything locally
    # without GPU
    docker run --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main

    # with GPU
    docker run --gpus all --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main

Building and Using Docker Locally
==================================

For local development and customization, you can build the VBI Docker image yourself:

**Quick Start:**

.. code-block:: bash

    # Build the optimized image
    docker build -t vbi:latest .
    
    # Start with convenience script
    ./run-vbi.sh start
    
    # Or start manually
    docker run --gpus all -p 8888:8888 vbi:latest

**Complete Guides:**

- :doc:`docker_build` - Comprehensive building guide with optimizations and troubleshooting
- :doc:`docker_quickstart` - Quick reference for daily usage and container management

   

**Verify Installation:**

.. code-block:: python 

   import vbi 
   vbi.tests()
   vbi.test_imports()  

**Example output for full installation:**

.. code-block:: text

   Dependency Check              
                                           
   Package      Version       Status        
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
   vbi          v0.2.1        ✅ Available  
   numpy        1.24.4        ✅ Available  
   scipy        1.10.1        ✅ Available  
   matplotlib   3.7.5         ✅ Available  
   sbi          0.22.0        ✅ Available  
   torch        2.4.1+cu121   ✅ Available  
   cupy         12.3.0        ✅ Available  
                                            
   Torch GPU available: True
   Torch device count: 1
   Torch CUDA version: 12.1
   CuPy GPU available: True
   CuPy device count: 1

**Example output for light version:**

.. code-block:: text

   Dependency Check              
                                           
   Package      Version       Status        
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
   vbi          v0.2.1        ✅ Available  
   numpy        1.24.4        ✅ Available  
   scipy        1.10.1        ✅ Available  
   matplotlib   3.7.5         ✅ Available  
   sbi          -             ❌ Not Found  
   torch        -             ❌ Not Found  
   cupy         -             ❌ Not Found  

   Note: Missing packages are expected for light installation.
   Install vbi[inference] or vbi[inference-gpu] for additional functionality.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   models
   docker_build
   docker_quickstart


Examples
=========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples/intro
   examples/intro_feature
   examples/do_cpp
   examples/do_nb
   examples/vep_sde
   examples/mpr_sde_cupy
   examples/mpr_sde_numba
   examples/mpr_sde_cpp
   examples/mpr_tvbk
   examples/jansen_rit_sde_cpp
   examples/jansen_rit_sde_cupy
   examples/jansen_rit_sde_numba
   examples/ww_sde_torch_kong
   examples/ghb_sde_cupy
   examples/wilson_cowan_cupy
   examples/wilson_cowan_sde_numba
   examples/ww_full_sde_cupy
   examples/ww_full_sde_numba



.. toctree::
    :maxdepth: 2
    :caption: API Reference

    API


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



