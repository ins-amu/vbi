.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/custom.css">

Installation Guide
==================

Quick Start
-----------

Create conda environment (recommended):

.. code-block:: bash

    conda create --name vbi python=3.10
    conda activate vbi

Install VBI:

.. code-block:: bash

    pip install vbi                    # Light version (CPU only)
    pip install vbi[light-gpu]         # Light + Cupy
    pip install vbi[inference]         # With (sbi, PyTorch)
    pip install vbi[all]               # Full (sbi, PyTorch, Cupy)

Installation Options
--------------------

.. list-table:: VBI Installation Variants
   :header-rows: 1
   :class: color-caption

   * - **Command**
     - **Includes**
     - **Best For**
   * - ``pip install vbi``
     - CPU simulation, feature extraction, CDE-based inference
     - Avoiding heavy dependencies
   * - ``pip install vbi[light-gpu]``
     - Everything + Cupy
     - GPU simulation
   * - ``pip install vbi[inference]``
     - Everything + PyTorch, SBI
     - Parameter inference (CPU)
   * - ``pip install vbi[inference-gpu]``
     - Everything + GPU acceleration
     - Full functionality with GPU

Docker Usage
------------

.. code-block:: bash

    # Quick start with pre-built image
    docker run --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main

    # With GPU support
    docker run --gpus all --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main

For Docker building and advanced usage, see :doc:`docker_build` and :doc:`docker_quickstart`.

Installation From Source
-------------------------

.. code-block:: bash

    git clone https://github.com/ins-amu/vbi.git
    cd vbi
    pip install .

For development:

.. code-block:: bash

    pip install -e .[all]

Platform-Specific Instructions
-------------------------------

**Google Colab**

Google Colab doesn't have VBI or SBI pre-installed, and **Docker is not supported** in Colab due to security restrictions. For optimal C++ module compilation, install from source:

.. code-block:: bash

    # In a Colab cell, run:
    !mkdir -p src && cd src
    !git clone --depth 1 https://github.com/ins-amu/vbi.git
    %cd src/vbi
    !pip install -e .

**Alternative: Use Colab Pro+ with Custom Runtimes**

If you have Colab Pro+ and need a containerized environment, consider:

- Using **Kaggle Notebooks** (supports Docker-based custom environments)
- Using **Binder** with our repository (though with limited resources)
- Setting up a **local Jupyter server** with our Docker image and connecting via ngrok

**Note:** The environment will be reset when the Colab runtime shuts down. You'll need to reinstall for each new session.

**EBRAINS Collab**

EBRAINS has dependency management restrictions. Here's a script to create a dedicated VBI environment:

.. code-block:: bash

    #!/bin/bash
    # Save this as setup_vbi_ebrains.sh

    set -eux

    # Create fresh environment
    rm -rf /tmp/vbi
    python3 -m venv /tmp/vbi
    unset PYTHONPATH
    source /tmp/vbi/bin/activate

    # Install core dependencies
    pip install ipykernel scikit_learn matplotlib

    # Install PyTorch (CPU version to save space)
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # Install SBI without dependencies to avoid reinstalling large packages
    pip install sbi --no-deps

    # Install SBI dependencies manually
    pip install pyro-ppl tensorboard nflows pyknos zuko arviz pymc

    # Install VBI from source
    mkdir -p /tmp/src && pushd /tmp/src
    git clone --depth 1 https://github.com/ins-amu/vbi.git
    cd vbi
    pip install -e .
    popd

    # Create Jupyter kernel
    python -m ipykernel install --user --name VBI

    echo "VBI environment created! Please reload your browser and select the 'VBI' kernel."
    echo "Note: This environment will be lost when the lab server shuts down."

Make the script executable and run it:

.. code-block:: bash

    chmod +x setup_vbi_ebrains.sh
    ./setup_vbi_ebrains.sh

**Important Notes:**

- Both environments are temporary and will be reset when the respective platforms shut down
- For EBRAINS, you'll need to rerun the setup script for each new session
- For Colab, you'll need to reinstall VBI in each new runtime

**Windows**

Windows installation is automatic - C++ compilation is automatically skipped:

.. code-block:: bash

    pip install vbi

Verification
------------

.. code-block:: python

    import vbi
    vbi.tests()
    vbi.test_imports()

Troubleshooting
---------------

**C++ Compilation Issues**

Note: the package is configured with SKIP_CPP=1 by default (C++ extensions are disabled).
If you want to compile and install the C++ components, set SKIP_CPP=0 when installing from source or via pip, for example:

.. code-block:: bash

    SKIP_CPP=0 pip install vbi

**Common Issues**

- **ImportError**: Check Python version (3.10+ recommended)
- **CUDA issues**: Verify GPU drivers and CUDA compatibility
- **Memory errors**: Try lighter installation variants

For detailed troubleshooting, platform guides, and advanced scenarios, see the complete `Installation Guide <https://github.com/ins-amu/vbi/blob/main/INSTALLATION.md>`_.
