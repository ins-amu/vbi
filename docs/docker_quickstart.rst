VBI Docker Quick Start
######################

Your VBI Docker image has been built successfully! 🎉

Image Details
=============

- **Image name**: ``vbi:latest``
- **Size**: 16.1GB (optimized from ~18GB)
- **Status**: ✅ Ready to use

Quick Start
===========

Build the Image (First Time)
------------------------------

.. code-block:: bash

    # Build VBI Docker image
    ./run-vbi.sh build
    
    # Force rebuild if needed
    ./run-vbi.sh build --force

Test the Installation
---------------------

.. code-block:: bash

    ./run-vbi.sh test

Start Interactive Container
---------------------------

.. code-block:: bash

    # Start interactive container (auto-builds if needed)
    ./run-vbi.sh start

    # With custom port
    ./run-vbi.sh start 8889

Start Jupyter Server
--------------------

.. code-block:: bash

    # Start Jupyter server in background
    ./run-vbi.sh jupyter

    # With custom port
    ./run-vbi.sh jupyter 8889

    # With data directory mounted
    ./run-vbi.sh jupyter 8888 ./my-data

Access Jupyter
--------------

1. The script will display the access URL automatically
2. Open browser to: http://localhost:8888
3. **No token required** - authentication is disabled by default

Interactive Shell
-----------------

.. code-block:: bash

    ./run-vbi.sh shell

Container Management
--------------------

.. code-block:: bash

    ./run-vbi.sh status    # Check status
    ./run-vbi.sh logs      # View logs  
    ./run-vbi.sh stop      # Stop container
    ./run-vbi.sh restart   # Restart container

Cleanup Commands
----------------

.. code-block:: bash

    ./run-vbi.sh clean     # Full cleanup (with confirmation)
    ./run-vbi.sh remove    # Remove containers only
    ./run-vbi.sh image     # Show images and containers info

Get Help
--------

.. code-block:: bash

    ./run-vbi.sh help      # Show all available commands

Manual Docker Commands
======================

If you prefer using Docker directly:

.. code-block:: bash

    # Start JupyterLab (with GPU if available)
    docker run --gpus all -p 8888:8888 vbi:latest

    # Start JupyterLab (CPU only)
    docker run -p 8888:8888 vbi:latest

    # Interactive shell
    docker run --gpus all -it --entrypoint /bin/bash vbi:latest

    # With data mounting
    docker run --gpus all -p 8888:8888 -v $(pwd)/data:/app/data vbi:latest

    # Interactive container with workspace mounting
    docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/app/workspace vbi:latest

What's Included
===============

- ✅ VBI v0.2.1 
- ✅ PyTorch with CUDA 12.x support
- ✅ CuPy for GPU acceleration (CUDA 12.x)
- ✅ NumPy, SciPy, Matplotlib
- ✅ JupyterLab (default interface)
- ✅ Ubuntu 22.04 with Python 3.10
- ✅ All VBI dependencies

Documentation
=============

For detailed documentation and advanced usage, see:

- :doc:`docker_build` - Complete build and usage guide
- :doc:`index` - Main VBI documentation

Troubleshooting
===============

**Container won't start?**

.. code-block:: bash

    ./run-vbi.sh status
    ./run-vbi.sh logs

**GPU not working?**

- GPU support requires NVIDIA Docker runtime
- CPU mode works perfectly for development

**Port already in use?**

.. code-block:: bash

    ./run-vbi.sh start 8889  # Use different port

**Need help?**

.. code-block:: bash

    ./run-vbi.sh help

----

Happy computing with VBI! 🧠⚡
