Building Docker Image
#####################

This guide provides step-by-step instructions for building and using the VBI Docker image locally. The Dockerfile uses CUDA 12.2.0 runtime with Ubuntu 22.04 and Python 3.10 to create a production-ready image optimized for GPU computation.

Quick Reference
===============

**Image Status:** ✅ Built successfully as ``vbi:latest`` (size: ~7-8GB typical)

**Quick Commands:**

.. code-block:: bash

    # Build the image (first time or force rebuild)
    ./run-vbi.sh build
    
    # Start JupyterLab with GPU support
    ./run-vbi.sh jupyter
    
    # Start interactive container
    ./run-vbi.sh start
    
    # With local data access
    ./run-vbi.sh jupyter 8888 ./data
    
    # Interactive shell
    ./run-vbi.sh shell
    
    # Test installation
    ./run-vbi.sh test

**Convenience Script:** Use ``./run-vbi.sh`` for comprehensive container management (see `Management Script`_ section)

**Access JupyterLab:** http://127.0.0.1:8888 (no authentication required)

Prerequisites
=============

Before building the Docker image, ensure you have the following installed:

System Requirements
-------------------

.. code-block:: bash

    # Check Docker installation
    docker --version
    
    # Check for GPU support (optional but recommended)
    nvidia-smi

Expected output:

.. code-block:: text

    Docker version 28.0.1, build 068a01e
    
    # GPU check should show your NVIDIA GPU details
    Wed Sep  4 12:11:10 2025       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    |   0  NVIDIA RTX A5000               Off | 00000000:01:00.0 Off |                  Off |
    +-----------------------------------------+----------------------+----------------------+

Project Structure
=================

Ensure you have the optimized Dockerfile and .dockerignore in your project root:

.. code-block:: bash

    cd /path/to/vbi_develop
    ls -la Dockerfile .dockerignore

Expected files:

.. code-block:: text

    -rw-rw-r-- 1 user user 2130 sept.  3 12:38 Dockerfile
    -rw-rw-r-- 1 user user  665 sept.  3 12:06 .dockerignore

Optimized Dockerfile Features
=============================

The current Dockerfile includes several key features:

**CUDA 12.2.0 Runtime**
  - **Base image**: nvidia/cuda:12.2.0-runtime-ubuntu22.04
  - **Python 3.10**: Default in Ubuntu 22.04
  - **GPU support**: Full CUDA 12.x compatibility

**Optimized Dependencies**
  - PyTorch with CUDA 12.1 wheels (compatible with 12.2 runtime)
  - CuPy with CUDA 12.x support
  - JupyterLab as the default interface
  - No authentication required for development use

**Security & Usability**
  - Disabled authentication for local development
  - Timezone configuration to avoid prompts
  - Efficient package installation order

Build Process
=============

Step 1: Navigate to Project Directory
-------------------------------------

.. code-block:: bash

    cd /home/ziaee/git/inference/vbi_develop
    pwd

Step 2: Verify Docker is Running
--------------------------------

.. code-block:: bash

    docker --version
    docker info --format '{{.ServerVersion}}'

Expected output:

.. code-block:: text

    Docker version 28.0.1, build 068a01e
    27.5.1

Step 3: Build the Docker Image
------------------------------

.. code-block:: bash

    # Using the management script (recommended)
    ./run-vbi.sh build
    
    # Or manually with Docker
    docker build -t vbi:latest .
    
    # Force rebuild if needed
    ./run-vbi.sh build --force

**Build Process Overview:**

The build process consists of the following stages:

1. **Base Image Download** (5-10 minutes)
   
   .. code-block:: text
   
       => [internal] load build definition from Dockerfile
    => [internal] load metadata for nvidia/cuda:12.2.0-runtime-ubuntu22.04
    => FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

2. **System Dependencies Installation** (2-3 minutes)
   
   .. code-block:: text
   
    => RUN apt-get update && apt-get install -y python3 python3-pip python3-dev build-essential...

3. **Python Environment Setup** (1-2 minutes)
   
   .. code-block:: text
   
    => RUN python -m pip install --upgrade pip
    => RUN pip install hatchling setuptools>=45 wheel swig>=4.0

4. **PyTorch Installation** (5-10 minutes)
   
   .. code-block:: text
   
    => RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

5. **VBI and Dependencies Installation** (5-10 minutes)
   
   .. code-block:: text
   
    => COPY . .
    => RUN pip install .[all] --no-cache-dir
    => RUN pip install cupy-cuda12x

6. **Jupyter Ecosystem Installation** (2-3 minutes)
   
   .. code-block:: text
   
    => RUN pip install jupyterlab jupyter notebook jupyter_server ipykernel ipython nbformat nbconvert

Step 4: Verify Build Success
----------------------------

.. code-block:: bash

    docker images | grep vbi

Expected output:

.. code-block:: text

    vbi          latest    abc123def456   2 minutes ago   6.8GB

Running the Container
=====================

Now that you have successfully built the VBI Docker image, here's how to use it effectively.

Quick Start
-----------

**Start JupyterLab with GPU support:**

.. code-block:: bash

    # Using the management script (recommended)
    ./run-vbi.sh jupyter
    
    # Or manually with Docker
    docker run --gpus all -p 8888:8888 vbi:latest

**Access the JupyterLab interface:**

1. The management script will display the access URL automatically:

.. code-block:: text

    [INFO] Starting VBI Jupyter server...
    [INFO] GPU support detected and enabled
    [SUCCESS] VBI Jupyter server started successfully!
    [INFO] Container name: vbi-workspace
    [INFO] Local port: 8888
    [INFO] Access URL: http://localhost:8888
    [INFO] Your current directory is mounted at: /app/workspace

2. **No authentication required** - open browser to: ``http://127.0.0.1:8888``

Usage Scenarios
---------------

**Scenario 1: Basic Data Science Work**

.. code-block:: bash

    # Using management script (recommended)
    ./run-vbi.sh jupyter
    
    # Or manually
    docker run --gpus all -p 8888:8888 vbi:latest

**Scenario 2: Working with Local Data**

.. code-block:: bash

    # Using management script with data directory
    ./run-vbi.sh jupyter 8888 /path/to/your/data
    
    # Or manually
    docker run --gpus all -p 8888:8888 -v /path/to/your/data:/app/data vbi:latest

**Scenario 3: Interactive Development**

.. code-block:: bash

    # Start interactive container with workspace mounting
    ./run-vbi.sh start
    
    # Or manually
    docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/app/workspace vbi:latest

**Scenario 4: Interactive Shell Access**

.. code-block:: bash

    # Using management script
    ./run-vbi.sh shell
    
    # Or manually
    docker run --gpus all -it --entrypoint /bin/bash vbi:latest

**Scenario 5: Background Running**

.. code-block:: bash

    # Using management script
    ./run-vbi.sh jupyter
    
    # Check status and logs
    ./run-vbi.sh status
    ./run-vbi.sh logs
    
    # Stop when done
    ./run-vbi.sh stop

Testing Your Installation
-------------------------

**1. Quick Functionality Test**

.. code-block:: bash

    # Using the management script (recommended)
    ./run-vbi.sh test

Expected output:

.. code-block:: text

    [INFO] Running VBI functionality test...
    [INFO] GPU support detected, testing with GPU access
    VBI Docker Test Results
    ==================================================
    VBI version: v0.2

                 Dependency Check              
                                               
      Package      Version       Status        
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
      vbi          v0.2          ✅ Available  
      numpy        2.1.2         ✅ Available  
      scipy        1.15.3        ✅ Available  
      matplotlib   3.10.6        ✅ Available  
      sbi          0.24.0        ✅ Available  
      torch        2.5.1+cu121   ✅ Available  
      cupy         13.6.0        ✅ Available  
                                               
    Torch GPU available: True
    Torch device count: 1
    Torch CUDA version: 12.1
    CuPy GPU available: True
    CuPy device count: 1
    CUDA Version: 12.9
    Device Name: NVIDIA RTX A5000
    Total Memory: 23.68 GB
    Compute Capability: 8.6

    🎉 VBI Docker image is working correctly!

**2. Manual Import Test**

Create a new notebook and test the VBI installation:

.. code-block:: python

    import vbi
    print(f"VBI version: {vbi.__version__}")

**3. GPU Functionality Test**

.. code-block:: python

    import torch
    import cupy as cp
    
    # Test PyTorch GPU
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.get_device_name()}")
    
    # Test CuPy GPU
    print(f"CuPy device count: {cp.cuda.runtime.getDeviceCount()}")
    with cp.cuda.Device(0):
        mempool = cp.get_default_memory_pool()
        print(f"GPU memory: {mempool.used_bytes()} / {mempool.total_bytes()} bytes")

**3. Run VBI Examples**

The container includes example notebooks. Access them via JupyterLab:

.. code-block:: bash

    # Start Jupyter and navigate to examples
    ./run-vbi.sh jupyter
    
    # In JupyterLab, look for the docs/examples/ directory

You can also list examples programmatically:

.. code-block:: python

    # List available examples
    import os
    examples_dir = "/app/docs/examples"
    example_files = [f for f in os.listdir(examples_dir) if f.endswith('.ipynb')]
    print("Available examples:")
    for example in example_files:
        print(f"  - {example}")

Working with Data
-----------------

**Mounting Local Directories**

The most efficient way to work with your data using the management script:

.. code-block:: bash

    # Create a working directory structure
    mkdir -p ~/vbi-workspace/{data,notebooks,output}
    
    # Start Jupyter with mounted data directory
    ./run-vbi.sh jupyter 8888 ~/vbi-workspace/data
    
    # Or start interactive container with workspace
    cd ~/vbi-workspace
    ./run-vbi.sh start

**Manual Docker Commands for Data Mounting:**

.. code-block:: bash

    # Run container with mounted directories
    docker run --gpus all -p 8888:8888 \
        -v ~/vbi-workspace/data:/app/data \
        -v ~/vbi-workspace/notebooks:/app/notebooks \
        -v ~/vbi-workspace/output:/app/output \
        vbi:latest

**File Structure Inside Container:**

.. code-block:: text

    /app/
    ├── data/          # Your input data (mounted via ./run-vbi.sh jupyter 8888 ./data)
    ├── workspace/     # Current directory (mounted via ./run-vbi.sh start)
    ├── docs/          # VBI documentation and examples
    └── vbi/           # VBI source code

**Copying Files to/from Container**

If you need to copy files without mounting:

.. code-block:: bash

    # Copy file TO container (get container name with ./run-vbi.sh status)
    docker cp /path/to/local/file.txt vbi-workspace:/app/data/
    
    # Copy file FROM container
    docker cp vbi-workspace:/app/output/results.csv /path/to/local/
    
    # Check container status and name
    ./run-vbi.sh status

Advanced Usage
==============

Custom Jupyter Configuration
----------------------------

**1. Custom Jupyter Settings**

Create a custom jupyter configuration:

.. code-block:: bash

    # Create config directory
    mkdir -p ~/vbi-workspace/jupyter-config
    
    # Run with custom config
    docker run --gpus all -p 8888:8888 \
        -v ~/vbi-workspace/jupyter-config:/root/.jupyter \
        vbi:latest

**2. Custom Python Packages**

Install additional packages at runtime:

.. code-block:: bash

    # Run interactive shell
    docker run --gpus all -it --entrypoint /bin/bash vbi:latest
    
    # Inside container, install packages
    pip install seaborn plotly

    # Or create a custom Dockerfile extending vbi:latest
    echo "FROM vbi:latest
    RUN pip install seaborn plotly" > Dockerfile.custom
    
    docker build -f Dockerfile.custom -t vbi:custom .

**3. Environment Variables**

.. code-block:: bash

    # Set custom environment variables
    docker run --gpus all -p 8888:8888 \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e OMP_NUM_THREADS=4 \
        vbi:latest

Multiple Container Instances
----------------------------

Run multiple instances for different projects:

.. code-block:: bash

    # Project 1 on port 8888
    docker run -d --name vbi-project1 --gpus all -p 8888:8888 \
        -v ~/project1:/app/notebooks vbi:latest
    
    # Project 2 on port 8889
    docker run -d --name vbi-project2 --gpus all -p 8889:8888 \
        -v ~/project2:/app/notebooks vbi:latest

Performance Optimization
========================

Memory Management
-----------------

**Monitor Resource Usage:**

.. code-block:: bash

    # Monitor container resources
    docker stats vbi-container
    
    # Check GPU usage
    nvidia-smi

**Memory Limits:**

.. code-block:: bash

    # Limit container memory (e.g., 8GB)
    docker run --gpus all -p 8888:8888 --memory=8g vbi:latest
    
    # Limit GPU memory (for multi-GPU systems)
    docker run --gpus '"device=0"' -p 8888:8888 vbi:latest

Best Practices for Development
------------------------------

**1. Use Volume Mounts for Code Development**

.. code-block:: bash

    # Mount your development directory
    docker run --gpus all -p 8888:8888 \
        -v ~/my-vbi-project:/app/workspace \
        vbi:latest

**2. Set Up Development Workflow**

.. code-block:: bash

    # Create a development script
    cat > run-vbi-dev.sh << 'EOF'
    #!/bin/bash
    docker run --gpus all -p 8888:8888 \
        -v $(pwd):/app/workspace \
        -v ~/.ssh:/root/.ssh:ro \
        -v ~/.gitconfig:/root/.gitconfig:ro \
        --name vbi-dev \
        --rm \
        vbi:latest
    EOF
    
    chmod +x run-vbi-dev.sh
    ./run-vbi-dev.sh

**3. Jupyter Lab Alternative**

If you prefer JupyterLab interface:

.. code-block:: bash

    # Install JupyterLab in the container
    docker run --gpus all -it --entrypoint /bin/bash vbi:latest
    pip install jupyterlab
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

Debugging and Troubleshooting
=============================

Common Runtime Issues
---------------------

**Image Not Found:**

.. code-block:: bash

    # The management script will auto-build if image is missing
    ./run-vbi.sh jupyter
    
    # Or build manually
    ./run-vbi.sh build

**Container Won't Start:**

.. code-block:: bash

    # Check container status and logs
    ./run-vbi.sh status
    ./run-vbi.sh logs
    
    # Or manually check
    docker ps -a
    docker logs vbi-workspace

**GPU Not Detected:**

.. code-block:: bash

    # Test GPU support
    ./run-vbi.sh test
    
    # Verify NVIDIA Docker runtime manually
    docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi
    
    # Check if GPU is accessible in VBI container
    ./run-vbi.sh shell
    # Then in container: nvidia-smi

**Port Already in Use:**

.. code-block:: bash

    # Use different port
    ./run-vbi.sh jupyter 8889
    # Access via http://127.0.0.1:8889

**Permission Issues:**

.. code-block:: bash

    # Check Docker permissions
    docker info
    
    # Make sure current user is in docker group
    sudo usermod -aG docker $USER
    # Then log out and back in

Container Management
--------------------

**View Running Containers:**

.. code-block:: bash

    # Using management script (recommended)
    ./run-vbi.sh status                  # Check VBI container status
    ./run-vbi.sh image                   # Show all VBI images and containers
    
    # Manual Docker commands
    docker ps                            # Running containers
    docker ps -a                         # All containers

**Container Lifecycle:**

.. code-block:: bash

    # Using management script
    ./run-vbi.sh start                   # Start interactive container
    ./run-vbi.sh jupyter                 # Start Jupyter server
    ./run-vbi.sh stop                    # Stop running container
    ./run-vbi.sh restart                 # Restart container
    
    # Manual Docker commands
    docker start vbi-workspace           # Start stopped container
    docker stop vbi-workspace            # Stop running container
    docker rm vbi-workspace              # Remove container

**Cleanup Operations:**

.. code-block:: bash

    # Safe cleanup with confirmation prompts
    ./run-vbi.sh clean                   # Full cleanup (containers + images)
    ./run-vbi.sh remove                  # Remove containers only
    
    # Force cleanup (skip confirmations)
    ./run-vbi.sh clean --force
    ./run-vbi.sh remove --force

**Accessing Running Container:**

.. code-block:: bash

    # Using management script
    ./run-vbi.sh shell                   # New interactive shell
    
    # Manual access to running container
    docker exec -it vbi-workspace bash
    
    # Run Python in running container
    docker exec -it vbi-workspace python

Management Script
=================

A comprehensive convenience script ``run-vbi.sh`` is provided in the project root to simplify all container management tasks:

**Available Commands:**

.. code-block:: bash

    # Building
    ./run-vbi.sh build                    # Build VBI Docker image
    ./run-vbi.sh build --force            # Force rebuild
    
    # Testing
    ./run-vbi.sh test                     # Test VBI installation
    
    # Running Containers
    ./run-vbi.sh start                    # Start interactive container (default port 8888)
    ./run-vbi.sh start 8889               # Start on custom port
    ./run-vbi.sh jupyter                  # Start Jupyter server (background)
    ./run-vbi.sh jupyter 8889             # Start Jupyter on custom port
    ./run-vbi.sh jupyter 8888 ./data      # Start Jupyter with data directory
    ./run-vbi.sh shell                    # Open interactive shell
    
    # Container Management
    ./run-vbi.sh stop                     # Stop running container
    ./run-vbi.sh restart                  # Restart container
    ./run-vbi.sh status                   # Check container status
    ./run-vbi.sh logs                     # View container logs
    
    # Cleanup (with confirmation prompts)
    ./run-vbi.sh clean                    # Full cleanup (containers + images)
    ./run-vbi.sh clean --force            # Skip confirmation
    ./run-vbi.sh remove                   # Remove containers only
    ./run-vbi.sh remove --force           # Skip confirmation
    
    # Information
    ./run-vbi.sh image                    # Show images and containers info
    ./run-vbi.sh help                     # Show all options

**Key Features:**

- **Auto-build functionality**: Automatically builds image if missing
- **GPU detection**: Enables GPU support if available, falls back to CPU
- **Colored output**: Clear status messages with color coding
- **Error handling**: Comprehensive Docker status and image checks
- **Token extraction**: Displays Jupyter access URL automatically
- **Volume mounting**: Easy data directory mounting
- **Container lifecycle**: Complete start, stop, restart, and monitor capabilities
- **Safety prompts**: Confirmation prompts for destructive operations
- **Comprehensive help**: Built-in help system with examples

**Example Usage:**

.. code-block:: bash

    # Quick start workflow
    ./run-vbi.sh build                    # Build image (first time)
    ./run-vbi.sh test                     # Test installation
    ./run-vbi.sh jupyter 8888 ~/my-data   # Start with data
    
    # Development workflow
    cd ~/my-vbi-project
    ./run-vbi.sh start                    # Interactive with workspace mounted
    
    # Background Jupyter workflow
    ./run-vbi.sh jupyter                  # Start Jupyter server
    ./run-vbi.sh status                   # Check status
    ./run-vbi.sh logs                     # View logs
    ./run-vbi.sh stop                     # Stop when done

**Script Output Example:**

.. code-block:: text

    [INFO] Starting VBI Jupyter server...
    [INFO] GPU support detected and enabled
    [INFO] Mounting data directory: /home/user/data -> /app/data
    [SUCCESS] VBI Jupyter server started successfully!
    [INFO] Container name: vbi-workspace
    [INFO] Local port: 8888
    [INFO] Access URL: http://localhost:8888
    [INFO] Your current directory is mounted at: /app/workspace
    
    To stop the server: ./run-vbi.sh stop
    To view logs: ./run-vbi.sh logs

Next Steps
==========

Now that you know how to use the VBI Docker image, you can:

1. **Explore the Examples**: Start with the included notebook examples
2. **Load Your Data**: Mount your datasets and begin analysis
3. **Develop Models**: Use the full VBI toolkit for brain modeling
4. **Scale Up**: Utilize GPU acceleration for larger computations
5. **Collaborate**: Share the Docker image with team members for consistent environments

For more information about VBI functionality, see the :doc:`API` documentation and :doc:`models` reference.

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Docker Daemon Not Running**

.. code-block:: text

    Error: Cannot connect to the Docker daemon at unix:///var/run/docker.sock

Solution:

.. code-block:: bash

    # Start Docker service
    sudo systemctl start docker
    # Or start Docker Desktop

**Package Not Found Error**

.. code-block:: text

    E: Unable to locate package python3.10-dev

Solution: The Dockerfile has been updated to use ``python3`` and ``python3-dev`` which are available in Ubuntu 20.04.

**Out of Disk Space**

.. code-block:: text

    ERROR: failed to solve: write /var/lib/docker/...: no space left on device

Solution: Ensure you have at least 15-20GB free disk space.

**Build Cancelled/Interrupted**

If the build is interrupted, simply restart:

.. code-block:: bash

    docker build -t vbi:latest .

Docker will use cached layers and continue from where it left off.

Image Optimization and Features
===============================

The VBI Dockerfile is optimized for production use with the following characteristics:

**Current Image Specifications:**
  - **Base**: nvidia/cuda:12.2.0-runtime-ubuntu22.04
  - **Size**: ~7-8GB (production-optimized)
  - **Python**: 3.10 (Ubuntu 22.04 default)
  - **CUDA**: 12.2.0 runtime with full GPU support

**Key Features:**
  - **No authentication**: Streamlined for development use
  - **JupyterLab**: Modern interface as default
  - **GPU-optimized**: PyTorch and CuPy with CUDA 12.x support
  - **Complete environment**: All VBI dependencies included

**Size Breakdown:**
  - Base CUDA runtime: ~2GB
  - Python packages (PyTorch, CuPy, etc.): ~4-5GB
  - VBI and scientific dependencies: ~1-2GB
  - System libraries and tools: ~500MB

Advanced Usage
==============

Building with Specific Tags
---------------------------

.. code-block:: bash

    # Build with version tag
    docker build -t vbi:0.2.1 .
    
    # Build with custom name
    docker build -t my-vbi-image:latest .

Skipping C++ Compilation
------------------------

If you encounter C++ compilation issues:

.. code-block:: bash

    docker build --build-arg SKIP_CPP=1 -t vbi:latest .

Using Docker Compose
--------------------

Create a ``docker-compose.yml`` file:

.. code-block:: yaml

    version: '3.8'
    services:
      vbi:
        build: .
        ports:
          - "8888:8888"
        volumes:
          - ./data:/app/data
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]

Run with:

.. code-block:: bash

    docker compose up --build

Performance Tips
================

**Build Performance:**
  - Use fast internet connection for initial build
  - Subsequent builds will be much faster due to layer caching
  - Build during off-peak hours for better download speeds

**Runtime Performance:**
  - Always use ``--gpus all`` for GPU acceleration
  - Mount volumes for large datasets to avoid copying into container
  - Use specific image tags to avoid rebuilding unnecessarily

**Resource Requirements:**
  - **Build time**: 15-30 minutes (first build), 2-5 minutes (subsequent builds)
  - **Disk space**: 15-20GB during build, 6-8GB final image
  - **Memory**: 4GB+ recommended during build
  - **Network**: High-speed internet recommended for package downloads

Maintenance
===========

Cleaning Up
-----------

Remove unused images and containers:

.. code-block:: bash

    # Using management script (recommended)
    ./run-vbi.sh clean                   # Full cleanup with confirmation
    ./run-vbi.sh remove                  # Remove containers only
    ./run-vbi.sh image                   # Show current VBI images/containers
    
    # Manual Docker commands
    docker image prune                   # Remove dangling images
    docker container prune               # Remove unused containers
    docker system prune -a               # Remove everything (be careful!)

Updating the Image
------------------

To rebuild with latest code changes:

.. code-block:: bash

    # Pull latest code
    git pull origin develop
    
    # Rebuild image using management script
    ./run-vbi.sh build --force
    
    # Or manually with Docker
    docker build --no-cache -t vbi:latest .

The ``--force`` flag or ``--no-cache`` ensures a complete rebuild when needed.
