# VBI Installation Guide

This guide provides detailed installation instructions for various platforms and troubleshooting common issues.

> **Credits:** Platform-specific installation instructions were developed based on valuable user feedback in [GitHub Issue #43](https://github.com/ins-amu/vbi/issues/43). Special thanks to community members who shared their experiences with Google Colab and EBRAINS installations.

## Quick Reference

For standard installation, see the [main README](README.md#installation). This guide covers:
- Platform-specific installations (Google Colab, EBRAINS)
- Troubleshooting common issues
- Advanced installation scenarios

## Platform-Specific Installation

### Google Colab

Google Colab doesn't have VBI or SBI pre-installed, and **Docker is not supported** in Colab due to security restrictions. For optimal C++ module compilation, install from source:

```bash
# In a Colab cell, run:
!mkdir -p src && cd src
!git clone --depth 1 https://github.com/ins-amu/vbi.git
%cd src/vbi
!pip install -e .
```

**Alternative: Use Colab Pro+ with Custom Runtimes**
If you have Colab Pro+ and need a containerized environment, consider:
- Using **Kaggle Notebooks** (supports Docker-based custom environments)
- Using **Binder** with our repository (though with limited resources)
- Setting up a **local Jupyter server** with our Docker image and connecting via ngrok

**Note:** The environment will be reset when the Colab runtime shuts down. You'll need to reinstall for each new session.

### EBRAINS Collab

EBRAINS has dependency management restrictions. Here's a script to create a dedicated VBI environment:

```bash
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
```

Make the script executable and run it:
```bash
chmod +x setup_vbi_ebrains.sh
./setup_vbi_ebrains.sh
```

**Important Notes:**
- Both environments are temporary and will be reset when the respective platforms shut down
- For EBRAINS, you'll need to rerun the setup script for each new session
- For Colab, you'll need to reinstall VBI in each new runtime

### Windows

**Windows installation is now automatic!** VBI automatically detects Windows systems and skips C++ compilation, eliminating the previous complexity.

**Standard Installation:**

```cmd
# Simple installation - C++ automatically skipped on Windows
pip install vbi

# Or from source
git clone https://github.com/ins-amu/vbi.git
cd vbi
pip install .
```

**What happens automatically:**
- VBI detects Windows system during installation
- C++ compilation is automatically skipped
- Installation proceeds with Python/NumPy/Numba implementations
- Full VBI functionality is available without C++ setup

**Manual override (if needed):**
You can still manually control C++ compilation on any system using environment variables:

```cmd
# Force skip C++ compilation (now redundant on Windows)
set SKIP_CPP=1
pip install .

# On other systems, force enable C++ compilation
set SKIP_CPP=0
pip install .
```

## Docker Alternatives for Cloud Platforms

Since many cloud platforms don't support Docker directly, here are alternatives:

<!-- ### MyBinder.org
Use our repository directly on Binder (free, but limited resources):
```
https://mybinder.org/v2/gh/ins-amu/vbi/main
``` -->

### Kaggle Notebooks
Kaggle supports Docker-based custom environments:
1. Create a new Kaggle notebook
2. Use our Docker image as a custom environment
3. Install VBI as shown above

### Local Jupyter + ngrok (for remote access)
Run our Docker image locally and access it remotely:
```bash
# Start VBI Docker container
docker run --gpus all -p 8888:8888 ghcr.io/ins-amu/vbi:main

# In another terminal, expose via ngrok
ngrok http 8888
```

### GitHub Codespaces
Use our repository in GitHub Codespaces with Docker support:
1. Open repository in Codespaces
2. Use the provided Docker configuration
3. Access Jupyter through the forwarded port

## Troubleshooting

### Common Issues

#### C++ Compilation Errors
For non-Windows systems, if you encounter C++ compilation issues:
```bash
# Skip C++ compilation during installation
SKIP_CPP=1 pip install -e .
```

**Note:** Windows users don't need to worry about this - C++ compilation is automatically skipped on Windows systems.

#### Dependency Conflicts
If you have conflicting dependencies:
```bash
# Create a fresh conda environment
conda create --name vbi-clean python=3.10
conda activate vbi-clean
pip install vbi[all]
```


### Custom Configurations
For specific use cases, you can install only what you need:

```bash
# Minimal installation (simulation only)
pip install vbi

# Add GPU support for simulations
pip install vbi[light-gpu]

# Add inference capabilities
pip install vbi[inference]

# Full installation with GPU support
pip install vbi[inference-gpu]
```

### Environment Variables
Useful environment variables for installation:

```bash
# Skip C++ compilation
export SKIP_CPP=1
```
## Getting Help

If you continue to experience issues:

1. Check our  [GitHub Issues](https://github.com/ins-amu/vbi/issues) for similar problems
2. Create a new issue with:
   - Your operating system and Python version
   - Complete error messages
   - Installation command you used
3. For platform-specific issues, mention the platform (Colab, EBRAINS, etc.)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to VBI.
