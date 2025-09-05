## Base image upgraded to CUDA 12 (runtime) for GPU support (minimum CUDA >=12 requested)
## Note: Previous tag with cudnn9 was unavailable. Using standard runtime tag.
## If you need cuDNN explicitly outside PyTorch/CuPy wheels, switch to a cudnn*-runtime tag.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

## Environment variables to expose GPU inside the container
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies (Python 3.10 is default in Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    swig \
    tzdata \
    git \
    && ln -s /usr/bin/python3 /usr/bin/python || true \
    && rm -rf /var/lib/apt/lists/*

# Set timezone (e.g., UTC) to avoid configuration prompts
RUN echo "Etc/UTC" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    hatchling \
    setuptools>=45 \
    wheel \
    swig>=4.0

## Install PyTorch with CUDA 12.x wheels before project install (ensures GPU-enabled torch)
## See: https://pytorch.org/get-started/locally/  (cu121 wheels work with CUDA 12.1+ runtime, works on 12.2 base)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

COPY . .

RUN pip install .[all] --no-cache-dir

# Install CuPy for CUDA 12.x
RUN pip install --no-cache-dir cupy-cuda12x

# Install Jupyter ecosystem with all required dependencies
RUN pip install --no-cache-dir \
    jupyterlab \
    jupyter \
    notebook \
    jupyter_server \
    ipykernel \
    ipython \
    nbformat \
    nbconvert

EXPOSE 8888

## Default command launches JupyterLab. For quick GPU validation you may instead run:
## docker run --rm --gpus all vbi:latest python -c "from vbi.utils import test_imports; test_imports()"
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
