FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    swig \
    tzdata \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Set timezone (e.g., UTC) to avoid configuration prompts
RUN echo "Etc/UTC" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    hatchling \
    setuptools>=45 \
    wheel \
    swig>=4.0

COPY . .

RUN pip install . --no-cache-dir
RUN pip install cupy-cuda11x

# Install Jupyter Notebook and related packages
RUN pip install --no-cache-dir \
    notebook \
    ipykernel \
    ipython

EXPOSE 8888

# Set the default command (modify as needed)
# CMD ["python", "-c", "from vbi.utils import test_imports; test_imports()"]
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
