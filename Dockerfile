# Use the smallest official Python image
FROM python:3.9-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*


# Copy only the package files (without installing dependencies)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir hatchling setuptools wheel && \
    pip install .

# Optional: Support GPU version (comment out if not needed)
ARG GPU_SUPPORT=false
RUN if [ "$GPU_SUPPORT" = "true" ]; then pip install cupy; fi

CMD ["python", "-c", "from vbi.utils import test_imports; test_imports()"]