# ET-partition Dockerfile
# This container provides a reproducible environment for running ET partitioning methods

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration files
COPY pyproject.toml /app/

# Create a minimal requirements.txt if needed (fallback)
# Note: pip install -e . will use pyproject.toml
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the package in editable mode with all dependencies
# This will install dependencies from pyproject.toml
COPY . /app/

# Install the ET-partition package and all dependencies
RUN pip install --no-cache-dir -e . && \
    # Install additional dependencies that might be needed
    pip install --no-cache-dir \
    scipy>=1.10 \
    emcee>=3.1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Create output directories
RUN mkdir -p /app/outputs /app/data

# Default command: run tests
# Users can override this to run different commands
ENTRYPOINT ["python"]
CMD ["tests/test_all_methods.py"]

# Alternative usage examples:
# docker run et-partition examples/basic_usage.py
# docker run et-partition examples/python_api_usage.py
# docker run et-partition -m pytest tests/
