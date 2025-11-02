# Docker Usage Guide for ET-partition

This guide explains how to use Docker to run ET-partition methods in a containerized environment.

## Quick Start

### Build the Docker image

```bash
docker build -t et-partition:latest .
```

### Run tests

```bash
docker run --rm et-partition:latest tests/test_all_methods.py
```

### Run examples

```bash
# Basic usage examples
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs:rw \
  et-partition:latest examples/basic_usage.py

# Python API examples
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs:rw \
  et-partition:latest examples/python_api_usage.py

# Advanced PFT analysis
docker run --rm \
  -v $(pwd)/outputs:/app/outputs:rw \
  et-partition:latest examples/advanced_pft_analysis.py
```

## Using Docker Compose

Docker Compose provides a simpler way to manage volumes and services.

### Run default service (tests)

```bash
docker-compose up et-partition
```

### Run specific batch processing

```bash
# uWUE batch processing
docker-compose up uwue-batch

# TEA batch processing
docker-compose up tea-batch

# Perez-Priego batch processing
docker-compose up perez-priego-batch
```

### Run all batch processing methods in parallel

```bash
docker-compose up uwue-batch tea-batch perez-priego-batch
```

## Volume Mounts

The Docker container uses the following volume mounts:

- `/app/data` - Input data directory (read-only)
  - Mount your FLUXNET data here
- `/app/outputs` - Output directory (read-write)
  - Results will be written here

## Environment Variables

- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered
- `MPLBACKEND=Agg` - Sets matplotlib to use non-interactive backend

## Advanced Usage

### Interactive shell

```bash
docker run --rm -it \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs:rw \
  et-partition:latest /bin/bash
```

### Run with custom command

```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs:rw \
  et-partition:latest python -c "from methods.uwue import zhou_part; print('uWUE imported successfully')"
```

### Development mode (mount source code)

```bash
docker run --rm -it \
  -v $(pwd):/app:rw \
  et-partition:latest /bin/bash
```

## Troubleshooting

### Permission issues with output files

If you encounter permission issues with output files, you may need to run the container with your user ID:

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs:rw \
  et-partition:latest examples/basic_usage.py
```

### Build cache issues

If you need to rebuild without cache:

```bash
docker build --no-cache -t et-partition:latest .
```

### Check installed packages

```bash
docker run --rm et-partition:latest -m pip list
```

## Resource Requirements

- **RAM**: Minimum 4GB recommended, 8GB+ for large datasets
- **CPU**: Multi-core recommended for parallel processing
- **Disk**: Depends on data size, allow sufficient space for outputs

## Notes

- The container includes all dependencies from `pyproject.toml`
- Scientific packages (numpy, scipy, pandas, scikit-learn) are included
- Additional dependencies (emcee for Perez-Priego method) are pre-installed
- The container is based on `python:3.10-slim` for a balance of size and functionality
