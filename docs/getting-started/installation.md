# Installation

## Requirements

- **GPU**: NVIDIA GPU with Compute Capability 8.9+ (RTX 40-series recommended)
- **CUDA**: CUDA Toolkit 12.4 or later
- **Python**: 3.10, 3.11, or 3.12
- **OS**: Linux (Ubuntu 22.04 tested)

## From Source

```bash
# Clone repository
git clone https://github.com/Flameingmoy/GPGPU_in-mem_msg_q.git
cd GPGPU_in-mem_msg_q

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate gpuqueue

# Install in development mode
pip install -e ".[dev]"
```

## From PyPI

```bash
# Coming soon
pip install gpuqueue
```

## Optional Dependencies

```bash
# Track A (Redis backend) support
pip install gpuqueue[track-a]

# Hardware monitoring tools
pip install gpuqueue[monitor]

# Development tools
pip install gpuqueue[dev]
```

## Verify Installation

```python
import gpuqueue

# Check version
print(gpuqueue.version())

# Check if GPU bindings available
print(f"GPU available: {gpuqueue.is_available()}")
```
