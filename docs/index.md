# GPUQueue

**CUDA-backed GPU-resident message queue with Python bindings.**

GPUQueue is a high-performance message queue that processes messages entirely on the GPU, avoiding CPU-GPU round trips for maximum throughput.

## Features

- **GPU-Resident Processing**: Messages stay in VRAM from enqueue to dequeue
- **Persistent CUDA Kernel**: Always-on consumer kernel eliminates launch overhead  
- **Lock-Free Ring Buffer**: Efficient producer-consumer pattern with atomic operations
- **Python Bindings**: Easy-to-use Python API via pybind11
- **Two-Track Architecture**:
    - **Track A**: Redis-backed queue for validation and comparison
    - **Track B**: Full GPU-resident implementation (primary)

## Performance

| Metric | Track A (Redis) | Track B (GPU) | Speedup |
|--------|-----------------|---------------|---------|
| Throughput | 38k msg/s | 65k msg/s | **1.7x** |
| Latency (p50) | 0.14ms | 0.13ms | ~same |
| VRAM Usage | N/A | ~200 MB | - |

## Quick Start

```python
from gpuqueue import GpuQueue, QueueConfig

# Create queue with 1024 slots, 512 bytes per slot
config = QueueConfig(capacity=1024, slot_bytes=512)

with GpuQueue(config) as q:
    # Enqueue message
    msg_id = q.enqueue(b"Hello, GPU!")
    
    # Poll for completion
    completed = q.poll_completions(10)
    
    # Dequeue result
    if msg_id in completed:
        success, data = q.try_dequeue_result(msg_id)
        print(f"Result: {data}")
```

## Requirements

- NVIDIA GPU (Compute Capability 8.9+, e.g., RTX 4070 Ti Super)
- CUDA Toolkit 12.4+
- Python 3.10-3.12
- Linux (Ubuntu 22.04 recommended)

## Installation

```bash
# From source
git clone https://github.com/Flameingmoy/GPGPU_in-mem_msg_q.git
cd GPGPU_in-mem_msg_q
pip install -e ".[dev]"

# From PyPI (coming soon)
pip install gpuqueue
```

## License

GNU General Public License v3.0
