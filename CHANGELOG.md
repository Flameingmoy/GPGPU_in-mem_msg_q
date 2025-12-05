# Changelog

All notable changes to GPUQueue will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] - 2025-12-05

### Added

- **GPU-Resident Queue (Track B)**
  - Ring buffer in VRAM with persistent CUDA kernel
  - Lock-free slot state machine (EMPTY → READY → INFLIGHT → DONE)
  - `GpuQueue` class with `enqueue()`, `try_dequeue_result()`, `poll_completions()`
  - Context manager support for automatic cleanup
  - ~65k msg/s throughput, 0.13ms p50 latency

- **Redis-Backed Queue (Track A)**
  - `Producer`, `Consumer`, `GpuProcessor` classes
  - Redis Streams integration with consumer groups
  - CuPy-based GPU processing
  - ~38k msg/s throughput

- **Python Bindings**
  - pybind11 bindings for C++ GpuQueue
  - `QueueConfig`, `QueueStats`, `QueueStatus` exposed to Python
  - Buffer protocol support (bytes, numpy, memoryview)

- **Testing & Benchmarking**
  - 43 Python unit tests
  - CUDA sanitizer verification (memcheck, racecheck, initcheck, synccheck)
  - Hardware monitoring scripts (CPU, GPU, RAM, VRAM)
  - Soak tests for stability verification

- **CI/CD**
  - GitHub Actions workflows for lint, build, test
  - Release workflow with cibuildwheel
  - mkdocs documentation site

### Technical Details

- Target: NVIDIA GPUs with Compute Capability 8.9+ (RTX 40-series)
- CUDA 12.4+, Python 3.10-3.12, Linux x86_64
- Uses `__threadfence_system()` for host/device synchronization
- Non-default CUDA streams (persistent kernel blocks default stream)

[Unreleased]: https://github.com/Flameingmoy/GPGPU_in-mem_msg_q/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/Flameingmoy/GPGPU_in-mem_msg_q/releases/tag/v0.1.0a1
