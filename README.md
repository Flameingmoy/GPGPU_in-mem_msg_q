# GPUQueue (CUDA Message Queue) — Scaffold

This repository contains a scaffold for a CUDA-backed, GPU-resident message queue with a Python API.

- Build system: scikit-build-core + CMake + PyBind11 + CUDA
- Python package: `gpuqueue`
- Target: Linux x86_64, Python 3.10–3.12, CUDA 12.6+, RTX 4070 Ti Super (sm_89)

## Layout
```
src/
  python/gpuqueue/        # Python package
  cpp/                    # C++ sources and PyBind11 bindings
  cuda/                   # CUDA kernels
include/gpuqueue/         # Public C++ headers
```

## Architecture

```mermaid
flowchart LR
  subgraph A[Track A - Redis-backed MVP]
    P["Producers"] --> R["Redis List/Stream"]
    R --> H["Host Worker (Python/C++)"]
    H --> S["Host Staging Pinned Buffer"]
    S --> GQ["GPU Queue API"]
    GQ --> K["Persistent Kernel"]
    K --> PR["Process Message (Device Function)"]
    PR --> HR["Host Results/ACK"]
    HR --> RA["Redis ACK"]
  end

  subgraph B[Track B - GPU-resident Stage-2]
    P2["Producers"] --> API["Host API: enqueue_async"]
    API --> RB["GPU Ring Buffer Device Global Memory"]
    RB --> K2["Persistent Kernel"]
    K2 --> PROC2["Process Message (Device Function)"]
    PROC2 --> RES["Results / DONE"]
    RES --> DQ["Host API: try_dequeue_result"]
  end

  %% Notes: correctness via CUDA atomics + memory fences
  %% (__threadfence, __threadfence_system) ensure publish/visibility across host/device
```

See `docs/design.md` and `docs/api.md` for detailed semantics and synchronization.

## Quickstart (build locally)

Prereqs: Python 3.10+, CUDA Toolkit (nvcc), CMake ≥3.24, Ninja ≥1.11

```bash
python -m pip install -U pip
python -m pip install -e .
python -c "import gpuqueue as gq; print(gq.__version__); print(gq.core_version() if gq.core_version else 'no core')"
```

Notes:
- The core currently exposes `init()`, `shutdown()`, and `version()` from the `_core` extension.
- See `docs/packaging.md` for packaging and CI details.

