# Packaging Plan: gpuqueue (Python + CUDA)

This document defines how we will package and distribute the CUDA-backed GPU message queue as a Python module with a C++/CUDA core.

Goals:
- Provide a single `pip install gpuqueue` experience for end users on Linux (x86_64) with NVIDIA GPUs.
- Build a binary wheel containing the CUDA-backed core via PyBind11 and scikit-build-core.
- Keep Redis strictly optional for Track A (MVP) via extras: `pip install gpuqueue[track-a]`.
- Avoid bundling NVIDIA drivers or CUDA toolkit runtime libraries; rely on system NVIDIA drivers and a compatible CUDA runtime.
- Target initial support: Linux x86_64, Python 3.10–3.12, CUDA Toolkit 12.6+, GPU CC 8.9 (RTX 4070 Ti Super).


## Package Architecture

- Python package: `gpuqueue`
- Native extension module: `gpuqueue._core` (C++/CUDA via PyBind11)
- Core library: `libgpuqueue` (C++/CUDA)
- Build system: `scikit-build-core` + `CMake` + `PyBind11` + `CUDA`

Suggested repo layout (no code yet, just a target structure):
```
Cuda_Message_Queue/
  docs/
  src/
    python/
      gpuqueue/
        __init__.py        # thin Python API layer
        _version.py        # version string
    cpp/
      bindings.cpp         # pybind11 module definitions
      queue_core.hpp
      queue_core.cpp
    cuda/
      queue_kernels.cu
    include/gpuqueue/
      queue.hpp
      ring_buffer.hpp
  CMakeLists.txt
  pyproject.toml
```


## Build Backend and Python Metadata (pyproject.toml)

Use scikit-build-core for a modern, minimal PEP 517 build.

Example `pyproject.toml` (illustrative):
```toml
[build-system]
requires = [
  "scikit-build-core>=0.9.0",
  "pybind11>=2.11",
  "cmake>=3.24",
  "ninja>=1.11",
  # Track A extra (optional for end users): bring in at install-time only if requested
]
build-backend = "scikit_build_core.build"

[project]
name = "gpuqueue"
version = "0.1.0a0"
description = "CUDA-backed GPU-resident message queue with Python bindings"
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "Your Name" }]
license = { text = "Apache-2.0" }
classifiers = [
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3 :: Only",
  "Programming Language :: C++",
  "Programming Language :: Python",
  "Programming Language :: Python :: Implementation :: CPython",
  "Operating System :: POSIX :: Linux",
  "License :: OSI Approved :: Apache Software License",
]

[project.optional-dependencies]
# Track A MVP dependencies kept optional
track-a = [
  "redis[hiredis]>=5,<6",
  "msgpack>=1.0,<2",
]

dev = [
  "pytest",
  "pytest-benchmark",
  "compute-sanitizer; platform_system == 'Linux'",  # run manually in CI images that provide it
  "mypy",
  "ruff",
]

[tool.scikit-build]
# Where the Python package lives
wheel.packages = ["src/python/gpuqueue"]
# CMake settings
cmake.verbose = true
ninja.minimum-version = "1.11"

[tool.scikit-build.cmake]
minimum-version = "3.24"

[tool.cibuildwheel]
# Build linux wheels; GPU CI needs self-hosted runners or specialized images
build = "cp310-* cp311-* cp312-*"
skip = ["*musllinux*"]
```

Notes:
- We keep Redis-client deps in an optional extra (`track-a`) so core users aren’t forced to install them.
- Version can later be managed by `setuptools_scm` or similar; pinned here for clarity.


## CMake Configuration

Minimal `CMakeLists.txt` skeleton (illustrative, not yet committed):
```cmake
cmake_minimum_required(VERSION 3.24)
project(gpuqueue LANGUAGES CXX CUDA)

# Prefer C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
# Default to Ada (sm_89). Allow override via -DCMAKE_CUDA_ARCHITECTURES or env GPUQUEUE_CUDA_ARCHS
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{GPUQUEUE_CUDA_ARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{GPUQUEUE_CUDA_ARCHS})
  else()
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)
# Either use CUDA language directly or find CUDAToolkit
# find_package(CUDAToolkit REQUIRED)

# Core library
add_library(gpuqueue_core
  src/cpp/queue_core.cpp
  src/cuda/queue_kernels.cu
)

set_target_properties(gpuqueue_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Python extension module
pybind11_add_module(_core
  src/cpp/bindings.cpp
)

# Link CUDA runtime and core
# target_link_libraries(_core PRIVATE CUDA::cudart)  # if find_package(CUDAToolkit)
target_link_libraries(_core PRIVATE gpuqueue_core)

# Place extension under gpuqueue/_core.so
set_target_properties(_core PROPERTIES
  OUTPUT_NAME "_core"
)
```

Key points:
- `CMAKE_CUDA_ARCHITECTURES` defaults to `89` (RTX 4070 Ti Super). Users can override.
- Use `pybind11_add_module` to build the Python extension.
- Keep `gpuqueue_core` separate to enable potential reuse/testing.


## Binding Stub (illustrative)

`src/cpp/bindings.cpp` (to be implemented later):
```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "gpuqueue core bindings";
  // TODO: expose init(), enqueue(), try_dequeue_result(), stats(), shutdown()
}
```


## Local Developer Workflow

- Prereqs: NVIDIA driver, CUDA Toolkit 12.6+, Python ≥3.10, CMake ≥3.24, Ninja ≥1.11
- Commands:
  - Editable dev install (no wheel): `pip install -e ".[dev]"`
  - With Track A extras: `pip install -e ".[dev,track-a]"`
  - Build wheel/sdist locally: `python -m build` (requires `build` package)
  - Run tests: `pytest -q`


## Wheel Strategy

- Target `manylinux2014_x86_64`. Do NOT bundle NVIDIA drivers.
- CUDA runtime: expect users to have a compatible driver/runtime. Document in `README.md` and `runbook.md`.
- Provide prebuilt wheels for CPython 3.10–3.12 on Linux.
- Source distribution (`sdist`) supported for custom builds.


## CI Strategy (sketch)

- Use `cibuildwheel` for wheels; use self-hosted Linux runners with NVIDIA GPUs for test jobs.
- Example GitHub Actions job (illustrative):
```yaml
name: Wheels
on: [push, pull_request, workflow_dispatch]
jobs:
  build_wheels:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: pypa/cibuildwheel@v2.16.2
        env:
          CIBW_BUILD: "cp310-* cp311-* cp312-*"
          CIBW_SKIP: "*musllinux*"
          # For GPU builds/tests, prefer a separate test job on a GPU runner
```

- GPU test job: runs on a self-hosted runner with `nvidia-docker`, executes unit/integration tests and selected benchmarks.


## Track A (Redis) Dependency Risks & Mitigations

- Redis server requirement: Not bundled. Users must run Redis separately (local or container). Document via `runbook.md`.
- Client libraries: Keep optional via `extras_require` (`track-a`). Pin conservative versions and test against them.
- Network and ops complexity: Provide Docker Compose recipes for local dev. Encourage containerized workflows for Track A.
- Packaging bloat: Avoid bundling Redis or heavy deps by default.
- ABI/compat: Core wheel remains independent of Redis; only Python layer imports Redis when extras installed.

Install examples:
- Core only: `pip install gpuqueue`
- With Track A tools: `pip install "gpuqueue[track-a]"`


## Versioning and Compatibility

- Semantic versioning. Start with `0.y.z` until API stabilizes.
- Pin minimum CUDA Toolkit in docs; check runtime GPU capability at import or `init()`.
- Default arch: `sm_89`; allow user override to include PTX fallback if desired.


## Publishing

- Build: `python -m build`
- Verify wheels: `auditwheel show dist/*.whl`
- Upload: `twine upload dist/*`


## Future Extensions

- Multi-GPU build variants; wheel tags per-arch or fat binaries (PTX inclusion for forward-compat at size cost).
- macOS/Windows evaluation (out-of-scope initially).
- Optional CuPy or NumPy adapters in the Python layer for zero-copy interfaces.
