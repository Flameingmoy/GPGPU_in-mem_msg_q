# Project Backlog and Milestones

This backlog tracks both Track A (Redis-backed MVP) and Track B (GPU VRAM-resident queue). Use this as the source of truth for execution order and status.

Legend: [ ] todo, [x] done, [~] in-progress, [!] blocked


## Milestone M0 — Documentation Foundations
- [x] `docs/design.md` (architecture, queues, API surface)
- [x] `docs/research.md` (CUDA notes, references)
- [x] `docs/api.md` (host + kernel plugin interface)
- [x] `docs/runbook.md` (ops, quickstart, troubleshooting)
- [x] `docs/testing.md` (test strategy, invariants)
- [x] `docs/benchmarking.md` (metrics, workloads, procedures)
- [x] `docs/use_cases.md` (scenarios, acceptance)
- [x] `docs/packaging.md` (Python packaging plan, Track A risks)

Exit criteria: docs reviewed and linked from repo README (TBD).


## Milestone M1 — Environment Setup
- [ ] Verify NVIDIA driver + CUDA Toolkit 12.6+ availability
- [ ] Script: device property query (compute capability 8.9)
- [ ] Build skeleton (CMake scaffolding for C++/CUDA, PyBind11)
- [ ] "Hello, CUDA" kernel + unit test
- [ ] Host↔Device mem management wrappers + h2d/d2h transfer test

Exit criteria: local build runs unit tests, GPU detected and usable.


## Milestone M2 — Track A (Redis-backed MVP)
- [ ] Local Redis (container or package) with config and healthcheck
- [ ] Host producer: enqueue to Redis List/Stream; backpressure policy
- [ ] Host consumer/worker: dequeue → stage in pinned buffer → async H2D copy
- [ ] GPU kernel(s): simple processing path using CUDA streams
- [ ] At-least-once delivery; idempotent processing demo
- [ ] Observability: queue depth, throughput, latency histograms
- [ ] Integration tests covering ordering and retry paths

Exit criteria: sustained N msgs/s with documented latency on dev box; resiliency under restarts.


## Milestone M3 — Track B (GPU-Resident Queue, Stage-2)
- [ ] Device ring buffer in global memory; fixed-size slots and metadata
- [ ] Persistent kernel that scans READY → processes → marks DONE
- [ ] Host API: `init()`, `enqueue_async()`, `try_dequeue_result()`, `stats()`, `shutdown()`
- [ ] Correct atomic + fence protocol (`__threadfence`/`__threadfence_system`)
- [ ] Multi-producer (host) support; single consumer kernel (initial)
- [ ] Backpressure + timeouts; metrics on stalls and occupancy
- [ ] Fault injection + sanitizer runs (Compute Sanitizer)

Exit criteria: correctness under stress; basic performance targets achieved.


## Milestone M4 — Python API and Packaging
- [ ] PyBind11 bindings for core API
- [ ] Pythonic wrapper: context manager, type conversions, exceptions
- [ ] `pyproject.toml` with scikit-build-core
- [ ] Wheels (manylinux2014) for CPython 3.10–3.12
- [ ] Optional extras: `track-a` for Redis client tooling
- [ ] Example notebooks/scripts showing enqueue/process/dequeue

Exit criteria: `pip install gpuqueue` usable locally; examples run on target GPU.


## Milestone M5 — Testing, Benchmarking, and Soak
- [ ] Unit tests (host+device), integration tests
- [ ] Benchmark suites for throughput/latency (Track A vs Track B)
- [ ] Soak tests with backpressure + restarts
- [ ] Automated reports and charts

Exit criteria: performance baselines documented; regressions detectable.


## Milestone M6 — CI/CD and Release
- [ ] GitHub Actions: lint, unit, integration (CPU-only)
- [ ] GPU test workflow on self-hosted runner (integration + perf smoke)
- [ ] cibuildwheel for wheels; auditwheel validation
- [ ] Pre-release `0.1.0a` to TestPyPI
- [ ] Promotion to PyPI with release notes and docs links

Exit criteria: repeatable builds, published artifacts, minimal manual steps.


## Nice-to-Haves and Future Work
- [ ] Multi-GPU support; NUMA-aware host staging
- [ ] Variable-size messages; slab allocator
- [ ] Exactly-once semantics; compaction and persistence
- [ ] Plugin kernels registry; dynamic loading
- [ ] CuPy/NumPy zero-copy adapters in Python API


## References
- `docs/design.md`, `docs/api.md`, `docs/runbook.md`
- NVIDIA CUDA Programming Guide and Samples
- Redis Streams and client libraries
