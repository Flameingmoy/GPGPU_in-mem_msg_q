# Project Backlog and Milestones

This backlog tracks both Track A (Redis-backed MVP) and Track B (GPU VRAM-resident queue). Use this as the source of truth for execution order and status.

Legend: [ ] todo, [x] done, [~] in-progress, [!] blocked

---

## Milestone M0 — Documentation Foundations
- [x] `docs/design.md` (architecture, queues, API surface)
- [x] `docs/research.md` (CUDA notes, references)
- [x] `docs/api.md` (host + kernel plugin interface)
- [x] `docs/runbook.md` (ops, quickstart, troubleshooting)
- [x] `docs/testing.md` (test strategy, invariants)
- [x] `docs/benchmarking.md` (metrics, workloads, procedures)
- [x] `docs/use_cases.md` (scenarios, acceptance)
- [x] `docs/packaging.md` (Python packaging plan, Track A risks)

Exit criteria: docs reviewed and linked from repo README.

---

## Milestone M1 — Environment & Build Verification
**Goal**: Prove the toolchain works end-to-end with actual GPU execution.

### M1.1 — Environment Verification
- [x] Verify NVIDIA driver ≥535 + CUDA Toolkit 12.6+ installed
- [x] Create `scripts/check_env.sh`: validate `nvcc --version`, `nvidia-smi`, driver/toolkit compat
- [x] Query device properties via CUDA API (confirm sm_89 / CC 8.9)

### M1.2 — Build System Hardening
- [x] CMakeLists.txt with CUDA + pybind11 (scaffold exists)
- [x] Add `CMAKE_CUDA_STANDARD 17` for libcu++ atomics
- [x] Add debug/release build variants with proper flags (`-G -lineinfo` for debug)
- [x] Fix `CUDA_SEPARABLE_COMPILATION` (only enable if needed for device linking)
- [x] Validate build: `cmake --build . --target all`

### M1.3 — Hello CUDA Kernel
- [x] Implement `tests/cuda/test_device_query.cu`: print GPU name, SM count, compute cap
- [x] Implement simple vector add kernel to verify H2D/D2H transfers
- [x] Add Google Test (gtest) framework for C++ tests
- [ ] Verify with `compute-sanitizer --tool memcheck ./test_device_query`

### M1.4 — Memory Management Utilities
- [x] Implement `include/gpuqueue/memory.hpp`: pinned alloc, device alloc, async copy wrappers
- [x] Test H2D/D2H roundtrip with timing (baseline for PCIe bandwidth: ~24 GB/s)
- [x] Implement CUDA stream pool utility for multi-stream pipelining

Exit criteria: `ctest` passes all M1 tests; GPU detected and kernel executed successfully. ✅ COMPLETE

---

## Milestone M2 — Track B Core: Ring Buffer & Persistent Kernel
**Goal**: Implement the GPU-resident queue core (prioritize Track B for innovation value).

### M2.1 — Ring Buffer Data Structures
- [ ] Define `SlotHeader` struct: `{ uint32_t len; uint32_t flags; uint64_t msg_id; }` (16B aligned)
- [ ] Define `SlotState` enum: `EMPTY=0, READY=1, INFLIGHT=2, DONE=3`
- [ ] Implement host-side `RingBuffer` class:
  - Allocate device arrays: `d_headers[capacity]`, `d_payloads[capacity][slot_bytes]`, `d_states[capacity]`
  - Allocate control block in pinned memory (head, tail, stop_flag, stats)
- [ ] Unit test: allocation/deallocation, index wrap math `(i & (capacity-1))`

### M2.2 — Host Enqueue Path
- [ ] Implement `q_enqueue()`: reserve slot → async H2D copy → publish READY
- [ ] Implement `q_enqueue_batch()`: batch multiple messages for throughput
- [ ] Use `cuda::atomic` (libcu++) for control block counters
- [ ] Correct fence pattern: H2D copy completes (stream sync) → set slot_state=READY
- [ ] Handle Q_ERR_FULL with backpressure (spin with backoff or return immediately)

### M2.3 — Persistent Kernel Consumer
- [ ] Implement persistent kernel loop:
  ```cuda
  while (!stop_flag) {
      // Scan for READY slots
      // atomicCAS to claim INFLIGHT
      // Call process_message()
      // __threadfence() → set DONE → advance tail
      // Backoff with __nanosleep() when idle
  }
  ```
- [ ] Use cooperative groups for grid-wide coordination (optional for multi-block)
- [ ] Implement simple `process_message()`: copy/transform payload, write result

### M2.4 — Result Dequeue & Completion
- [ ] Implement `q_try_dequeue_result()`: check slot DONE, copy result D2H
- [ ] Implement completion queue pattern (host-visible ring of completed msg_ids)
- [ ] `__threadfence_system()` before signaling host-visible completion

### M2.5 — Shutdown & Error Handling
- [ ] Graceful shutdown: set stop_flag, wait for INFLIGHT→DONE, drain
- [ ] Implement `q_stats()`: counters, queue depth, error flags
- [ ] CUDA error propagation to host API

Exit criteria: correctness tests pass; `compute-sanitizer --tool racecheck` clean.

---

## Milestone M3 — Track A: Redis-Backed MVP (Parallel Track)
**Goal**: Validate batching and GPU processing with Redis as broker.

### M3.1 — Redis Setup
- [ ] Docker Compose for Redis 7.x
- [ ] Python client: redis-py with hiredis
- [ ] Verify connectivity and basic LPUSH/BRPOP

### M3.2 — Producer/Consumer Workers
- [ ] Host producer: enqueue to Redis List with rate limiting
- [ ] Host consumer: BRPOP → batch collect → pack to fixed-size blobs
- [ ] Stage in pinned buffer, async H2D, launch GPU kernel

### M3.3 — GPU Kernel Processing
- [ ] Simple processing kernel (e.g., checksum, transform)
- [ ] Multi-stream pipelining: overlap H2D/kernel/D2H across streams
- [ ] Measure achieved overlap with Nsight Systems

### M3.4 — Reliability
- [ ] Use RPOPLPUSH or Streams (XREADGROUP/XACK) for at-least-once
- [ ] Retry logic with backoff; dead-letter after N failures

Exit criteria: sustained throughput; at-least-once verified under worker restart.

---

## Milestone M4 — Python API and Packaging
**Goal**: Pythonic API with pip-installable wheel.

### M4.1 — PyBind11 Bindings (Track B API)
- [ ] Bind: `q_init()`, `q_enqueue()`, `q_enqueue_batch()`, `q_try_dequeue_result()`, `q_stats()`, `q_shutdown()`
- [ ] Expose `QueueConfig`, `QueueStats`, `QueueStatus` as Python types
- [ ] Exception wrapping for CUDA errors

### M4.2 — Python Wrapper
- [ ] Context manager: `with GPUQueue(capacity=4096, slot_bytes=2048) as q:`
- [ ] NumPy/bytes integration for payloads
- [ ] Async-friendly API with `asyncio` compatibility (future)

### M4.3 — Packaging & Distribution
- [x] `pyproject.toml` with scikit-build-core (scaffold exists)
- [ ] Fix wheel.packages path for scikit-build
- [ ] Local `pip install -e .` working
- [ ] cibuildwheel config for manylinux2014_x86_64

Exit criteria: `pip install gpuqueue` works; example script runs.

---

## Milestone M5 — Testing, Benchmarking, and Soak
**Goal**: Comprehensive test coverage and performance baselines.

### M5.1 — Unit Tests
- [ ] Host tests: index math, config validation, error codes (gtest)
- [ ] Device tests: slot state machine, fence correctness, atomic ops
- [ ] Python tests: pytest for API, error handling, edge cases

### M5.2 — Integration Tests
- [ ] End-to-end: enqueue N messages → process → dequeue all → verify
- [ ] Stress test: max rate, full queue, backpressure
- [ ] Fault injection: kernel assert, timeout, restart

### M5.3 — Benchmarks
- [ ] Throughput vs payload size (256B, 512B, 1KB, 2KB, 4KB)
- [ ] Latency histogram (p50/p95/p99)
- [ ] GPU utilization and overlap metrics (Nsight Systems)
- [ ] Comparison: Track A vs Track B

### M5.4 — Soak Tests
- [ ] 30-60 min continuous run; verify no leaks, stable latency
- [ ] Memory profiling with `cuda-memcheck`

Exit criteria: test pass rate >95%; baselines documented; no memory leaks.

---

## Milestone M6 — CI/CD and Release
**Goal**: Automated builds, tests, and releases.

- [ ] GitHub Actions: lint (ruff), format (clang-format), type check (mypy)
- [ ] CI: build + CPU-only unit tests on ubuntu-22.04
- [ ] GPU runner: integration tests, sanitizer runs (self-hosted or cloud GPU)
- [ ] cibuildwheel → upload to TestPyPI
- [ ] Release workflow: tag → build → publish to PyPI
- [ ] Documentation site (mkdocs or sphinx)

Exit criteria: green CI on main; published 0.1.0 release.

---

## Future Work (Post-MVP)
- [ ] Multi-GPU support; NUMA-aware pinned memory
- [ ] Variable-size messages with slab allocator
- [ ] Exactly-once semantics (idempotent processing + dedupe)
- [ ] Plugin kernel registry; dynamic .cubin loading
- [ ] CuPy/NumPy zero-copy via `__cuda_array_interface__`
- [ ] GPUDirect RDMA for network → VRAM path

---

## References
- `docs/design.md`, `docs/api.md`, `docs/runbook.md`
- NVIDIA CUDA C++ Programming Guide (memory fences, atomics)
- NVIDIA cuda-samples: asyncAPI, threadFenceReduction
- Redis Streams and hiredis
- scikit-build-core documentation
