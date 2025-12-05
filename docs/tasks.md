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

## Milestone M2 — Track B Core: Ring Buffer & Persistent Kernel ✅ COMPLETE
**Goal**: Implement the GPU-resident queue core (prioritize Track B for innovation value).

### M2.1 — Ring Buffer Data Structures
- [x] Define `SlotHeader` struct: `{ uint32_t len; uint32_t flags; uint64_t msg_id; }` (16B aligned)
- [x] Define `SlotState` enum: `EMPTY=0, READY=1, INFLIGHT=2, DONE=3`
- [x] Implement host-side `RingBuffer` class:
  - Allocate device arrays: `d_slot_data[capacity * slot_size]`, `d_states[capacity]`
  - Allocate control block in pinned memory (head, tail, stop_flag, stats)
- [x] Unit test: allocation/deallocation, index wrap math `(i & (capacity-1))`

### M2.2 — Host Enqueue Path
- [x] Implement `enqueue()`: reserve slot → async H2D copy → publish READY
- [ ] Implement `enqueue_batch()`: batch multiple messages for throughput (deferred to M4)
- [x] Use `std::atomic` for control block counters (pinned memory)
- [x] Correct fence pattern: H2D copy completes (stream sync) → set slot_state=READY
- [x] Handle ERR_FULL with backpressure (timeout or return immediately)

### M2.3 — Persistent Kernel Consumer
- [x] Implement persistent kernel loop with atomicCAS state transitions
- [x] `volatile` slot states for cross-stream visibility
- [x] `__threadfence_system()` for host-visible memory ordering
- [x] `__nanosleep()` backoff when idle

### M2.4 — Result Dequeue & Completion
- [x] Implement `try_dequeue_result()`: check slot DONE, copy result D2H
- [x] Implement `poll_completions()`: scan DONE slots, return msg_ids
- [x] Use non-default streams for all memcpy (critical: default stream blocks on persistent kernel!)

### M2.5 — Shutdown & Error Handling
- [x] Graceful shutdown: set stop_flag, kernel exits loop
- [x] Implement `stats()`: enqueue_count, dequeue_count, processed
- [x] CUDA error propagation via CUDA_CHECK macro

### M2.6 — Integration Tests
- [x] `test_queue_integration.cu`: init/shutdown, single message, multiple messages, poll, throughput
- [x] All 5 integration tests passing
- [x] Performance: 172k msg/s, 42 MB/s throughput on RTX 4070 Ti Super

Exit criteria: correctness tests pass ✅; `compute-sanitizer --tool racecheck` pending.

---

## Milestone M3 — Track A: Redis-Backed MVP (Validation for Track B) ✅ COMPLETE
**Goal**: Validate the GPU processing pipeline with Redis as external broker. This serves as:
1. **Validation harness** for Track B's GPU kernel and memory patterns
2. **Comparison baseline** for performance benchmarking
3. **Reference implementation** for at-least-once semantics

> **Note**: Track A is NOT the production target. Track B (GPU-resident queue) is the primary deliverable.

### M3.1 — Redis Infrastructure
- [x] `docker/docker-compose.yml`: Redis 7.x with persistence disabled
- [x] Python client: `redis-py>=5.0` with `hiredis` for speed
- [x] Verify connectivity: PING, SET/GET, XADD/XREAD (scripts/test_redis.py)
- [x] Add `redis`, `numpy` to `pyproject.toml` optional dependencies (`[track-a]`)

### M3.2 — Producer/Consumer Pattern
- [x] `src/python/gpuqueue/track_a/producer.py`:
  - Rate-limited message generation
  - XADD to Redis Stream with automatic trimming
  - Batch send with pipelining (287k msg/s)
- [x] `src/python/gpuqueue/track_a/consumer.py`:
  - XREADGROUP with consumer group for at-least-once
  - Batch fetch with configurable size and timeout
  - XAUTOCLAIM for pending message recovery
- [x] Use pinned memory for staging buffer (CuPy)

### M3.3 — GPU Processing Bridge
- [x] `src/python/gpuqueue/track_a/gpu_processor.py`:
  - Accept Batch from Consumer
  - Pack payloads into fixed-size slots
  - H2D transfer via CuPy async stream
  - Echo kernel (copy input to output)
  - D2H transfer to host
- [x] Pinned memory staging for faster transfers
- [x] NumPy fallback for testing without GPU

### M3.4 — Reliability & Comparison
- [x] At-least-once: XREADGROUP + XACK pattern implemented
- [x] Claim pending: XAUTOCLAIM for timed-out messages
- [x] Comparison tests: `tests/integration/test_comparison.py`
- [x] Performance baselines documented:
  - Producer: 42k msg/s single, 287k msg/s batched
  - Full pipeline: 191k msg/s with GPU
  - Latency: p50=0.14ms, p95=0.24ms

Exit criteria: Track A pipeline working ✅; comparison data collected ✅; validates Track B patterns ✅

---

## Milestone M4 — Python API and Packaging ✅ COMPLETE
**Goal**: Pythonic API with pip-installable wheel.

### M4.1 — PyBind11 Bindings (Track B API)
- [x] Bind: `GpuQueue` class with `enqueue()`, `try_dequeue_result()`, `poll_completions()`, `stats()`, `shutdown()`
- [x] Expose `QueueConfig`, `QueueStats`, `QueueStatus` as Python types
- [x] Exception wrapping for errors (ValueError on failures)

### M4.2 — Python Wrapper
- [x] Context manager: `with GpuQueue(QueueConfig(capacity=1024, slot_bytes=512)) as q:`
- [x] bytes/buffer protocol integration for payloads
- [ ] Async-friendly API with `asyncio` compatibility (future)

### M4.3 — Packaging & Distribution
- [x] `pyproject.toml` with scikit-build-core
- [x] Local `pip install -e .` working
- [ ] cibuildwheel config for manylinux2014_x86_64 (deferred to M6)

### M4.4 — Comparison Tests
- [x] Track B benchmarks: enqueue throughput, full roundtrip
- [x] Track A vs Track B comparison tests
- [x] Results: Track B is **2x faster** than Track A (73k vs 36k msg/s)
- [x] Latency: Track B p50=0.13ms vs Track A p50=0.14ms

Exit criteria: `pip install -e .` works ✅; example script runs ✅; comparison tests pass ✅

---

## Milestone M5 — Testing, Benchmarking, and Soak
**Goal**: Comprehensive test coverage and performance baselines.

### M5.0 — Hardware Monitoring
- [x] `scripts/hardware_monitor.py`: Real-time CPU, RAM, GPU, VRAM monitoring
  - psutil for CPU/RAM metrics
  - pynvml/nvidia-smi for GPU metrics
  - CSV export for analysis
- [x] `scripts/benchmark_with_monitor.py`: Benchmarks with resource tracking
  - Track A and Track B throughput/latency
  - Hardware summary (avg/max utilization)
- [x] Example output:
  - CPU: ~19% avg, 24% max
  - RAM: 7.4 GB (24%)
  - GPU: 23% util, 1.2 GB VRAM
  - Track B: 65k msg/s, Track A: 38k msg/s (1.7x speedup)

### M5.1 — Unit Tests
- [ ] Host tests: index math, config validation, error codes (gtest)
- [ ] Device tests: slot state machine, fence correctness, atomic ops
- [x] Python tests: `tests/python/test_api.py` (27 tests)
  - QueueConfig validation
  - GpuQueue lifecycle (init, shutdown, context manager)
  - Enqueue/dequeue roundtrip
  - Error handling (payload too large, shutdown)
  - Buffer protocol (bytes, numpy, memoryview)
- [x] Track A tests: `tests/python/test_track_a.py` (15 tests)
  - Producer/Consumer creation and operations
  - Batch processing
  - End-to-end pipeline

### M5.2 — Sanitizer Verification ✅
- [x] `scripts/run_sanitizers.sh`: Automated sanitizer runner
- [x] memcheck: 0 memory access errors
- [x] racecheck: 0 race condition hazards
- [x] initcheck: 0 uninitialized memory errors
- [x] synccheck: 0 synchronization errors

### M5.3 — Integration Tests
- [x] End-to-end: enqueue N messages → process → dequeue all → verify (test_queue_integration)
- [ ] Stress test: max rate, full queue, backpressure
- [ ] Fault injection: kernel assert, timeout, restart

### M5.3 — Benchmarks
- [ ] Throughput vs payload size (256B, 512B, 1KB, 2KB, 4KB)
- [ ] Latency histogram (p50/p95/p99)
- [ ] GPU utilization and overlap metrics (Nsight Systems)
- [ ] Comparison: Track A vs Track B

### M5.4 — Soak Tests ✅
- [x] `scripts/soak_test.py`: Long-running stability test
- [x] 30-second test: 12,050 messages, 0 errors, ~397 msg/s steady
- [x] Memory stability: VRAM delta +200MB (expected queue allocation), no leaks
- [x] Performance consistency: min=393 msg/s, max=398 msg/s (stable)
- [x] Queue cleanup: depth=0 at end, all slots properly recycled

Exit criteria: test pass rate >95%; baselines documented; no memory leaks.

---

## Milestone M6 — CI/CD and Release
**Goal**: Automated builds, tests, and releases.

### M6.1 — Lint & Format CI ✅
- [x] `.github/workflows/ci.yml`: Main CI workflow
- [x] ruff linter + formatter check
- [x] mypy type checking (warn-only)
- [x] clang-format C++ style check
- [x] `.clang-format`: Google-based C++ style
- [x] `pyproject.toml`: ruff, mypy, pytest config

### M6.2 — Build & Test CI
- [x] CUDA toolkit install via Jimver/cuda-toolkit action
- [x] Build package with scikit-build-core
- [x] Python 3.10/3.11/3.12 matrix
- [ ] CPU-only pytest subset

### M6.3 — GPU Runner
- [ ] Self-hosted runner with NVIDIA GPU (optional)
- [ ] Manual workflow dispatch for GPU tests

### M6.4 — Wheels & PyPI
- [ ] cibuildwheel configuration
- [ ] TestPyPI upload
- [ ] Release workflow: tag → build → PyPI

### M6.5 — Documentation
- [ ] mkdocs site
- [ ] GitHub Pages deployment

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
