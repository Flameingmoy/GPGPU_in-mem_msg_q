# CUDA GPU-Resident Message Queue — Testing Plan

Status: Active. Focus: correctness, reliability (at-least-once), and concurrency safety.

---

## 1) Test Framework & Directory Structure

### Frameworks
- **C++ Unit Tests**: Google Test (gtest) — header-only, CMake-friendly
- **CUDA Device Tests**: gtest + custom CUDA test harness
- **Python Tests**: pytest with pytest-benchmark
- **Sanitizers**: NVIDIA Compute Sanitizer (memcheck, racecheck, synccheck)

### Directory Structure
```
tests/
├── cpp/                    # Host-side C++ unit tests
│   ├── test_ring_math.cpp
│   ├── test_config.cpp
│   └── test_error_codes.cpp
├── cuda/                   # Device-side CUDA tests
│   ├── test_device_query.cu
│   ├── test_atomics.cu
│   ├── test_fences.cu
│   └── test_slot_state.cu
├── integration/            # End-to-end tests
│   ├── test_enqueue_dequeue.cpp
│   ├── test_backpressure.cpp
│   └── test_shutdown.cpp
├── python/                 # Python API tests
│   ├── test_api.py
│   ├── test_exceptions.py
│   └── test_benchmark.py
└── CMakeLists.txt          # Test build configuration
```

### Naming Convention
- Test files: `test_<component>.{cpp,cu,py}`
- Test functions: `TEST(<Suite>, <Case>)` for gtest, `test_<description>` for pytest
- Fixtures: `<Component>Fixture` class

---

## 2) Invariants (Must-Hold Properties)

These invariants must be verified by every correctness test:

| ID | Invariant | Verification Method |
|----|-----------|---------------------|
| INV-1 | `0 ≤ (head - tail) ≤ capacity` (mod 2^64) | Assertions in enqueue/dequeue |
| INV-2 | Slot state machine: `EMPTY→READY→INFLIGHT→DONE→EMPTY` (no skips) | State transition logging + fuzzing |
| INV-3 | Publish fence before READY; complete fence before DONE | Sanitizer racecheck |
| INV-4 | `__threadfence_system()` before host-visible writes | Host read verification after event |
| INV-5 | FIFO ordering for single consumer | Sequence number verification |
| INV-6 | No double-claim of slots (atomicCAS exclusivity) | Concurrent claim test |
| INV-7 | No memory leaks after shutdown | cuda-memcheck + valgrind |

---

## 3) Host Unit Tests (C++/gtest)

### test_ring_math.cpp
```cpp
TEST(RingMath, IndexWrapPowerOfTwo) {
    // Verify (i & (capacity-1)) == i % capacity for power-of-two
    for (uint32_t cap : {256, 1024, 4096, 65536}) {
        for (uint64_t i = 0; i < cap * 3; ++i) {
            EXPECT_EQ(i & (cap - 1), i % cap);
        }
    }
}

TEST(RingMath, FullEmptyDetection) {
    // head == tail → empty
    // head - tail == capacity → full
}

TEST(RingMath, WrapAroundU64) {
    // Test near UINT64_MAX to verify no overflow bugs
}
```

### test_config.cpp
- Validate `QueueConfig` constraints: capacity must be power-of-two, slot_bytes > 0
- Reject invalid configs with proper error codes

### test_error_codes.cpp
- Verify all `QueueStatus` codes are distinct
- Verify `q_last_error()` returns meaningful strings

---

## 4) Device Unit Tests (CUDA/gtest)

### test_device_query.cu
```cuda
TEST(DeviceQuery, ComputeCapability) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    EXPECT_GE(prop.major * 10 + prop.minor, 89);  // sm_89 or higher
}
```

### test_atomics.cu
- `atomicCAS` claim exclusivity: N threads race to claim; exactly one succeeds
- `atomicAdd` counter consistency: N increments → final value = N

### test_fences.cu
```cuda
// Producer-consumer fence correctness (see NVIDIA guide)
__global__ void test_fence_visibility(int* flag, int* data) {
    if (threadIdx.x == 0) {  // Consumer
        while (atomicAdd(flag, 0) == 0);
        __threadfence();
        EXPECT_EQ(*data, 42);  // Must see producer's write
    } else if (threadIdx.x == 1) {  // Producer
        *data = 42;
        __threadfence();
        atomicExch(flag, 1);
    }
}
```

### test_slot_state.cu
- State machine: EMPTY→READY valid, EMPTY→INFLIGHT invalid
- Concurrent state transitions: no lost updates

---

## 5) Integration Tests

### test_enqueue_dequeue.cpp
1. Init queue with capacity=1024, slot_bytes=512
2. Enqueue N=500 messages with sequential payloads
3. Wait for processing (poll stats until deq_count == N)
4. Dequeue all results, verify:
   - All message IDs present
   - Payloads correctly processed
   - FIFO order preserved

### test_backpressure.cpp
1. Init queue with capacity=64 (small)
2. Enqueue at rate >> consumer rate
3. Verify `Q_ERR_FULL` returned (not hang)
4. Verify `dropped_full` counter incremented
5. After consumer catches up, enqueue resumes

### test_shutdown.cpp
1. Init queue, enqueue 100 messages
2. Call shutdown() while ~50 are INFLIGHT
3. Verify:
   - No crash
   - All INFLIGHT complete to DONE
   - Resources freed (no leaks)

### test_fault_injection.cpp
1. Configure `process_message` to fail for msg_id % 10 == 7
2. Verify retry up to N times
3. Verify DLQ receives failed messages after N retries

---

## 6) Sanitizer Verification

All tests should pass with sanitizers enabled:

```bash
# Memory errors (out-of-bounds, use-after-free)
compute-sanitizer --tool memcheck ./bin/tests

# Race conditions (concurrent access without sync)
compute-sanitizer --tool racecheck ./bin/tests

# Synchronization errors (barrier misuse)
compute-sanitizer --tool synccheck ./bin/tests

# Initialize check (uninitialized reads)
compute-sanitizer --tool initcheck ./bin/tests
```

### CI Integration
```yaml
sanitizer-tests:
  runs-on: [self-hosted, gpu]
  steps:
    - run: compute-sanitizer --tool memcheck --error-exitcode 1 ./bin/tests
    - run: compute-sanitizer --tool racecheck --error-exitcode 1 ./bin/tests
```

---

## 7) Python Tests (pytest)

### test_api.py
```python
import pytest
import gpuqueue as gq

def test_init_shutdown():
    gq.init(device=0)
    assert gq.core_version() is not None
    gq.shutdown()

def test_enqueue_basic():
    with gq.GPUQueue(capacity=256, slot_bytes=1024) as q:
        msg_id = q.enqueue(b"hello world")
        assert msg_id >= 0

def test_stats():
    with gq.GPUQueue(capacity=256, slot_bytes=1024) as q:
        stats = q.stats()
        assert stats.queue_depth == 0
```

### test_exceptions.py
```python
def test_invalid_config():
    with pytest.raises(ValueError):
        gq.GPUQueue(capacity=100, slot_bytes=1024)  # Not power of two

def test_payload_too_large():
    with gq.GPUQueue(capacity=256, slot_bytes=512) as q:
        with pytest.raises(gq.QueueFullError):
            q.enqueue(b"x" * 1024)  # Exceeds slot_bytes
```

### test_benchmark.py
```python
import pytest

@pytest.mark.benchmark(group="enqueue")
def test_enqueue_throughput(benchmark):
    with gq.GPUQueue(capacity=4096, slot_bytes=512) as q:
        payload = b"x" * 256
        benchmark(q.enqueue, payload)
```

---

## 8) Performance / Soak Tests

### Throughput Test
- Measure messages/sec for payloads: 256B, 512B, 1KB, 2KB, 4KB
- Record p50/p95/p99 latency
- Capture Nsight Systems trace for pipeline analysis

### Soak Test Procedure
1. Run for 60 minutes with steady 10k msg/s rate
2. Monitor:
   - GPU memory usage (should be stable)
   - Queue depth (should oscillate, not grow unbounded)
   - Latency (p99 should remain stable)
   - CPU usage (backpressure shouldn't cause spin)
3. Verify at end:
   - `enq_count == deq_count` (no lost messages)
   - No CUDA errors logged
   - Clean shutdown

### Memory Leak Detection
```bash
# CUDA memory leaks
cuda-memcheck --leak-check full ./bin/soak_test

# Host memory leaks
valgrind --leak-check=full --show-leak-kinds=all ./bin/soak_test
```

---

## 9) CI/CD Test Matrix

| Test Level | Trigger | Runner | Timeout |
|------------|---------|--------|---------|
| Lint (ruff, clang-format) | Every push | ubuntu-22.04 | 5m |
| Host unit tests | Every push | ubuntu-22.04 | 10m |
| CUDA unit tests | Every push | self-hosted GPU | 15m |
| Integration tests | PR + main | self-hosted GPU | 30m |
| Sanitizer tests | Nightly | self-hosted GPU | 60m |
| Soak tests | Weekly | self-hosted GPU | 90m |

### CTest Labels
```bash
ctest -L unit          # All unit tests
ctest -L integration   # Integration tests
ctest -L cuda          # CUDA-specific tests
ctest -L sanitizer     # With sanitizer wrappers
```

---

## 10) Test Artifacts

### Output Locations
- Logs: `build/test_logs/` with timestamps
- JUnit XML: `build/test_results/*.xml` (for CI)
- Nsight traces: `build/profiles/*.nsys-rep`
- Coverage: `build/coverage/` (if gcov enabled)

### Failure Investigation
1. Check `test_logs/<test_name>.log` for stderr
2. Run failed test with `--gtest_filter=<Suite>.<Case>`
3. For CUDA failures, re-run with `compute-sanitizer`
4. For race conditions, increase thread count and iterations

---

## 11) References
- `docs/design.md` §4 (fences), §6 (reliability)
- NVIDIA Compute Sanitizer User Guide
- NVIDIA CUDA C++ Programming Guide: Memory Fence Functions
- Google Test Primer: https://google.github.io/googletest/primer.html
- pytest documentation: https://docs.pytest.org/
