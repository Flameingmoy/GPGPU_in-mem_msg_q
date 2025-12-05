# CUDA GPU-Resident Message Queue — Testing Plan

Status: Active. Focus: correctness, reliability (at-least-once), and concurrency safety.

**Testing Philosophy:**
- **Track B (GPU-Resident)** is the primary deliverable — extensive unit, integration, and soak tests
- **Track A (Redis-Backed)** is a validation harness — used to verify GPU patterns and provide comparison baselines
- All tests should be automatable for CI/CD

---

## 1) Test Framework & Directory Structure

### Frameworks
- **C++ Unit Tests**: Google Test (gtest) — header-only, CMake-friendly
- **CUDA Device Tests**: gtest + custom CUDA test harness
- **Python Tests**: pytest with pytest-benchmark, pytest-asyncio
- **Sanitizers**: NVIDIA Compute Sanitizer (memcheck, racecheck, synccheck)
- **Mocking**: FakeRedis for Track A tests without Redis server

### Directory Structure
```
tests/
├── cpp/                    # Host-side C++ unit tests
│   ├── test_ring_math.cpp          ✅ Implemented
│   ├── test_config.cpp
│   └── test_error_codes.cpp
├── cuda/                   # Device-side CUDA tests
│   ├── test_device_query.cu        ✅ Implemented
│   ├── test_queue_integration.cu   ✅ Implemented (5 tests passing)
│   ├── test_atomics.cu
│   ├── test_fences.cu
│   └── test_slot_state.cu
├── integration/            # End-to-end tests (Python)
│   ├── test_track_b_api.py         # Track B Python API
│   ├── test_track_a_pipeline.py    # Track A Redis pipeline
│   └── test_comparison.py          # Track A vs B comparison
├── python/                 # Python unit tests
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

## 5) Integration Tests — Track B (GPU-Resident Queue)

> These tests validate the primary deliverable: the GPU-resident message queue.

### test_queue_integration.cu ✅ IMPLEMENTED
Already implemented with 5 passing tests:
1. **Init/Shutdown**: Queue lifecycle management
2. **Single Message Roundtrip**: Enqueue → process → dequeue
3. **Multiple Messages**: 50 messages, verify all processed
4. **Poll Completions**: Scan DONE slots, return msg_ids
5. **Throughput**: 1000 messages, measure msg/s and MB/s

### test_backpressure.cpp
1. Init queue with capacity=64 (small)
2. Enqueue at rate >> consumer rate
3. Verify `ERR_FULL` or `ERR_TIMEOUT` returned (not hang)
4. Verify `dropped_full` counter incremented
5. After consumer catches up, enqueue resumes

### test_shutdown.cpp
1. Init queue, enqueue 100 messages
2. Call shutdown() while ~50 are INFLIGHT
3. Verify:
   - No crash
   - All INFLIGHT complete to DONE
   - Resources freed (no leaks)

### test_concurrent_enqueue.cpp
1. Launch N producer threads, each enqueuing M messages
2. Verify all N*M messages processed exactly once
3. Verify no duplicate msg_ids
4. Verify stats consistency

---

## 6) Integration Tests — Track A (Redis Validation)

> Track A tests validate the GPU processing patterns using Redis as a broker.
> These tests also provide comparison baselines for Track B performance.

### test_track_a_pipeline.py
```python
import pytest
from gpuqueue.track_a import Producer, Consumer, GpuProcessor

@pytest.fixture
def redis_stream():
    """Setup Redis stream, cleanup after test."""
    import redis
    r = redis.Redis()
    stream_name = "test_stream"
    r.delete(stream_name)
    yield stream_name
    r.delete(stream_name)

def test_producer_consumer_roundtrip(redis_stream):
    """Basic message flow through Redis → GPU → back."""
    producer = Producer(redis_stream)
    consumer = Consumer(redis_stream, group="test_group")
    processor = GpuProcessor()
    
    # Produce messages
    msg_ids = [producer.send(f"msg_{i}".encode()) for i in range(10)]
    
    # Consume and process
    batch = consumer.fetch_batch(max_messages=10, timeout_ms=1000)
    results = processor.process(batch)
    consumer.ack(batch)
    
    assert len(results) == 10

def test_at_least_once_delivery(redis_stream):
    """Verify messages are redelivered on consumer failure."""
    producer = Producer(redis_stream)
    consumer1 = Consumer(redis_stream, group="test_group", consumer_id="c1")
    consumer2 = Consumer(redis_stream, group="test_group", consumer_id="c2")
    
    producer.send(b"critical_message")
    
    # Consumer 1 fetches but doesn't ACK (simulating crash)
    batch = consumer1.fetch_batch(max_messages=1, timeout_ms=1000)
    assert len(batch) == 1
    # No ack - consumer "crashes"
    
    # Consumer 2 claims pending message after timeout
    pending = consumer2.claim_pending(min_idle_ms=100)
    assert len(pending) == 1

def test_batch_gpu_processing(redis_stream):
    """Verify batch processing matches Track B patterns."""
    producer = Producer(redis_stream)
    processor = GpuProcessor(batch_size=64, slot_bytes=512)
    
    # Send batch
    for i in range(64):
        producer.send(f"payload_{i:04d}".encode())
    
    consumer = Consumer(redis_stream, group="test_group")
    batch = consumer.fetch_batch(max_messages=64, timeout_ms=1000)
    
    # Process should use pinned memory + async H2D
    results = processor.process(batch)
    
    assert len(results) == 64
    # Verify GPU was actually used (check stats)
    stats = processor.stats()
    assert stats.gpu_batches_processed == 1
```

### test_comparison.py
```python
import pytest
import time
from gpuqueue import GpuQueue  # Track B
from gpuqueue.track_a import Producer, Consumer, GpuProcessor  # Track A

@pytest.fixture
def track_b_queue():
    q = GpuQueue(capacity=1024, slot_bytes=512)
    yield q
    q.shutdown()

def test_throughput_comparison(track_b_queue, redis_stream):
    """Compare throughput: Track A vs Track B."""
    num_messages = 1000
    payload = b"x" * 256
    
    # Track B throughput
    start = time.perf_counter()
    for i in range(num_messages):
        track_b_queue.enqueue(payload)
    # Wait for processing
    while track_b_queue.stats().processed < num_messages:
        time.sleep(0.001)
    track_b_time = time.perf_counter() - start
    track_b_rate = num_messages / track_b_time
    
    # Track A throughput
    producer = Producer(redis_stream)
    consumer = Consumer(redis_stream, group="test_group")
    processor = GpuProcessor()
    
    start = time.perf_counter()
    for i in range(num_messages):
        producer.send(payload)
    
    processed = 0
    while processed < num_messages:
        batch = consumer.fetch_batch(max_messages=64, timeout_ms=100)
        if batch:
            processor.process(batch)
            consumer.ack(batch)
            processed += len(batch)
    track_a_time = time.perf_counter() - start
    track_a_rate = num_messages / track_a_time
    
    print(f"Track A: {track_a_rate:.0f} msg/s")
    print(f"Track B: {track_b_rate:.0f} msg/s")
    print(f"Track B speedup: {track_b_rate/track_a_rate:.1f}x")
    
    # Track B should be significantly faster (no Redis overhead)
    assert track_b_rate > track_a_rate * 2, "Track B should be >2x faster"

def test_latency_comparison(track_b_queue, redis_stream):
    """Compare p50/p95/p99 latency: Track A vs Track B."""
    # Implementation: measure individual message round-trip times
    pass
```

### test_redis_streams.py (Unit tests for Redis integration)
```python
import pytest
from unittest.mock import MagicMock

def test_xadd_message_format():
    """Verify message format matches Track B slot format."""
    pass

def test_consumer_group_creation():
    """Verify consumer group is created correctly."""
    pass

def test_xclaim_pending_messages():
    """Verify pending messages can be claimed after timeout."""
    pass

def test_dead_letter_queue():
    """Verify failed messages move to DLQ after N retries."""
    pass
```

---

## 7) Sanitizer Verification

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

## 8) Python Tests (pytest)

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

## 9) Performance / Soak Tests

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

## 10) CI/CD Test Matrix

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

## 11) Test Artifacts

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

## 12) References
- `docs/design.md` §4 (fences), §6 (reliability)
- NVIDIA Compute Sanitizer User Guide
- NVIDIA CUDA C++ Programming Guide: Memory Fence Functions
- Google Test Primer: https://google.github.io/googletest/primer.html
- pytest documentation: https://docs.pytest.org/
