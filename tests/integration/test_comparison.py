#!/usr/bin/env python3
"""
Track A vs Track B Comparison Tests.

Compares throughput and latency between:
- Track A: Redis Streams → GPU (CuPy) → back
- Track B: GPU-resident queue (C++ with pybind11) [when available]

For now, runs Track A benchmarks as baseline. Track B comparison
will be enabled once Python bindings are complete (M4).
"""

import pytest
import time
import statistics
from typing import Optional

# Skip if redis not available
redis = pytest.importorskip("redis")

from gpuqueue.track_a import Producer, Consumer, GpuProcessor


@pytest.fixture
def redis_client():
    """Get Redis client, skip if not available."""
    r = redis.Redis(host="localhost", port=6379)
    try:
        r.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not available")
    yield r
    r.close()


@pytest.fixture
def test_stream(redis_client):
    """Create a test stream, cleanup after."""
    stream_name = "gpuqueue:benchmark_stream"
    redis_client.delete(stream_name)
    try:
        redis_client.xgroup_destroy(stream_name, "benchmark_group")
    except redis.ResponseError:
        pass
    yield stream_name
    redis_client.delete(stream_name)


class TestTrackAThroughput:
    """Track A throughput benchmarks."""
    
    def test_producer_throughput(self, test_stream):
        """Measure producer-only throughput."""
        num_messages = 10000
        payload_size = 256
        payload = b"x" * payload_size
        
        producer = Producer(test_stream)
        
        start = time.perf_counter()
        for _ in range(num_messages):
            producer.send(payload)
        elapsed = time.perf_counter() - start
        
        rate = num_messages / elapsed
        throughput_mb = (num_messages * payload_size) / (1024 * 1024) / elapsed
        
        print(f"\nProducer throughput:")
        print(f"  Messages: {num_messages}")
        print(f"  Payload: {payload_size} bytes")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {rate:,.0f} msg/s")
        print(f"  Throughput: {throughput_mb:.2f} MB/s")
        
        producer.close()
        
        # Baseline: should achieve at least 10k msg/s
        assert rate > 10000, f"Producer too slow: {rate:.0f} msg/s"
    
    def test_producer_batch_throughput(self, test_stream):
        """Measure batched producer throughput (pipelining)."""
        num_batches = 100
        batch_size = 100
        payload_size = 256
        payload = b"x" * payload_size
        
        producer = Producer(test_stream)
        
        start = time.perf_counter()
        for _ in range(num_batches):
            payloads = [payload] * batch_size
            producer.send_batch(payloads)
        elapsed = time.perf_counter() - start
        
        total_messages = num_batches * batch_size
        rate = total_messages / elapsed
        throughput_mb = (total_messages * payload_size) / (1024 * 1024) / elapsed
        
        print(f"\nBatched producer throughput:")
        print(f"  Messages: {total_messages}")
        print(f"  Batch size: {batch_size}")
        print(f"  Payload: {payload_size} bytes")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {rate:,.0f} msg/s")
        print(f"  Throughput: {throughput_mb:.2f} MB/s")
        
        producer.close()
        
        # Batched should be faster than single
        assert rate > 50000, f"Batched producer too slow: {rate:.0f} msg/s"
    
    def test_full_pipeline_throughput(self, test_stream):
        """Measure full pipeline: produce → consume → GPU process → ack."""
        num_messages = 1000
        payload_size = 256
        payload = b"x" * payload_size
        batch_size = 64
        
        # Setup
        producer = Producer(test_stream)
        consumer = Consumer(test_stream, group="benchmark_group", batch_size=batch_size)
        processor = GpuProcessor(batch_size=batch_size, slot_bytes=512)
        
        # Send all messages first
        for _ in range(num_messages):
            producer.send(payload)
        
        # Process all messages
        processed = 0
        start = time.perf_counter()
        
        while processed < num_messages:
            batch = consumer.fetch_batch(timeout_ms=1000)
            if batch:
                results = processor.process(batch)
                consumer.ack(batch)
                processed += len(batch)
        
        elapsed = time.perf_counter() - start
        
        rate = num_messages / elapsed
        throughput_mb = (num_messages * payload_size) / (1024 * 1024) / elapsed
        
        print(f"\nFull pipeline throughput (Track A):")
        print(f"  Messages: {num_messages}")
        print(f"  Batch size: {batch_size}")
        print(f"  Payload: {payload_size} bytes")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {rate:,.0f} msg/s")
        print(f"  Throughput: {throughput_mb:.2f} MB/s")
        print(f"  GPU: {processor.has_gpu}")
        
        stats = processor.stats
        print(f"  Avg H2D: {stats.avg_h2d_time_ms:.3f} ms")
        print(f"  Avg Kernel: {stats.avg_kernel_time_ms:.3f} ms")
        print(f"  Avg D2H: {stats.avg_d2h_time_ms:.3f} ms")
        
        producer.close()
        consumer.close()


class TestTrackALatency:
    """Track A latency measurements."""
    
    def test_round_trip_latency(self, test_stream):
        """Measure single message round-trip latency."""
        num_samples = 100
        payload = b"latency_test_payload"
        
        producer = Producer(test_stream)
        consumer = Consumer(test_stream, group="latency_group", batch_size=1)
        processor = GpuProcessor(batch_size=1, slot_bytes=256)
        
        latencies_ms = []
        
        for i in range(num_samples):
            start = time.perf_counter()
            
            # Send single message
            producer.send(payload)
            
            # Wait for it
            batch = None
            while batch is None:
                batch = consumer.fetch_batch(max_messages=1, timeout_ms=100)
            
            # Process on GPU
            processor.process(batch)
            consumer.ack(batch)
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies_ms.append(elapsed)
        
        p50 = statistics.median(latencies_ms)
        p95 = statistics.quantiles(latencies_ms, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies_ms, n=100)[98]  # 99th percentile
        
        print(f"\nRound-trip latency (Track A):")
        print(f"  Samples: {num_samples}")
        print(f"  p50: {p50:.2f} ms")
        print(f"  p95: {p95:.2f} ms")
        print(f"  p99: {p99:.2f} ms")
        print(f"  GPU: {processor.has_gpu}")
        
        producer.close()
        consumer.close()
        
        # Latency should be reasonable (< 10ms for local Redis + GPU)
        assert p50 < 50, f"Latency too high: p50={p50:.2f}ms"


class TestTrackBComparison:
    """
    Comparison tests between Track A and Track B.
    
    NOTE: Track B Python bindings not yet available (M4).
    These tests are marked as skipped until bindings are ready.
    """
    
    @pytest.mark.skip(reason="Track B Python bindings not yet available (M4)")
    def test_throughput_comparison(self, test_stream):
        """Compare throughput: Track A vs Track B."""
        # This will be implemented after M4 (Python bindings)
        pass
    
    @pytest.mark.skip(reason="Track B Python bindings not yet available (M4)")
    def test_latency_comparison(self, test_stream):
        """Compare latency: Track A vs Track B."""
        # This will be implemented after M4 (Python bindings)
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
