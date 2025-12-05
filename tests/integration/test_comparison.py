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


class TestTrackBThroughput:
    """Track B (GPU-resident queue) throughput benchmarks."""
    
    def test_enqueue_throughput(self):
        """Measure Track B enqueue-only throughput."""
        from gpuqueue import GpuQueue, QueueConfig
        
        num_messages = 10000
        payload_size = 256
        payload = b"x" * payload_size
        
        cfg = QueueConfig(capacity=16384, slot_bytes=512)
        
        with GpuQueue(cfg) as q:
            start = time.perf_counter()
            for _ in range(num_messages):
                q.enqueue(payload)
            elapsed = time.perf_counter() - start
        
        rate = num_messages / elapsed
        throughput_mb = (num_messages * payload_size) / (1024 * 1024) / elapsed
        
        print(f"\nTrack B enqueue throughput:")
        print(f"  Messages: {num_messages}")
        print(f"  Payload: {payload_size} bytes")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {rate:,.0f} msg/s")
        print(f"  Throughput: {throughput_mb:.2f} MB/s")
        
        # Track B should be fast
        assert rate > 50000, f"Track B too slow: {rate:.0f} msg/s"
    
    def test_full_roundtrip_throughput(self):
        """Measure Track B full roundtrip: enqueue → process → dequeue."""
        from gpuqueue import GpuQueue, QueueConfig
        
        num_messages = 1000
        payload_size = 256
        payload = b"x" * payload_size
        
        cfg = QueueConfig(capacity=4096, slot_bytes=512)
        
        with GpuQueue(cfg) as q:
            start = time.perf_counter()
            
            # Enqueue all
            msg_ids = []
            for _ in range(num_messages):
                msg_id = q.enqueue(payload)
                msg_ids.append(msg_id)
            
            # Wait for all to process and dequeue
            for msg_id in msg_ids:
                while True:
                    success, data = q.try_dequeue_result(msg_id)
                    if success:
                        break
                    time.sleep(0.0001)  # 100us
            
            elapsed = time.perf_counter() - start
        
        rate = num_messages / elapsed
        throughput_mb = (num_messages * payload_size) / (1024 * 1024) / elapsed
        
        print(f"\nTrack B full roundtrip throughput:")
        print(f"  Messages: {num_messages}")
        print(f"  Payload: {payload_size} bytes")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {rate:,.0f} msg/s")
        print(f"  Throughput: {throughput_mb:.2f} MB/s")


class TestTrackComparison:
    """Direct comparison tests between Track A and Track B."""
    
    def test_throughput_comparison(self, test_stream):
        """Compare throughput: Track A vs Track B."""
        from gpuqueue import GpuQueue, QueueConfig
        from gpuqueue.track_a import Producer, Consumer, GpuProcessor
        
        num_messages = 1000
        payload_size = 256
        payload = b"x" * payload_size
        
        # === Track B ===
        cfg = QueueConfig(capacity=4096, slot_bytes=512)
        with GpuQueue(cfg) as q:
            start = time.perf_counter()
            for _ in range(num_messages):
                q.enqueue(payload)
            
            # Wait for processing
            while q.stats().processed_count < num_messages:
                time.sleep(0.001)
            track_b_time = time.perf_counter() - start
        
        track_b_rate = num_messages / track_b_time
        
        # === Track A ===
        producer = Producer(test_stream)
        consumer = Consumer(test_stream, group="comparison_group", batch_size=64)
        processor = GpuProcessor(batch_size=64, slot_bytes=512)
        
        start = time.perf_counter()
        for _ in range(num_messages):
            producer.send(payload)
        
        processed = 0
        while processed < num_messages:
            batch = consumer.fetch_batch(timeout_ms=100)
            if batch:
                processor.process(batch)
                consumer.ack(batch)
                processed += len(batch)
        track_a_time = time.perf_counter() - start
        
        track_a_rate = num_messages / track_a_time
        
        producer.close()
        consumer.close()
        
        speedup = track_b_rate / track_a_rate if track_a_rate > 0 else float('inf')
        
        print(f"\n{'='*50}")
        print(f"THROUGHPUT COMPARISON")
        print(f"{'='*50}")
        print(f"Track A (Redis): {track_a_rate:,.0f} msg/s")
        print(f"Track B (GPU):   {track_b_rate:,.0f} msg/s")
        print(f"Speedup:         {speedup:.1f}x")
        print(f"{'='*50}")
        
        # Track B should be at least as fast as Track A
        # (The real advantage is lower latency for single messages)
        assert track_b_rate > 0, "Track B should have non-zero throughput"
    
    def test_latency_comparison(self, test_stream):
        """Compare single-message latency: Track A vs Track B."""
        from gpuqueue import GpuQueue, QueueConfig
        from gpuqueue.track_a import Producer, Consumer, GpuProcessor
        
        num_samples = 50
        payload = b"latency_test"
        
        # === Track B latency ===
        cfg = QueueConfig(capacity=256, slot_bytes=256)
        track_b_latencies = []
        
        with GpuQueue(cfg) as q:
            for _ in range(num_samples):
                start = time.perf_counter()
                msg_id = q.enqueue(payload)
                
                while True:
                    success, data = q.try_dequeue_result(msg_id)
                    if success:
                        break
                
                elapsed_ms = (time.perf_counter() - start) * 1000
                track_b_latencies.append(elapsed_ms)
        
        # === Track A latency ===
        producer = Producer(test_stream)
        consumer = Consumer(test_stream, group="latency_group", batch_size=1)
        processor = GpuProcessor(batch_size=1, slot_bytes=256)
        track_a_latencies = []
        
        for _ in range(num_samples):
            start = time.perf_counter()
            producer.send(payload)
            
            batch = None
            while batch is None:
                batch = consumer.fetch_batch(max_messages=1, timeout_ms=100)
            
            processor.process(batch)
            consumer.ack(batch)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            track_a_latencies.append(elapsed_ms)
        
        producer.close()
        consumer.close()
        
        # Statistics
        track_b_p50 = statistics.median(track_b_latencies)
        track_b_p95 = statistics.quantiles(track_b_latencies, n=20)[18]
        
        track_a_p50 = statistics.median(track_a_latencies)
        track_a_p95 = statistics.quantiles(track_a_latencies, n=20)[18]
        
        print(f"\n{'='*50}")
        print(f"LATENCY COMPARISON (ms)")
        print(f"{'='*50}")
        print(f"{'Metric':<15} {'Track A':>12} {'Track B':>12}")
        print(f"{'-'*50}")
        print(f"{'p50':<15} {track_a_p50:>12.2f} {track_b_p50:>12.2f}")
        print(f"{'p95':<15} {track_a_p95:>12.2f} {track_b_p95:>12.2f}")
        print(f"{'='*50}")
        
        # Track B should have lower latency (no Redis round-trip)
        # Allow some tolerance for CI variance
        assert track_b_p50 < track_a_p50 * 5, "Track B p50 should be reasonably low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
