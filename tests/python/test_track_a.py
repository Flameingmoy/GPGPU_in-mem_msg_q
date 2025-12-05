"""
Unit tests for Track A (Redis-backed queue) components.

Requires Redis to be running.
"""

import pytest
import time
import os


# Skip all tests if Redis is not available
def redis_available():
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return True
    except:
        return False


pytestmark = pytest.mark.skipif(
    not redis_available(),
    reason="Redis not available"
)


@pytest.fixture
def unique_stream():
    """Generate unique stream name for test isolation."""
    import redis
    stream = f"test_stream_{int(time.time() * 1000)}_{os.getpid()}"
    yield stream
    
    # Cleanup
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.delete(stream)
    except:
        pass


class TestProducer:
    """Tests for Track A Producer."""
    
    def test_producer_create(self, unique_stream):
        """Producer should initialize correctly."""
        from gpuqueue.track_a import Producer
        
        p = Producer(unique_stream)
        # Check producer was created and has expected methods
        assert hasattr(p, 'send')
        assert hasattr(p, 'send_batch')
        assert hasattr(p, 'close')
        p.close()
    
    def test_producer_send(self, unique_stream):
        """Producer should send messages."""
        from gpuqueue.track_a import Producer
        
        p = Producer(unique_stream)
        msg_id = p.send(b"test message")
        
        assert msg_id is not None
        assert isinstance(msg_id, str)
        p.close()
    
    def test_producer_send_multiple(self, unique_stream):
        """Producer should send multiple messages."""
        from gpuqueue.track_a import Producer
        
        p = Producer(unique_stream)
        ids = []
        
        for i in range(10):
            msg_id = p.send(f"message {i}".encode())
            ids.append(msg_id)
        
        assert len(set(ids)) == 10  # All unique
        p.close()
    
    def test_producer_send_batch(self, unique_stream):
        """Producer should send batched messages efficiently."""
        from gpuqueue.track_a import Producer
        
        p = Producer(unique_stream)
        payloads = [f"batch msg {i}".encode() for i in range(100)]
        
        ids = p.send_batch(payloads)
        assert len(ids) == 100
        p.close()
    
    def test_producer_context_manager(self, unique_stream):
        """Producer should work as context manager."""
        from gpuqueue.track_a import Producer
        
        with Producer(unique_stream) as p:
            msg_id = p.send(b"context test")
            assert msg_id is not None


class TestConsumer:
    """Tests for Track A Consumer."""
    
    def test_consumer_create(self, unique_stream):
        """Consumer should initialize correctly."""
        from gpuqueue.track_a import Consumer
        
        c = Consumer(unique_stream, group="test_group")
        # Check consumer was created and has expected methods
        assert hasattr(c, 'fetch_batch')
        assert hasattr(c, 'ack')
        assert hasattr(c, 'close')
        c.close()
    
    def test_consumer_fetch_empty(self, unique_stream):
        """Consumer should handle empty stream."""
        from gpuqueue.track_a import Consumer
        
        c = Consumer(unique_stream, group="test_group")
        batch = c.fetch_batch(timeout_ms=100)
        
        # Should return None or empty batch
        assert batch is None or len(batch) == 0
        c.close()
    
    def test_consumer_fetch_after_produce(self, unique_stream):
        """Consumer should fetch produced messages."""
        from gpuqueue.track_a import Producer, Consumer
        
        # Produce messages
        p = Producer(unique_stream)
        for i in range(5):
            p.send(f"msg {i}".encode())
        p.close()
        
        # Consume messages
        c = Consumer(unique_stream, group="test_group", batch_size=10)
        batch = c.fetch_batch(timeout_ms=1000)
        
        assert batch is not None
        assert len(batch) == 5
        
        c.ack(batch)
        c.close()
    
    def test_consumer_ack(self, unique_stream):
        """Consumer should acknowledge messages."""
        from gpuqueue.track_a import Producer, Consumer
        
        p = Producer(unique_stream)
        p.send(b"ack test")
        p.close()
        
        c = Consumer(unique_stream, group="test_group")
        batch = c.fetch_batch(timeout_ms=1000)
        
        if batch:
            # Should not raise
            c.ack(batch)
        
        c.close()
    
    def test_batch_properties(self, unique_stream):
        """Batch should have expected properties."""
        from gpuqueue.track_a import Producer, Consumer
        
        p = Producer(unique_stream)
        p.send(b"batch test")
        p.close()
        
        c = Consumer(unique_stream, group="test_group")
        batch = c.fetch_batch(timeout_ms=1000)
        
        assert batch is not None
        assert hasattr(batch, 'messages')
        assert len(batch) == len(batch.messages)
        
        c.close()


class TestGpuProcessor:
    """Tests for Track A GPU Processor."""
    
    def test_processor_create(self):
        """Processor should initialize."""
        from gpuqueue.track_a import GpuProcessor
        
        # GpuProcessor stores config internally
        proc = GpuProcessor(batch_size=64, slot_bytes=256)
        # Just check it was created without error
        assert proc is not None
    
    def test_processor_has_gpu_attribute(self):
        """Processor should report GPU availability."""
        from gpuqueue.track_a import GpuProcessor
        
        proc = GpuProcessor(batch_size=64, slot_bytes=256)
        assert isinstance(proc.has_gpu, bool)
    
    def test_processor_process_batch(self, unique_stream):
        """Processor should process a batch."""
        from gpuqueue.track_a import Producer, Consumer, GpuProcessor
        
        # Setup
        p = Producer(unique_stream)
        for i in range(10):
            p.send(f"process test {i}".encode())
        p.close()
        
        c = Consumer(unique_stream, group="test_group", batch_size=64)
        proc = GpuProcessor(batch_size=64, slot_bytes=256)
        
        # Fetch and process
        batch = c.fetch_batch(timeout_ms=1000)
        assert batch is not None
        
        results = proc.process(batch)
        
        # Results should match input count
        assert len(results) == len(batch)
        
        c.ack(batch)
        c.close()
    
    def test_processor_timing(self, unique_stream):
        """Processor should track timing."""
        from gpuqueue.track_a import Producer, Consumer, GpuProcessor
        
        p = Producer(unique_stream)
        for i in range(5):
            p.send(b"x" * 100)
        p.close()
        
        c = Consumer(unique_stream, group="test_group", batch_size=64)
        proc = GpuProcessor(batch_size=64, slot_bytes=256)
        
        batch = c.fetch_batch(timeout_ms=1000)
        if batch:
            results = proc.process(batch)
            
            # Results should be a list
            assert isinstance(results, list)
            assert len(results) == len(batch)
            
            # Check stats for timing info
            stats = proc.stats
            assert stats is not None
        
        c.close()


class TestEndToEnd:
    """End-to-end tests for Track A pipeline."""
    
    def test_full_pipeline(self, unique_stream):
        """Test complete produce → consume → process → ack cycle."""
        from gpuqueue.track_a import Producer, Consumer, GpuProcessor
        
        # Produce
        with Producer(unique_stream) as p:
            for i in range(20):
                p.send(f"e2e message {i}".encode())
        
        # Consume and process
        c = Consumer(unique_stream, group="e2e_group", batch_size=32)
        proc = GpuProcessor(batch_size=32, slot_bytes=256)
        
        total_processed = 0
        attempts = 0
        max_attempts = 10
        
        while total_processed < 20 and attempts < max_attempts:
            batch = c.fetch_batch(timeout_ms=500)
            if batch:
                results = proc.process(batch)
                c.ack(batch)
                total_processed += len(batch)
            attempts += 1
        
        assert total_processed == 20, f"Only processed {total_processed}/20 messages"
        c.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
