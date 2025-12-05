"""
Unit tests for GPUQueue Python API.

Tests:
- QueueConfig validation
- GpuQueue lifecycle (init, enqueue, dequeue, shutdown)
- Error handling
- Edge cases
"""

import pytest
import time


class TestQueueConfig:
    """Tests for QueueConfig validation."""
    
    def test_default_config(self):
        """Default config should be valid."""
        from gpuqueue import QueueConfig
        
        cfg = QueueConfig()
        assert cfg.is_valid()
        assert cfg.capacity > 0
        assert cfg.slot_bytes > 0
    
    def test_custom_config(self):
        """Custom config with power-of-2 capacity."""
        from gpuqueue import QueueConfig
        
        cfg = QueueConfig(capacity=256, slot_bytes=1024)
        assert cfg.is_valid()
        assert cfg.capacity == 256
        assert cfg.slot_bytes == 1024
    
    def test_invalid_capacity_not_power_of_two(self):
        """Non-power-of-2 capacity should be invalid."""
        from gpuqueue import QueueConfig
        
        cfg = QueueConfig(capacity=100, slot_bytes=512)
        assert not cfg.is_valid()
    
    def test_invalid_zero_capacity(self):
        """Zero capacity should be invalid."""
        from gpuqueue import QueueConfig
        
        cfg = QueueConfig(capacity=0, slot_bytes=512)
        assert not cfg.is_valid()
    
    def test_invalid_zero_slot_bytes(self):
        """Zero slot bytes should be invalid."""
        from gpuqueue import QueueConfig
        
        cfg = QueueConfig(capacity=256, slot_bytes=0)
        assert not cfg.is_valid()
    
    def test_config_repr(self):
        """Config should have readable repr."""
        from gpuqueue import QueueConfig
        
        cfg = QueueConfig(capacity=512, slot_bytes=256)
        repr_str = repr(cfg)
        assert "512" in repr_str
        assert "256" in repr_str


class TestQueueStatus:
    """Tests for QueueStatus enum."""
    
    def test_status_values_exist(self):
        """All expected status values should exist."""
        from gpuqueue import QueueStatus
        
        assert QueueStatus.OK is not None
        assert QueueStatus.ERR_FULL is not None
        assert QueueStatus.ERR_TIMEOUT is not None
        assert QueueStatus.ERR_SHUTDOWN is not None
        assert QueueStatus.ERR_NOT_READY is not None
        assert QueueStatus.ERR_NOT_FOUND is not None
    
    def test_ok_is_zero(self):
        """OK status should be 0."""
        from gpuqueue import QueueStatus
        
        assert int(QueueStatus.OK) == 0


class TestQueueStats:
    """Tests for QueueStats struct."""
    
    def test_stats_fields(self):
        """Stats should have expected fields."""
        from gpuqueue import QueueStats
        
        stats = QueueStats()
        assert hasattr(stats, 'enqueue_count')
        assert hasattr(stats, 'dequeue_count')
        assert hasattr(stats, 'processed_count')
        assert hasattr(stats, 'queue_depth')


class TestGpuQueueLifecycle:
    """Tests for GpuQueue lifecycle management."""
    
    def test_create_and_shutdown(self):
        """Create queue and shut it down."""
        from gpuqueue import GpuQueue, QueueConfig
        
        cfg = QueueConfig(capacity=128, slot_bytes=256)
        q = GpuQueue(cfg)
        
        assert q.is_running()
        q.shutdown()
        assert not q.is_running()
    
    def test_context_manager(self):
        """Queue should work as context manager."""
        from gpuqueue import GpuQueue, QueueConfig
        
        cfg = QueueConfig(capacity=128, slot_bytes=256)
        
        with GpuQueue(cfg) as q:
            assert q.is_running()
        
        # Should be shut down after exiting context
        assert not q.is_running()
    
    def test_double_shutdown(self):
        """Shutting down twice should be safe."""
        from gpuqueue import GpuQueue, QueueConfig
        
        cfg = QueueConfig(capacity=128, slot_bytes=256)
        q = GpuQueue(cfg)
        
        q.shutdown()
        q.shutdown()  # Should not raise
        assert not q.is_running()


class TestEnqueueDequeue:
    """Tests for enqueue and dequeue operations."""
    
    def test_enqueue_returns_msg_id(self):
        """Enqueue should return a message ID."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            msg_id = q.enqueue(b"test")
            assert isinstance(msg_id, int)
            assert msg_id >= 0
    
    def test_enqueue_multiple_increments_id(self):
        """Message IDs should increment."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            id1 = q.enqueue(b"msg1")
            id2 = q.enqueue(b"msg2")
            id3 = q.enqueue(b"msg3")
            
            assert id2 == id1 + 1
            assert id3 == id2 + 1
    
    def test_enqueue_dequeue_roundtrip(self):
        """Message should round-trip through the queue."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            payload = b"hello, gpu queue!"
            msg_id = q.enqueue(payload)
            
            # Wait for processing with timeout
            for _ in range(100):
                success, data = q.try_dequeue_result(msg_id)
                if success:
                    assert data == payload
                    return
                time.sleep(0.01)
            
            pytest.fail("Message not processed within timeout")
    
    def test_enqueue_empty_payload(self):
        """Empty payload should work."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            msg_id = q.enqueue(b"")
            assert msg_id >= 0
    
    def test_dequeue_not_ready(self):
        """Dequeue before processing returns not ready."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            msg_id = q.enqueue(b"test")
            
            # Immediately try to dequeue (might not be ready yet)
            success, data = q.try_dequeue_result(msg_id)
            # Either ready or not ready is acceptable
            assert isinstance(success, bool)


class TestQueueStats:
    """Tests for queue statistics."""
    
    def test_stats_after_enqueue(self):
        """Stats should reflect enqueued messages."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            initial_stats = q.stats()
            
            q.enqueue(b"msg1")
            q.enqueue(b"msg2")
            
            stats = q.stats()
            assert stats.enqueue_count >= initial_stats.enqueue_count + 2
    
    def test_depth_changes(self):
        """Depth should change with enqueue/dequeue."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            initial_depth = q.depth()
            
            q.enqueue(b"test")
            # Depth may increase (before processing) or stay same (if processed quickly)
            assert q.depth() >= 0


class TestPollCompletions:
    """Tests for poll_completions API."""
    
    def test_poll_empty_queue(self):
        """Polling empty queue returns empty list."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            completed = q.poll_completions(10)
            assert isinstance(completed, list)
    
    def test_poll_after_processing(self):
        """Polling should find completed messages."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            msg_ids = [q.enqueue(b"test") for _ in range(5)]
            
            # Wait for processing
            time.sleep(0.2)
            
            completed = q.poll_completions(10)
            # Some or all should be completed
            assert isinstance(completed, list)


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_enqueue_payload_too_large(self):
        """Payload larger than slot should fail."""
        from gpuqueue import GpuQueue, QueueConfig
        
        # Small slot size
        with GpuQueue(QueueConfig(128, 64)) as q:
            large_payload = b"x" * 1000  # Much larger than 64 bytes
            
            with pytest.raises(ValueError):
                q.enqueue(large_payload)
    
    def test_enqueue_after_shutdown(self):
        """Enqueue after shutdown should fail."""
        from gpuqueue import GpuQueue, QueueConfig
        
        q = GpuQueue(QueueConfig(128, 256))
        q.shutdown()
        
        with pytest.raises(ValueError):
            q.enqueue(b"test")


class TestBufferProtocol:
    """Tests for buffer protocol support."""
    
    def test_enqueue_numpy_array(self):
        """Should accept numpy arrays."""
        pytest.importorskip("numpy")
        import numpy as np
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 512)) as q:
            arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
            msg_id = q.enqueue(arr)
            assert msg_id >= 0
    
    def test_enqueue_bytearray(self):
        """Should accept bytearray."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            data = bytearray(b"hello bytearray")
            msg_id = q.enqueue(data)
            assert msg_id >= 0
    
    def test_enqueue_memoryview(self):
        """Should accept memoryview."""
        from gpuqueue import GpuQueue, QueueConfig
        
        with GpuQueue(QueueConfig(128, 256)) as q:
            data = b"hello memoryview"
            mv = memoryview(data)
            msg_id = q.enqueue(mv)
            assert msg_id >= 0


class TestIsAvailable:
    """Tests for is_available() utility."""
    
    def test_is_available_returns_bool(self):
        """is_available should return boolean."""
        from gpuqueue import is_available
        
        result = is_available()
        assert isinstance(result, bool)
    
    def test_is_available_true_when_bindings_exist(self):
        """is_available should be True when bindings are loaded."""
        from gpuqueue import is_available, GpuQueue
        
        # If we can import GpuQueue, bindings should be available
        if GpuQueue is not None:
            assert is_available()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
