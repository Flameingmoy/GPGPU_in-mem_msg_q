"""
GPUQueue: CUDA-backed GPU-resident message queue.

Track B (GPU-Resident Queue):
    from gpuqueue import GpuQueue, QueueConfig
    
    with GpuQueue(QueueConfig(capacity=1024, slot_bytes=512)) as q:
        msg_id = q.enqueue(b"hello world")
        # Wait for processing...
        success, result = q.try_dequeue_result(msg_id)

Track A (Redis Validation):
    from gpuqueue.track_a import Producer, Consumer, GpuProcessor
"""
from ._version import __version__

# Core bindings
try:
    from ._core import (
        init, 
        shutdown, 
        version as core_version,
        # Track B classes
        GpuQueue,
        QueueConfig,
        QueueStats,
        QueueStatus,
    )
    _BINDINGS_AVAILABLE = True
except ImportError as e:
    _BINDINGS_AVAILABLE = False
    _import_error = e
    
    # Stubs for when bindings aren't built
    core_version = None
    GpuQueue = None
    QueueConfig = None
    QueueStats = None
    QueueStatus = None
    
    def init(*_, **__):
        raise RuntimeError(
            "gpuqueue extension not built. Build/install the package first:\n"
            "  pip install -e .\n"
            f"Original error: {_import_error}"
        )
    
    def shutdown():
        return None


def is_available() -> bool:
    """Check if the CUDA extension is available."""
    return _BINDINGS_AVAILABLE


__all__ = [
    "__version__",
    # Legacy API
    "init",
    "shutdown", 
    "core_version",
    # Track B API
    "GpuQueue",
    "QueueConfig",
    "QueueStats",
    "QueueStatus",
    # Utility
    "is_available",
]
