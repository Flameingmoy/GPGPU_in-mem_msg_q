"""
Track A GPU Processor: Bridge between Redis and GPU processing.

This module validates the GPU processing patterns used in Track B:
- Pinned memory staging
- Batch packing into fixed-size slots
- Async H2D/D2H transfers
- GPU kernel execution

Uses CuPy for GPU operations with NumPy fallback for testing without GPU.
"""

import time
from dataclasses import dataclass

import numpy as np

from .consumer import Batch

# Try to import cupy, fall back to numpy for testing without GPU
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore
    HAS_CUPY = False


@dataclass
class ProcessorStats:
    """Statistics for GPU processor."""

    batches_processed: int = 0
    messages_processed: int = 0
    bytes_processed: int = 0
    total_h2d_time_ms: float = 0.0
    total_kernel_time_ms: float = 0.0
    total_d2h_time_ms: float = 0.0

    @property
    def avg_h2d_time_ms(self) -> float:
        return self.total_h2d_time_ms / max(1, self.batches_processed)

    @property
    def avg_kernel_time_ms(self) -> float:
        return self.total_kernel_time_ms / max(1, self.batches_processed)

    @property
    def avg_d2h_time_ms(self) -> float:
        return self.total_d2h_time_ms / max(1, self.batches_processed)

    @property
    def throughput_mb_s(self) -> float:
        total_time_s = (
            self.total_h2d_time_ms + self.total_kernel_time_ms + self.total_d2h_time_ms
        ) / 1000.0
        if total_time_s == 0:
            return 0.0
        return (self.bytes_processed / 1024 / 1024) / total_time_s


@dataclass
class ProcessorConfig:
    """Configuration for GPU processor."""

    batch_size: int = 64
    slot_bytes: int = 512
    device_id: int = 0
    use_pinned_memory: bool = True


class GpuProcessor:
    """
    GPU processor for Track A validation.

    Processes batches from Redis Consumer using GPU:
    1. Pack messages into fixed-size slots (like Track B)
    2. Transfer to GPU (H2D)
    3. Execute processing kernel
    4. Transfer results back (D2H)

    This validates the memory patterns used in Track B.
    """

    def __init__(
        self,
        batch_size: int = 64,
        slot_bytes: int = 512,
        device_id: int = 0,
        use_pinned_memory: bool = True,
    ):
        """
        Initialize GPU processor.

        Args:
            batch_size: Maximum messages per batch
            slot_bytes: Fixed size for each message slot
            device_id: CUDA device ID
            use_pinned_memory: Use pinned memory for H2D/D2H (faster)
        """
        self.config = ProcessorConfig(
            batch_size=batch_size,
            slot_bytes=slot_bytes,
            device_id=device_id,
            use_pinned_memory=use_pinned_memory,
        )

        self._stats = ProcessorStats()

        # Set device
        if HAS_CUPY:
            cp.cuda.Device(device_id).use()

        # Pre-allocate buffers
        self._init_buffers()

    def _init_buffers(self):
        """Initialize host and device buffers."""
        shape = (self.config.batch_size, self.config.slot_bytes)
        total_bytes = self.config.batch_size * self.config.slot_bytes

        if HAS_CUPY:
            # Host buffers (pinned memory for faster transfers)
            if self.config.use_pinned_memory:
                self._h_input = cp.cuda.alloc_pinned_memory(total_bytes)
                self._h_output = cp.cuda.alloc_pinned_memory(total_bytes)
                # Create numpy views of pinned memory (use exact size)
                self._h_input_np = np.frombuffer(
                    self._h_input, dtype=np.uint8, count=total_bytes
                ).reshape(shape)
                self._h_output_np = np.frombuffer(
                    self._h_output, dtype=np.uint8, count=total_bytes
                ).reshape(shape)
            else:
                self._h_input_np = np.zeros(shape, dtype=np.uint8)
                self._h_output_np = np.zeros(shape, dtype=np.uint8)

            # Device buffers
            self._d_input = cp.zeros(shape, dtype=cp.uint8)
            self._d_output = cp.zeros(shape, dtype=cp.uint8)

            # CUDA stream for async operations
            self._stream = cp.cuda.Stream(non_blocking=True)
        else:
            # NumPy fallback (no GPU)
            self._h_input_np = np.zeros(shape, dtype=np.uint8)
            self._h_output_np = np.zeros(shape, dtype=np.uint8)
            self._d_input = self._h_input_np
            self._d_output = self._h_output_np
            self._stream = None

    def _pack_batch(self, batch: Batch) -> int:
        """
        Pack batch messages into fixed-size slots.

        Returns number of messages packed.
        """
        # Clear input buffer
        self._h_input_np.fill(0)

        count = 0
        for i, msg in enumerate(batch.messages):
            if i >= self.config.batch_size:
                break

            payload = msg.payload
            payload_len = min(len(payload), self.config.slot_bytes)

            # Copy payload into slot
            self._h_input_np[i, :payload_len] = np.frombuffer(payload[:payload_len], dtype=np.uint8)
            count += 1

        return count

    def _process_kernel(self, count: int):
        """
        Execute GPU processing kernel.

        For validation, just copies input to output (echo).
        Real implementation would do actual processing.
        """
        if HAS_CUPY:
            # Simple copy kernel (validates H2D/D2H pattern)
            self._d_output[:count] = self._d_input[:count]
        else:
            # NumPy fallback
            self._d_output[:count] = self._d_input[:count]

    def _unpack_results(self, count: int) -> list[bytes]:
        """Unpack results from output buffer."""
        results = []

        for i in range(count):
            data = bytes(self._h_output_np[i])
            # Trim trailing zeros
            data = data.rstrip(b"\x00")
            results.append(data)

        return results

    def process(self, batch: Batch) -> list[bytes]:
        """
        Process a batch of messages on GPU.

        Args:
            batch: Batch from Consumer

        Returns:
            List of processed payloads (bytes)
        """
        if not batch or not batch.messages:
            return []

        # Pack messages into slots
        count = self._pack_batch(batch)

        if HAS_CUPY:
            # H2D transfer (async on stream)
            h2d_start = time.perf_counter()
            with self._stream:
                self._d_input.set(self._h_input_np)
            self._stream.synchronize()
            h2d_time = (time.perf_counter() - h2d_start) * 1000

            # Kernel execution
            kernel_start = time.perf_counter()
            with self._stream:
                self._process_kernel(count)
            self._stream.synchronize()
            kernel_time = (time.perf_counter() - kernel_start) * 1000

            # D2H transfer
            d2h_start = time.perf_counter()
            with self._stream:
                self._d_output.get(out=self._h_output_np)
            self._stream.synchronize()
            d2h_time = (time.perf_counter() - d2h_start) * 1000
        else:
            # CPU fallback (for testing)
            h2d_time = 0.0
            kernel_start = time.perf_counter()
            self._process_kernel(count)
            kernel_time = (time.perf_counter() - kernel_start) * 1000
            d2h_time = 0.0
            self._h_output_np[:] = self._d_output[:]

        # Update stats
        self._stats.batches_processed += 1
        self._stats.messages_processed += count
        self._stats.bytes_processed += batch.total_bytes
        self._stats.total_h2d_time_ms += h2d_time
        self._stats.total_kernel_time_ms += kernel_time
        self._stats.total_d2h_time_ms += d2h_time

        # Unpack results
        return self._unpack_results(count)

    @property
    def stats(self) -> ProcessorStats:
        """Get processor statistics."""
        return self._stats

    def reset_stats(self):
        """Reset statistics."""
        self._stats = ProcessorStats()

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return HAS_CUPY

    def device_info(self) -> dict:
        """Get GPU device information."""
        if HAS_CUPY:
            props = cp.cuda.runtime.getDeviceProperties(self.config.device_id)
            return {
                "name": props["name"].decode(),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "total_memory_gb": props["totalGlobalMem"] / (1024**3),
                "multiprocessors": props["multiProcessorCount"],
            }
        else:
            return {"error": "CuPy not available, using CPU fallback"}
