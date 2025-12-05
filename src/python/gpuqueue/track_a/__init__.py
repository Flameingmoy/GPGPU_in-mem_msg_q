"""
Track A: Redis-backed MVP for GPU message queue.

This module provides a validation harness for the GPU processing patterns
used in Track B. It uses Redis Streams as the message broker.

Components:
- Producer: Sends messages to Redis Stream
- Consumer: Fetches batches from Redis Stream with consumer groups
- GpuProcessor: Processes batches on GPU (reuses Track B patterns)

Usage:
    from gpuqueue.track_a import Producer, Consumer, GpuProcessor
    
    producer = Producer("my_stream")
    consumer = Consumer("my_stream", group="workers")
    processor = GpuProcessor()
    
    # Send messages
    producer.send(b"payload")
    
    # Process in batches
    batch = consumer.fetch_batch(max_messages=64)
    results = processor.process(batch)
    consumer.ack(batch)
"""

from .producer import Producer
from .consumer import Consumer, Message, Batch
from .gpu_processor import GpuProcessor

__all__ = ["Producer", "Consumer", "Message", "Batch", "GpuProcessor"]
