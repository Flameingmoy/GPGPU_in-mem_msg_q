# Quick Start

## Basic Usage

```python
from gpuqueue import GpuQueue, QueueConfig

# Configure queue
config = QueueConfig(
    capacity=1024,      # Number of slots (must be power of 2)
    slot_bytes=512      # Bytes per slot
)

# Create and use queue
with GpuQueue(config) as q:
    # Enqueue a message
    msg_id = q.enqueue(b"Hello, GPU Queue!")
    
    # Wait for processing
    import time
    time.sleep(0.1)
    
    # Check for completions
    completed = q.poll_completions(10)
    print(f"Completed: {completed}")
    
    # Dequeue result
    success, data = q.try_dequeue_result(msg_id)
    if success:
        print(f"Result: {data}")
```

## Batch Processing

```python
from gpuqueue import GpuQueue, QueueConfig

with GpuQueue(QueueConfig(4096, 256)) as q:
    # Enqueue batch
    msg_ids = []
    for i in range(100):
        payload = f"Message {i}".encode()
        msg_ids.append(q.enqueue(payload))
    
    # Process all completions
    import time
    time.sleep(0.5)
    
    completed = q.poll_completions(100)
    for msg_id in completed:
        success, data = q.try_dequeue_result(msg_id)
        if success:
            print(f"Got: {data.decode()}")
```

## Queue Statistics

```python
from gpuqueue import GpuQueue, QueueConfig

with GpuQueue(QueueConfig(1024, 256)) as q:
    # Enqueue some messages
    for _ in range(50):
        q.enqueue(b"test")
    
    # Get statistics
    stats = q.stats()
    print(f"Enqueued: {stats.enqueue_count}")
    print(f"Processed: {stats.processed_count}")
    print(f"Queue depth: {stats.queue_depth}")
```

## Track A (Redis Backend)

```python
from gpuqueue.track_a import Producer, Consumer, GpuProcessor

# Producer
with Producer("my_stream") as p:
    for i in range(100):
        p.send(f"Message {i}".encode())

# Consumer with GPU processing
consumer = Consumer("my_stream", group="my_group")
processor = GpuProcessor(batch_size=64)

batch = consumer.fetch_batch(timeout_ms=1000)
if batch:
    results = processor.process(batch)
    consumer.ack(batch)

consumer.close()
```
