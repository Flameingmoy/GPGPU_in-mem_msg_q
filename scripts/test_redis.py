#!/usr/bin/env python3
"""
Test Redis connectivity and basic operations.

Usage:
    # Start Redis first
    docker compose -f docker/docker-compose.yml up -d

    # Run this script
    python scripts/test_redis.py
"""

import sys


def test_redis_connectivity():
    """Test basic Redis connectivity."""
    try:
        import redis
    except ImportError:
        print("❌ redis-py not installed. Run: pip install 'gpuqueue[track-a]'")
        return False

    print("Testing Redis connectivity...")

    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Test PING
        response = r.ping()
        print(f"  ✓ PING: {response}")

        # Test SET/GET
        r.set("gpuqueue:test", "hello")
        value = r.get("gpuqueue:test")
        assert value == "hello", f"Expected 'hello', got '{value}'"
        r.delete("gpuqueue:test")
        print("  ✓ SET/GET: working")

        # Test Streams (XADD/XREAD)
        stream_name = "gpuqueue:test_stream"
        r.delete(stream_name)

        entry_id = r.xadd(stream_name, {"payload": "test_message"})
        print(f"  ✓ XADD: created entry {entry_id}")

        entries = r.xread({stream_name: "0"}, count=1)
        assert len(entries) == 1
        assert len(entries[0][1]) == 1
        print(f"  ✓ XREAD: read {len(entries[0][1])} entry")

        # Cleanup
        r.delete(stream_name)

        # Server info
        info = r.info("server")
        print(f"  ✓ Redis version: {info['redis_version']}")

        r.close()
        return True

    except redis.ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print("   Make sure Redis is running: docker compose -f docker/docker-compose.yml up -d")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_track_a_imports():
    """Test Track A module imports."""
    print("\nTesting Track A imports...")

    try:
        from gpuqueue.track_a import Consumer, GpuProcessor, Producer

        # Verify classes exist
        assert Producer is not None, "Producer should be importable"
        assert Consumer is not None, "Consumer should be importable"
        assert GpuProcessor is not None, "GpuProcessor should be importable"
        print("  ✓ Producer imported")
        print("  ✓ Consumer imported")
        print("  ✓ GpuProcessor imported")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_track_a_pipeline():
    """Test Track A producer/consumer/processor pipeline."""
    print("\nTesting Track A pipeline...")

    try:
        import redis

        from gpuqueue.track_a import Consumer, GpuProcessor, Producer
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    stream_name = "gpuqueue:test_pipeline"
    group_name = "test_group"

    try:
        # Cleanup first
        r = redis.Redis()
        r.delete(stream_name)
        try:
            r.xgroup_destroy(stream_name, group_name)
        except redis.ResponseError:
            pass
        r.close()

        # Create producer
        producer = Producer(stream_name)
        print("  ✓ Producer created")

        # Send messages
        num_messages = 10
        for i in range(num_messages):
            producer.send(f"message_{i}".encode())
        print(f"  ✓ Sent {num_messages} messages")

        info = producer.stream_info()
        print(f"  ✓ Stream length: {info['length']}")

        # Create consumer
        consumer = Consumer(stream_name, group=group_name)
        print(f"  ✓ Consumer created (id: {consumer.config.consumer_id})")

        # Fetch batch
        batch = consumer.fetch_batch(max_messages=num_messages, timeout_ms=1000)
        if batch:
            print(f"  ✓ Fetched batch: {len(batch)} messages, {batch.total_bytes} bytes")
        else:
            print("  ❌ No batch fetched")
            return False

        # Create processor
        processor = GpuProcessor(batch_size=64, slot_bytes=256)
        print(f"  ✓ GpuProcessor created (GPU: {processor.has_gpu})")

        if processor.has_gpu:
            dev_info = processor.device_info()
            print(f"    Device: {dev_info.get('name', 'N/A')}")

        # Process batch
        results = processor.process(batch)
        print(f"  ✓ Processed {len(results)} messages")

        # Verify results
        for i, result in enumerate(results):
            expected = f"message_{i}".encode()
            assert result == expected, f"Expected {expected}, got {result}"
        print("  ✓ Results verified (echo test passed)")

        # ACK messages
        ack_count = consumer.ack(batch)
        print(f"  ✓ ACKed {ack_count} messages")

        # Check stats
        stats = processor.stats
        print(
            f"  ✓ Processor stats: {stats.messages_processed} msgs, "
            f"{stats.throughput_mb_s:.2f} MB/s"
        )

        # Cleanup
        producer.close()
        consumer.close()

        r = redis.Redis()
        r.delete(stream_name)
        r.close()

        return True

    except Exception as e:
        import traceback

        print(f"❌ Pipeline test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("GPUQueue Track A - Redis Connectivity Test")
    print("=" * 60)

    results = []

    # Test 1: Imports
    results.append(("Track A imports", test_track_a_imports()))

    # Test 2: Redis connectivity
    results.append(("Redis connectivity", test_redis_connectivity()))

    # Test 3: Full pipeline (only if Redis is available)
    if results[-1][1]:
        results.append(("Track A pipeline", test_track_a_pipeline()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
