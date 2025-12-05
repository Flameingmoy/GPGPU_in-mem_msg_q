#!/usr/bin/env python3
"""
Benchmark GPUQueue with hardware monitoring.

Runs throughput and latency benchmarks while recording CPU, RAM, GPU, and VRAM usage.

Usage:
    python scripts/benchmark_with_monitor.py
    python scripts/benchmark_with_monitor.py --track-a  # Also benchmark Track A
    python scripts/benchmark_with_monitor.py --csv results.csv  # Save metrics
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add scripts to path for hardware_monitor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_monitor import HardwareMonitor, save_to_csv


def benchmark_track_b_throughput(num_messages: int = 10000, payload_size: int = 256):
    """Benchmark Track B enqueue throughput."""
    from gpuqueue import GpuQueue, QueueConfig

    payload = b"x" * payload_size
    cfg = QueueConfig(capacity=16384, slot_bytes=max(512, payload_size + 64))

    with GpuQueue(cfg) as q:
        start = time.perf_counter()
        for _ in range(num_messages):
            q.enqueue(payload)

        # Wait for all to process
        while q.stats().processed_count < num_messages:
            time.sleep(0.001)
        elapsed = time.perf_counter() - start

    return {
        "messages": num_messages,
        "payload_bytes": payload_size,
        "elapsed_s": elapsed,
        "rate_msg_s": num_messages / elapsed,
        "throughput_mb_s": (num_messages * payload_size) / (1024 * 1024) / elapsed,
    }


def benchmark_track_b_latency(num_samples: int = 100, payload_size: int = 64):
    """Benchmark Track B single-message latency."""
    import statistics

    from gpuqueue import GpuQueue, QueueConfig

    payload = b"x" * payload_size
    cfg = QueueConfig(capacity=256, slot_bytes=256)
    latencies = []

    with GpuQueue(cfg) as q:
        for _ in range(num_samples):
            start = time.perf_counter()
            msg_id = q.enqueue(payload)

            while True:
                success, _ = q.try_dequeue_result(msg_id)
                if success:
                    break

            latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "samples": num_samples,
        "payload_bytes": payload_size,
        "p50_ms": statistics.median(latencies),
        "p95_ms": statistics.quantiles(latencies, n=20)[18],
        "p99_ms": statistics.quantiles(latencies, n=100)[98],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def benchmark_track_a_throughput(
    num_messages: int = 10000, payload_size: int = 256, batch_size: int = 64
):
    """Benchmark Track A throughput."""
    from gpuqueue.track_a import Consumer, GpuProcessor, Producer

    stream_name = f"bench_{int(time.time())}"
    payload = b"x" * payload_size

    producer = Producer(stream_name)
    consumer = Consumer(stream_name, group="bench_group", batch_size=batch_size)
    processor = GpuProcessor(batch_size=batch_size, slot_bytes=max(512, payload_size + 64))

    try:
        start = time.perf_counter()

        # Produce all messages
        for _ in range(num_messages):
            producer.send(payload)

        # Consume and process
        processed = 0
        while processed < num_messages:
            batch = consumer.fetch_batch(timeout_ms=100)
            if batch:
                processor.process(batch)
                consumer.ack(batch)
                processed += len(batch)

        elapsed = time.perf_counter() - start
    finally:
        producer.close()
        consumer.close()

    return {
        "messages": num_messages,
        "payload_bytes": payload_size,
        "batch_size": batch_size,
        "elapsed_s": elapsed,
        "rate_msg_s": num_messages / elapsed,
        "throughput_mb_s": (num_messages * payload_size) / (1024 * 1024) / elapsed,
    }


def print_results(name: str, results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")
    for key, value in results.items():
        if isinstance(value, float):
            if "rate" in key or "throughput" in key:
                print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()


def run_benchmarks(
    include_track_a: bool = False, csv_file: str = None, monitor_interval: float = 0.5
):
    """Run all benchmarks with hardware monitoring."""

    print("=" * 60)
    print(" GPUQueue Benchmark Suite")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Start hardware monitor
    monitor = HardwareMonitor(interval=monitor_interval)
    monitor.start()

    all_results = {}

    try:
        # Warmup
        print("\n[Warmup]")
        benchmark_track_b_throughput(num_messages=100)
        print("  Done")

        # Track B benchmarks
        print("\n[Track B: GPU-Resident Queue]")

        print("  Running throughput benchmark...")
        results = benchmark_track_b_throughput(num_messages=10000, payload_size=256)
        all_results["track_b_throughput"] = results
        print_results("Track B Throughput", results)

        print("  Running latency benchmark...")
        results = benchmark_track_b_latency(num_samples=100)
        all_results["track_b_latency"] = results
        print_results("Track B Latency", results)

        # Track A benchmarks (optional)
        if include_track_a:
            print("\n[Track A: Redis-Backed Queue]")

            print("  Running throughput benchmark...")
            results = benchmark_track_a_throughput(num_messages=5000, payload_size=256)
            all_results["track_a_throughput"] = results
            print_results("Track A Throughput", results)

        # Large payload benchmark
        print("\n[Large Payload Test]")
        print("  Running with 4KB payloads...")
        results = benchmark_track_b_throughput(num_messages=1000, payload_size=4096)
        all_results["track_b_large_payload"] = results
        print_results("Track B (4KB payloads)", results)

    finally:
        # Stop monitoring and get summary
        snapshots = monitor.stop()
        summary = monitor.get_summary()

    # Print hardware summary
    print(summary)

    # Comparison if Track A was run
    if include_track_a and "track_a_throughput" in all_results:
        track_a = all_results["track_a_throughput"]
        track_b = all_results["track_b_throughput"]

        speedup = track_b["rate_msg_s"] / track_a["rate_msg_s"]

        print("\n" + "=" * 60)
        print(" COMPARISON: Track B vs Track A")
        print("=" * 60)
        print(f"  Track A: {track_a['rate_msg_s']:,.0f} msg/s")
        print(f"  Track B: {track_b['rate_msg_s']:,.0f} msg/s")
        print(f"  Speedup: {speedup:.1f}x")
        print("=" * 60)

    # Save to CSV
    if csv_file:
        save_to_csv(snapshots, csv_file)

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPUQueue with hardware monitoring")
    parser.add_argument(
        "--track-a", "-a", action="store_true", help="Include Track A (Redis) benchmarks"
    )
    parser.add_argument(
        "--csv", "-c", type=str, default=None, help="Save hardware metrics to CSV file"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=0.5, help="Hardware sampling interval in seconds"
    )

    args = parser.parse_args()

    run_benchmarks(
        include_track_a=args.track_a,
        csv_file=args.csv,
        monitor_interval=args.interval,
    )


if __name__ == "__main__":
    main()
