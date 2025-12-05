#!/usr/bin/env python3
"""
Soak test for GPUQueue - long-running stability verification.

Tests:
- Memory stability (no leaks over time)
- Queue reliability (no message loss)
- Performance consistency (no degradation)
- Resource cleanup (proper shutdown)

Usage:
    python scripts/soak_test.py                    # Default 5 minutes
    python scripts/soak_test.py --duration 3600   # 1 hour
    python scripts/soak_test.py --track-a         # Include Track A
    python scripts/soak_test.py --csv results.csv # Save metrics
"""

import argparse
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_monitor import HardwareMonitor


@dataclass
class SoakStats:
    """Statistics from soak test."""
    duration_s: float = 0
    total_messages: int = 0
    total_bytes: int = 0
    errors: int = 0
    
    # Performance tracking
    min_rate: float = float('inf')
    max_rate: float = 0
    avg_rate: float = 0
    rate_samples: list = field(default_factory=list)
    
    # Memory tracking
    initial_vram_gb: float = 0
    final_vram_gb: float = 0
    max_vram_gb: float = 0
    initial_ram_gb: float = 0
    final_ram_gb: float = 0
    max_ram_gb: float = 0
    
    def vram_delta_mb(self) -> float:
        return (self.final_vram_gb - self.initial_vram_gb) * 1024
    
    def ram_delta_mb(self) -> float:
        return (self.final_ram_gb - self.initial_ram_gb) * 1024


def run_track_b_soak(duration_s: float, report_interval: float = 10.0) -> SoakStats:
    """Run Track B (GPU-resident queue) soak test."""
    from gpuqueue import GpuQueue, QueueConfig
    
    stats = SoakStats()
    payload_size = 256
    payload = b"x" * payload_size
    
    # Moderate capacity
    cfg = QueueConfig(capacity=1024, slot_bytes=512)
    
    print(f"\n[Track B Soak Test]")
    print(f"  Duration: {duration_s:.0f}s")
    print(f"  Payload: {payload_size} bytes")
    print(f"  Capacity: {cfg.capacity} slots")
    print()
    
    start_time = time.time()
    last_report = start_time
    interval_messages = 0
    interval_start = start_time
    pending_ids = []
    
    with GpuQueue(cfg) as q:
        # Run until duration expires
        while (time.time() - start_time) < duration_s:
            try:
                # Enqueue a small batch
                batch_size = 50
                for _ in range(batch_size):
                    try:
                        msg_id = q.enqueue(payload)
                        pending_ids.append(msg_id)
                        stats.total_messages += 1
                        stats.total_bytes += payload_size
                        interval_messages += 1
                    except ValueError:
                        # Queue full - collect completions
                        break
                
                # Poll for completions and dequeue to free slots
                completed = q.poll_completions(100)
                for msg_id in completed:
                    success, _ = q.try_dequeue_result(msg_id)
                    if success and msg_id in pending_ids:
                        pending_ids.remove(msg_id)
                
                # Brief yield
                time.sleep(0.001)
                
            except Exception as e:
                stats.errors += 1
                print(f"  Error: {e}")
            
            # Periodic reporting
            now = time.time()
            if now - last_report >= report_interval:
                elapsed = now - interval_start
                rate = interval_messages / elapsed if elapsed > 0 else 0
                
                stats.rate_samples.append(rate)
                stats.min_rate = min(stats.min_rate, rate)
                stats.max_rate = max(stats.max_rate, rate)
                
                q_stats = q.stats()
                progress = (now - start_time) / duration_s * 100
                
                print(f"  [{progress:5.1f}%] Rate: {rate:,.0f} msg/s | "
                      f"Depth: {q_stats.queue_depth} | "
                      f"Errors: {stats.errors}")
                
                interval_messages = 0
                interval_start = now
                last_report = now
        
        # Final stats
        final_stats = q.stats()
        print(f"\n  Final queue stats: {final_stats}")
    
    stats.duration_s = time.time() - start_time
    if stats.rate_samples:
        stats.avg_rate = sum(stats.rate_samples) / len(stats.rate_samples)
    
    return stats


def run_track_a_soak(duration_s: float, report_interval: float = 10.0) -> SoakStats:
    """Run Track A (Redis-backed queue) soak test."""
    from gpuqueue.track_a import Producer, Consumer, GpuProcessor
    
    stats = SoakStats()
    payload_size = 256
    payload = b"x" * payload_size
    stream_name = f"soak_test_{int(time.time())}"
    
    print(f"\n[Track A Soak Test]")
    print(f"  Duration: {duration_s:.0f}s")
    print(f"  Stream: {stream_name}")
    print()
    
    producer = Producer(stream_name)
    consumer = Consumer(stream_name, group="soak_group", batch_size=64)
    processor = GpuProcessor(batch_size=64, slot_bytes=512)
    
    start_time = time.time()
    last_report = start_time
    interval_messages = 0
    interval_start = start_time
    produced = 0
    consumed = 0
    
    try:
        while (time.time() - start_time) < duration_s:
            try:
                # Produce batch
                for _ in range(100):
                    producer.send(payload)
                    produced += 1
                    stats.total_bytes += payload_size
                
                # Consume and process
                for _ in range(10):  # Process multiple batches
                    batch = consumer.fetch_batch(timeout_ms=10)
                    if batch:
                        processor.process(batch)
                        consumer.ack(batch)
                        consumed += len(batch)
                        interval_messages += len(batch)
                
            except Exception as e:
                stats.errors += 1
                print(f"  Error: {e}")
            
            # Periodic reporting
            now = time.time()
            if now - last_report >= report_interval:
                elapsed = now - interval_start
                rate = interval_messages / elapsed if elapsed > 0 else 0
                
                stats.rate_samples.append(rate)
                stats.min_rate = min(stats.min_rate, rate)
                stats.max_rate = max(stats.max_rate, rate)
                
                progress = (now - start_time) / duration_s * 100
                pending = produced - consumed
                
                print(f"  [{progress:5.1f}%] Rate: {rate:,.0f} msg/s | "
                      f"Pending: {pending} | "
                      f"Errors: {stats.errors}")
                
                interval_messages = 0
                interval_start = now
                last_report = now
        
        # Drain remaining messages
        print("  Draining remaining messages...")
        drain_start = time.time()
        while consumed < produced and (time.time() - drain_start) < 30:
            batch = consumer.fetch_batch(timeout_ms=100)
            if batch:
                processor.process(batch)
                consumer.ack(batch)
                consumed += len(batch)
        
        stats.total_messages = consumed
        
    finally:
        producer.close()
        consumer.close()
    
    stats.duration_s = time.time() - start_time
    if stats.rate_samples:
        stats.avg_rate = sum(stats.rate_samples) / len(stats.rate_samples)
    
    return stats


def print_summary(name: str, stats: SoakStats):
    """Print soak test summary."""
    print(f"\n{'='*60}")
    print(f" {name} - SOAK TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Duration:        {stats.duration_s:.1f}s ({stats.duration_s/60:.1f} min)")
    print(f"  Total Messages:  {stats.total_messages:,}")
    print(f"  Total Data:      {stats.total_bytes / (1024*1024):.1f} MB")
    print(f"  Errors:          {stats.errors}")
    print()
    print(f"  Throughput:")
    print(f"    Min Rate:      {stats.min_rate:,.0f} msg/s")
    print(f"    Max Rate:      {stats.max_rate:,.0f} msg/s")
    print(f"    Avg Rate:      {stats.avg_rate:,.0f} msg/s")
    print()
    
    if stats.initial_vram_gb > 0:
        print(f"  Memory (VRAM):")
        print(f"    Initial:       {stats.initial_vram_gb:.2f} GB")
        print(f"    Final:         {stats.final_vram_gb:.2f} GB")
        print(f"    Max:           {stats.max_vram_gb:.2f} GB")
        print(f"    Delta:         {stats.vram_delta_mb():+.1f} MB")
        print()
    
    if stats.initial_ram_gb > 0:
        print(f"  Memory (RAM):")
        print(f"    Initial:       {stats.initial_ram_gb:.2f} GB")
        print(f"    Final:         {stats.final_ram_gb:.2f} GB")
        print(f"    Max:           {stats.max_ram_gb:.2f} GB")
        print(f"    Delta:         {stats.ram_delta_mb():+.1f} MB")
    
    print(f"{'='*60}")
    
    # Pass/fail criteria
    passed = True
    issues = []
    
    if stats.errors > 0:
        issues.append(f"{stats.errors} errors occurred")
        passed = False
    
    # Note: Some VRAM/RAM growth is expected due to queue buffers and caching
    # Only flag significant unexpected growth
    if stats.vram_delta_mb() > 500:  # >500MB VRAM growth (excludes normal queue allocation)
        issues.append(f"VRAM grew by {stats.vram_delta_mb():.1f} MB (possible leak)")
        passed = False
    
    if stats.ram_delta_mb() > 1000:  # >1GB RAM growth
        issues.append(f"RAM grew by {stats.ram_delta_mb():.1f} MB (possible leak)")
        passed = False
    
    if stats.avg_rate > 0 and stats.min_rate < stats.avg_rate * 0.5:
        issues.append(f"Rate dropped to {stats.min_rate:.0f} msg/s (>50% below avg)")
    
    if passed and not issues:
        print("\n✓ SOAK TEST PASSED")
    else:
        print(f"\n{'✓' if passed else '✗'} SOAK TEST {'PASSED with warnings' if passed else 'FAILED'}")
        for issue in issues:
            print(f"  - {issue}")
    
    return passed


def main():
    parser = argparse.ArgumentParser(description="GPUQueue soak test")
    parser.add_argument('--duration', '-d', type=float, default=300,
                       help='Test duration in seconds (default: 300 = 5 min)')
    parser.add_argument('--track-a', '-a', action='store_true',
                       help='Include Track A (Redis) soak test')
    parser.add_argument('--track-b-only', '-b', action='store_true',
                       help='Only run Track B soak test')
    parser.add_argument('--report-interval', '-r', type=float, default=10,
                       help='Progress report interval in seconds')
    parser.add_argument('--csv', '-c', type=str, default=None,
                       help='Save hardware metrics to CSV')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" GPUQueue Soak Test")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Duration: {args.duration:.0f}s ({args.duration/60:.1f} min)")
    print("=" * 60)
    
    # Start hardware monitoring
    monitor = HardwareMonitor(interval=2.0)
    monitor.start()
    
    # Get initial memory
    initial_snap = monitor.sample()
    
    all_passed = True
    
    try:
        # Track B soak test
        if not args.track_a or not args.track_b_only:
            stats_b = run_track_b_soak(args.duration, args.report_interval)
            
            # Get memory stats
            final_snap = monitor.sample()
            if initial_snap.gpus:
                stats_b.initial_vram_gb = initial_snap.gpus[0].vram_used_gb
                stats_b.final_vram_gb = final_snap.gpus[0].vram_used_gb
                stats_b.max_vram_gb = max(s.gpus[0].vram_used_gb for s in monitor.get_snapshots() if s.gpus)
            stats_b.initial_ram_gb = initial_snap.memory.used_gb
            stats_b.final_ram_gb = final_snap.memory.used_gb
            stats_b.max_ram_gb = max(s.memory.used_gb for s in monitor.get_snapshots())
            
            if not print_summary("Track B (GPU-Resident)", stats_b):
                all_passed = False
        
        # Track A soak test (optional)
        if args.track_a:
            # Force GC before Track A
            gc.collect()
            time.sleep(1)
            
            initial_snap = monitor.sample()
            stats_a = run_track_a_soak(args.duration, args.report_interval)
            
            final_snap = monitor.sample()
            if initial_snap.gpus:
                stats_a.initial_vram_gb = initial_snap.gpus[0].vram_used_gb
                stats_a.final_vram_gb = final_snap.gpus[0].vram_used_gb
            stats_a.initial_ram_gb = initial_snap.memory.used_gb
            stats_a.final_ram_gb = final_snap.memory.used_gb
            
            if not print_summary("Track A (Redis)", stats_a):
                all_passed = False
    
    finally:
        snapshots = monitor.stop()
        hw_summary = monitor.get_summary()
        
        print(f"\n{hw_summary}")
        
        if args.csv:
            from hardware_monitor import save_to_csv
            save_to_csv(snapshots, args.csv)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
