#!/usr/bin/env python3
"""
Hardware monitoring script for GPUQueue benchmarks.

Monitors:
- CPU utilization (per-core and overall)
- RAM usage (used, available, percent)
- GPU utilization (compute, memory)
- VRAM usage (used, free, total)

Usage:
    # Standalone monitoring
    python scripts/hardware_monitor.py
    
    # With custom interval and duration
    python scripts/hardware_monitor.py --interval 0.5 --duration 60
    
    # Output to CSV
    python scripts/hardware_monitor.py --csv metrics.csv
    
    # As a library
    from scripts.hardware_monitor import HardwareMonitor
    with HardwareMonitor() as mon:
        # ... run workload ...
        stats = mon.get_summary()
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with: pip install psutil")

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


@dataclass
class CPUMetrics:
    """CPU metrics snapshot."""
    timestamp: float
    percent_overall: float
    percent_per_core: list[float]
    
    def __str__(self):
        return f"CPU: {self.percent_overall:5.1f}%"


@dataclass
class MemoryMetrics:
    """RAM metrics snapshot."""
    timestamp: float
    total_gb: float
    used_gb: float
    available_gb: float
    percent: float
    
    def __str__(self):
        return f"RAM: {self.used_gb:.1f}/{self.total_gb:.1f} GB ({self.percent:.1f}%)"


@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""
    timestamp: float
    device_id: int
    name: str
    gpu_util_percent: float
    memory_util_percent: float
    vram_used_gb: float
    vram_free_gb: float
    vram_total_gb: float
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    
    def __str__(self):
        s = f"GPU[{self.device_id}]: {self.gpu_util_percent:5.1f}% | VRAM: {self.vram_used_gb:.2f}/{self.vram_total_gb:.2f} GB ({self.memory_util_percent:.1f}%)"
        if self.temperature_c is not None:
            s += f" | {self.temperature_c:.0f}°C"
        if self.power_w is not None:
            s += f" | {self.power_w:.0f}W"
        return s


@dataclass
class HardwareSnapshot:
    """Complete hardware snapshot."""
    timestamp: float
    cpu: CPUMetrics
    memory: MemoryMetrics
    gpus: list[GPUMetrics]
    
    def __str__(self):
        lines = [
            f"[{datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S.%f')[:-3]}]",
            f"  {self.cpu}",
            f"  {self.memory}",
        ]
        for gpu in self.gpus:
            lines.append(f"  {gpu}")
        return "\n".join(lines)


@dataclass
class HardwareSummary:
    """Summary statistics from monitoring session."""
    duration_s: float
    samples: int
    
    # CPU
    cpu_avg: float
    cpu_max: float
    cpu_min: float
    
    # Memory
    ram_avg_gb: float
    ram_max_gb: float
    ram_avg_percent: float
    
    # GPU (per device)
    gpu_util_avg: list[float] = field(default_factory=list)
    gpu_util_max: list[float] = field(default_factory=list)
    vram_avg_gb: list[float] = field(default_factory=list)
    vram_max_gb: list[float] = field(default_factory=list)
    
    def __str__(self):
        lines = [
            "=" * 60,
            "HARDWARE SUMMARY",
            "=" * 60,
            f"Duration: {self.duration_s:.1f}s ({self.samples} samples)",
            "",
            "CPU:",
            f"  Avg: {self.cpu_avg:.1f}%  Max: {self.cpu_max:.1f}%  Min: {self.cpu_min:.1f}%",
            "",
            "RAM:",
            f"  Avg: {self.ram_avg_gb:.2f} GB ({self.ram_avg_percent:.1f}%)  Max: {self.ram_max_gb:.2f} GB",
        ]
        
        for i, (util_avg, util_max, vram_avg, vram_max) in enumerate(
            zip(self.gpu_util_avg, self.gpu_util_max, self.vram_avg_gb, self.vram_max_gb)
        ):
            lines.extend([
                "",
                f"GPU[{i}]:",
                f"  Util Avg: {util_avg:.1f}%  Max: {util_max:.1f}%",
                f"  VRAM Avg: {vram_avg:.2f} GB  Max: {vram_max:.2f} GB",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


class HardwareMonitor:
    """
    Monitor CPU, RAM, and GPU hardware usage.
    
    Can be used as a context manager for automatic start/stop.
    """
    
    def __init__(self, interval: float = 1.0, gpu_devices: Optional[list[int]] = None):
        """
        Initialize the hardware monitor.
        
        Args:
            interval: Sampling interval in seconds
            gpu_devices: List of GPU device IDs to monitor (None = all)
        """
        self.interval = interval
        self.gpu_devices = gpu_devices
        
        self._snapshots: list[HardwareSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Initialize NVML if available
        self._nvml_initialized = False
        self._gpu_handles = []
        self._init_nvml()
    
    def _init_nvml(self):
        """Initialize NVIDIA Management Library."""
        if not HAS_PYNVML:
            return
        
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            
            device_count = pynvml.nvmlDeviceGetCount()
            devices = self.gpu_devices if self.gpu_devices else list(range(device_count))
            
            for i in devices:
                if i < device_count:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self._gpu_handles.append((i, handle))
        except Exception as e:
            print(f"Warning: Failed to initialize NVML: {e}")
            self._nvml_initialized = False
    
    def _shutdown_nvml(self):
        """Shutdown NVML."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            self._nvml_initialized = False
    
    def _get_cpu_metrics(self) -> CPUMetrics:
        """Get current CPU metrics."""
        if not HAS_PSUTIL:
            return CPUMetrics(
                timestamp=time.time(),
                percent_overall=0.0,
                percent_per_core=[],
            )
        
        return CPUMetrics(
            timestamp=time.time(),
            percent_overall=psutil.cpu_percent(interval=None),
            percent_per_core=psutil.cpu_percent(interval=None, percpu=True),
        )
    
    def _get_memory_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        if not HAS_PSUTIL:
            return MemoryMetrics(
                timestamp=time.time(),
                total_gb=0.0,
                used_gb=0.0,
                available_gb=0.0,
                percent=0.0,
            )
        
        mem = psutil.virtual_memory()
        return MemoryMetrics(
            timestamp=time.time(),
            total_gb=mem.total / (1024**3),
            used_gb=mem.used / (1024**3),
            available_gb=mem.available / (1024**3),
            percent=mem.percent,
        )
    
    def _get_gpu_metrics(self) -> list[GPUMetrics]:
        """Get current GPU metrics."""
        gpus = []
        
        if self._nvml_initialized:
            for device_id, handle in self._gpu_handles:
                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Optional metrics
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None
                    
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    except:
                        power = None
                    
                    gpus.append(GPUMetrics(
                        timestamp=time.time(),
                        device_id=device_id,
                        name=name,
                        gpu_util_percent=util.gpu,
                        memory_util_percent=util.memory,
                        vram_used_gb=mem.used / (1024**3),
                        vram_free_gb=mem.free / (1024**3),
                        vram_total_gb=mem.total / (1024**3),
                        temperature_c=temp,
                        power_w=power,
                    ))
                except Exception as e:
                    print(f"Warning: Failed to get GPU {device_id} metrics: {e}")
        else:
            # Fallback to nvidia-smi
            gpus = self._get_gpu_metrics_nvidia_smi()
        
        return gpus
    
    def _get_gpu_metrics_nvidia_smi(self) -> list[GPUMetrics]:
        """Fallback GPU metrics using nvidia-smi."""
        gpus = []
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        device_id = int(parts[0])
                        
                        # Skip if not in requested devices
                        if self.gpu_devices and device_id not in self.gpu_devices:
                            continue
                        
                        gpus.append(GPUMetrics(
                            timestamp=time.time(),
                            device_id=device_id,
                            name=parts[1],
                            gpu_util_percent=float(parts[2]) if parts[2] != '[N/A]' else 0,
                            memory_util_percent=float(parts[3]) if parts[3] != '[N/A]' else 0,
                            vram_used_gb=float(parts[4]) / 1024 if parts[4] != '[N/A]' else 0,
                            vram_free_gb=float(parts[5]) / 1024 if parts[5] != '[N/A]' else 0,
                            vram_total_gb=float(parts[6]) / 1024 if parts[6] != '[N/A]' else 0,
                            temperature_c=float(parts[7]) if len(parts) > 7 and parts[7] != '[N/A]' else None,
                            power_w=float(parts[8]) if len(parts) > 8 and parts[8] != '[N/A]' else None,
                        ))
        except Exception as e:
            print(f"Warning: nvidia-smi failed: {e}")
        
        return gpus
    
    def sample(self) -> HardwareSnapshot:
        """Take a single hardware snapshot."""
        ts = time.time()
        return HardwareSnapshot(
            timestamp=ts,
            cpu=self._get_cpu_metrics(),
            memory=self._get_memory_metrics(),
            gpus=self._get_gpu_metrics(),
        )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        # Prime CPU measurement
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None)
        
        while self._running:
            snapshot = self.sample()
            with self._lock:
                self._snapshots.append(snapshot)
            time.sleep(self.interval)
    
    def start(self):
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._snapshots = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> list[HardwareSnapshot]:
        """Stop monitoring and return all snapshots."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        with self._lock:
            return list(self._snapshots)
    
    def get_snapshots(self) -> list[HardwareSnapshot]:
        """Get current snapshots (thread-safe copy)."""
        with self._lock:
            return list(self._snapshots)
    
    def get_summary(self) -> HardwareSummary:
        """Compute summary statistics from collected snapshots."""
        snapshots = self.get_snapshots()
        
        if not snapshots:
            return HardwareSummary(
                duration_s=0,
                samples=0,
                cpu_avg=0, cpu_max=0, cpu_min=0,
                ram_avg_gb=0, ram_max_gb=0, ram_avg_percent=0,
            )
        
        duration = snapshots[-1].timestamp - snapshots[0].timestamp if len(snapshots) > 1 else 0
        
        # CPU stats
        cpu_values = [s.cpu.percent_overall for s in snapshots]
        
        # Memory stats
        ram_values = [s.memory.used_gb for s in snapshots]
        ram_percent = [s.memory.percent for s in snapshots]
        
        # GPU stats (per device)
        num_gpus = len(snapshots[0].gpus) if snapshots[0].gpus else 0
        gpu_util_avg = []
        gpu_util_max = []
        vram_avg = []
        vram_max = []
        
        for i in range(num_gpus):
            utils = [s.gpus[i].gpu_util_percent for s in snapshots if len(s.gpus) > i]
            vrams = [s.gpus[i].vram_used_gb for s in snapshots if len(s.gpus) > i]
            
            if utils:
                gpu_util_avg.append(sum(utils) / len(utils))
                gpu_util_max.append(max(utils))
            if vrams:
                vram_avg.append(sum(vrams) / len(vrams))
                vram_max.append(max(vrams))
        
        return HardwareSummary(
            duration_s=duration,
            samples=len(snapshots),
            cpu_avg=sum(cpu_values) / len(cpu_values),
            cpu_max=max(cpu_values),
            cpu_min=min(cpu_values),
            ram_avg_gb=sum(ram_values) / len(ram_values),
            ram_max_gb=max(ram_values),
            ram_avg_percent=sum(ram_percent) / len(ram_percent),
            gpu_util_avg=gpu_util_avg,
            gpu_util_max=gpu_util_max,
            vram_avg_gb=vram_avg,
            vram_max_gb=vram_max,
        )
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
        self._shutdown_nvml()
    
    def __del__(self):
        self._shutdown_nvml()


def save_to_csv(snapshots: list[HardwareSnapshot], filename: str):
    """Save snapshots to CSV file."""
    if not snapshots:
        return
    
    num_gpus = len(snapshots[0].gpus)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['timestamp', 'cpu_percent', 'ram_used_gb', 'ram_percent']
        for i in range(num_gpus):
            header.extend([
                f'gpu{i}_util', f'gpu{i}_mem_util',
                f'gpu{i}_vram_used_gb', f'gpu{i}_temp_c', f'gpu{i}_power_w'
            ])
        writer.writerow(header)
        
        # Data
        for s in snapshots:
            row = [
                s.timestamp,
                s.cpu.percent_overall,
                s.memory.used_gb,
                s.memory.percent,
            ]
            for gpu in s.gpus:
                row.extend([
                    gpu.gpu_util_percent,
                    gpu.memory_util_percent,
                    gpu.vram_used_gb,
                    gpu.temperature_c or '',
                    gpu.power_w or '',
                ])
            writer.writerow(row)
    
    print(f"Saved {len(snapshots)} samples to {filename}")


def live_monitor(interval: float = 1.0, duration: Optional[float] = None, csv_file: Optional[str] = None):
    """Run live monitoring with console output."""
    monitor = HardwareMonitor(interval=interval)
    
    print("Hardware Monitor")
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    
    try:
        while True:
            snapshot = monitor.sample()
            
            # Clear line and print
            print(f"\033[2K{snapshot}", end='\n\n')
            
            with monitor._lock:
                monitor._snapshots.append(snapshot)
            
            if duration and (time.time() - start_time) >= duration:
                break
            
            time.sleep(interval)
            
            # Move cursor up for overwrite effect
            lines = str(snapshot).count('\n') + 2
            print(f"\033[{lines}A", end='')
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    # Final summary
    print("\n" * 5)  # Clear display area
    summary = monitor.get_summary()
    print(summary)
    
    # Save CSV if requested
    if csv_file:
        save_to_csv(monitor.get_snapshots(), csv_file)
    
    monitor._shutdown_nvml()


def main():
    parser = argparse.ArgumentParser(description="Hardware monitoring for GPUQueue")
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                       help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                       help='Monitoring duration in seconds (default: infinite)')
    parser.add_argument('--csv', '-c', type=str, default=None,
                       help='Output CSV file for metrics')
    parser.add_argument('--once', '-1', action='store_true',
                       help='Take a single snapshot and exit')
    
    args = parser.parse_args()
    
    if args.once:
        monitor = HardwareMonitor()
        snapshot = monitor.sample()
        print(snapshot)
        monitor._shutdown_nvml()
    else:
        live_monitor(
            interval=args.interval,
            duration=args.duration,
            csv_file=args.csv,
        )


if __name__ == '__main__':
    main()
