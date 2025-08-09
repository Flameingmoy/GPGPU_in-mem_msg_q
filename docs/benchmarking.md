# CUDA GPU-Resident Message Queue — Benchmarking Guide

Status: Draft. Goal: quantify throughput, latency, GPU util; guide tuning (batch size, streams, slot size).

## 1) Hardware/Software Baseline
- GPU: RTX 4070 Ti Super (CC 8.9)
- Driver + CUDA Toolkit 12.x
- CPU/RAM, PCIe Gen (document in results)
- Redis version (Track A)

## 2) Metrics
- Throughput: msgs/sec (enqueue), msgs/sec (processed)
- Latency: enqueue→DONE (p50/p95/p99)
- Queue depth over time
- GPU: SM occupancy, DRAM BW, achieved occupancy
- H2D/D2H bandwidth and overlap percentage

## 3) Tools
- Nsight Systems: timeline overlap (`nsys profile -t cuda,nvtx -o profile ./bin/gpu_queue_demo ...`)
- Nsight Compute: kernel metrics (`ncu --set speedOfLight ./bin/gpu_queue_demo ...`)
- nvidia-smi dmon: `nvidia-smi dmon -s pucvmet -i 0 1`
- App metrics: logs or HTTP endpoint

## 4) Workloads
- W1: Fixed-size payloads (e.g., 512B, 1KB, 2KB, 4KB)
- W2: Vary batch size (256, 1K, 4K, 8K)
- W3: Streams (Track A) 1, 2, 4, 8
- W4: Compute intensity sweep in `process_message` (light → heavy)

## 5) Procedure (Track B)
1) Launch demo with given capacity/slot bytes.
2) Run `enqueue_bench` with N messages and chosen size.
3) Record: throughput, latency histogram, queue depth.
4) Capture an `nsys` trace for each run and annotate parameters.

## 6) Procedure (Track A)
1) Start Redis and worker with `--streams` and `--batch` set.
2) Run producer at various rates; ensure steady-state (not backlog) except when testing backpressure.
3) Record consumer throughput, Redis queue depth, GPU util.

## 7) Analysis
- Plot throughput vs payload and batch size.
- Identify transfer-bound vs compute-bound regimes (from `nsys`/`ncu`).
- Tune: increase streams, adjust block size, right-size slot_bytes.

## 8) Reporting Template
- Hardware/driver/CUDA versions
- App commit hash/config
- Workload (payload, batch, streams)
- Results: throughput, latency (p50/p95/p99), GPU util
- Notes (bottlenecks, anomalies)

## 9) Targets (initial, refine empirically)
- GPU util > 80% on steady-state
- Overlap ≥ 60% for H2D/Kernel/D2H

## 10) References
- `docs/design.md` §§ 2, 7, 8
- NVIDIA Nsight Systems/Compute docs
