# CUDA GPU-Resident Message Queue — Runbook

Status: Draft. Target: Linux + CUDA 12.x on RTX 4070 Ti Super (CC 8.9). See `docs/design.md` for architecture and `docs/api.md` for API.

## 1) Prerequisites
- NVIDIA driver supporting CUDA 12.x (`nvidia-smi` should list the GPU)
- CUDA Toolkit 12.x (`nvcc --version`)
- Nsight Systems/Compute (optional but recommended)
- Redis (Track A only)
- Build toolchain (gcc/clang, CMake/Make)

## 2) Environment Checks
- GPU presence: `nvidia-smi`
- CUDA: `nvcc --version`
- Persistence mode (optional): `sudo nvidia-smi -pm 1`
- Compute mode (optional to reduce contention): `sudo nvidia-smi -c EXCLUSIVE_PROCESS`

Notes:
- Long-running CUDA kernels are fine on Linux; avoid running heavy compute on the display GPU to keep desktop responsive.
- Use a headless/secondary GPU for best results.

## 3) Quickstart
### Track A (Redis-backed MVP)
1) Start Redis: `redis-server` (or Docker `redis:latest`).
2) Build the worker (once the code is added): `mkdir -p build && cmake .. && make -j`.
3) Run worker: `./bin/redis_gpu_worker --queue task_queue --streams 4 --batch 4096`.
4) Produce data: `./bin/producer --queue task_queue --rate 50000`.
5) Observe: `nvidia-smi dmon -i 0 1` and application logs/metrics.

### Track B (GPU VRAM Queue)
1) Build queue runtime and demo app: `mkdir -p build && cmake .. && make -j`.
2) Start demo (persistent kernel inside): `./bin/gpu_queue_demo --capacity 4096 --slot-bytes 2048`.
3) In another terminal, run enqueuer: `./bin/enqueue_bench --count 100000 --size 512`.
4) Observe throughput/latency in logs and via Nsight.

## 4) Configuration
- QUEUE_CAPACITY (power of two, default 4096)
- SLOT_BYTES (payload per slot, default 2048)
- NUM_STREAMS (Track A), default 4–8
- LOG_LEVEL (trace|debug|info|warn|error)
- METRICS_PORT (if HTTP metrics exposed)

## 5) Observability
- Nsight Systems (timeline): `nsys profile -t cuda,nvtx -o run ./bin/gpu_queue_demo ...`
- Nsight Compute (kernel metrics): `ncu --set full --target-processes all ./bin/gpu_queue_demo ...`
- GPU telemetry: `nvidia-smi dmon -s pucvmet -i 0 1`
- App logs/metrics: see `logs/` or metrics endpoint.

## 6) Troubleshooting
- Q_ERR_FULL: ring full. Increase capacity, slot_bytes, or slow producers. Verify consumer is running.
- cudaErrorIllegalAddress / invalid memory access: run with Compute Sanitizer: `compute-sanitizer ./bin/gpu_queue_demo`.
- Low GPU util: increase batch size, num streams; check PCIe bandwidth; ensure pinned memory for H2D/D2H.
- Starvation/Busy wait: ensure poll loop backoff; see design §4 (fences) and §10 (risks).
- Redis path reliability: prefer RPOPLPUSH/Streams with XACK for at-least-once (see `docs/design.md` §9).

## 7) Safety & Ops Notes
- Use EXCLUSIVE_PROCESS to minimize contention.
- Validate fence patterns when modifying device code (see `docs/design.md` §4).
- Keep drivers and CUDA Toolkit versions aligned. Rebuild after driver upgrades.

## 8) Next Steps
- Link exact binary names/flags once code scaffolding lands.
- Add scripts in `scripts/` for setup, profiling, and log collection.
