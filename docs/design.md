# CUDA GPU-Resident Message Queue — Design Document (MVP + Stage-2)

Author: Cascade + User
Target GPU: NVIDIA GeForce RTX 4070 Ti Super (Ada, Compute Capability 8.9)
OS: Linux

## 0) Executive Summary
We will build a high-throughput, low-latency, CUDA-based in-memory message queue inspired by Redis semantics but resident in GPU VRAM. To de-risk and iterate quickly, we adopt a two-track plan:

- Track A (MVP validation path): Integrate Redis as the host-side broker and use CUDA kernels for parallel processing. This validates batching, transfer, and kernel scheduling while providing baseline perf/telemetry with minimal systems risk.
- Track B (Stage-2 true GPU queue): Implement a GPU VRAM–resident queue (ring buffer) with CUDA atomics and memory fences, serviced by a persistent kernel, eliminating the Redis runtime path and keeping hot data entirely on the GPU.

Authoritative references are prioritized (redis.io, docs.nvidia.com, NVIDIA cuda-samples). See References at the end.

## 1) Goals, Non-Goals, Assumptions
- Goals (MVP):
  - Single-GPU, single-node, single-process host control.
  - High-throughput enqueue→GPU→dequeue pipeline with batching.
  - At-least-once processing semantics; basic retry/dead-letter.
  - Metrics and profiling hooks (GPU util, queue depth, throughput).
- Non-Goals (MVP):
  - Persistence/durability across restarts (RDB/AOF equivalents).
  - Multi-GPU distribution / NVLink / GPUDirect RDMA.
  - Exactly-once semantics.
- Assumptions:
  - CUDA Toolkit 12.x; RTX 4070 Ti Super (Ada CC 8.9) [NVIDIA CUDA GPUs].
  - Linux with recent NVIDIA driver; Redis only for Track A.

## 2) System Architecture Overview

### 2.1 Track A (Validation): Redis + CUDA Processing
- Components:
  - Redis List/Streams for ingestion and simple reliability (BLPOP/BRPOP or XREADGROUP/XACK) [redis.io docs].
  - Host integration: batch collector, serializer, GPU memory pool, async copies, multi-stream kernel launches, result publisher.
  - CUDA: compute kernels, streams/events, page-locked host memory for fast H2D/D2H.
- Flow:
  1) Producers push to Redis list/stream.
  2) Batch collector BRPOP/XREADGROUP N messages.
  3) Transfer batched payloads to GPU (pinned→device, cudaMemcpyAsync).
  4) Launch kernels in parallel (multiple streams) and collect results.
  5) Publish results/ACK (e.g., XACK or move from processing→done list).
- Why Track A? Immediate end-to-end validation, benchmarking, and operational metrics while we design the VRAM queue data structures and synchronization.

### 2.2 Track B (Stage-2): GPU VRAM–Resident Queue (No Redis)
- Core idea: GPU ring buffer in global memory holding messages; a persistent kernel coordinates producers/consumers using CUDA atomics and fences. Host interacts via a small control block in page-locked host memory (mapped) or via explicit async copies.
- Components:
  - GPU ring buffer: fixed-size slots (power-of-two capacity), 64-bit head/tail indices, per-slot header {len, flags, id}, payload region.
  - Control block (host↔device): atomic counters, status flags, stats; allocated with cudaHostAlloc(..., cudaHostAllocMapped) or managed memory, with careful synchronization.
  - Persistent kernel: long-running kernel polling control block, moving data between control queues and the GPU ring, and invoking user kernels on available messages.
- Flow (Host→GPU enqueue):
  1) Host reserves a slot by atomically incrementing head (in control block) and computing index = head & (capacity-1).
  2) Copy payload into device slot (cudaMemcpyAsync to d_ring[idx]).
  3) Publish metadata (len, id, flags) and update a ready counter/bitmap.
  4) Ensure visibility to device consumers (__threadfence_system on device-side publisher or stream/order guarantees; see §4).
- Flow (GPU consume→process→complete):
  1) Persistent kernel scans ready slots, claims with atomicCAS on slot state.
  2) Runs user kernel(s) on payload; writes result.
  3) Sets slot to done, updates tail and stats with proper fences.

## 3) Data Structures & Memory Layout (Track B)
- Ring buffer (device global memory), capacity C = 2^k slots, alignment 128B:
  - SlotHeader { uint32_t len; uint32_t flags; uint64_t msg_id; } // 16B
  - Payload: up to SLOT_PAYLOAD_MAX bytes (fixed for MVP to avoid fragmentation). Total slot size S.
  - Arrays: d_headers[C], d_payloads[C][SLOT_PAYLOAD_MAX].
- Indices and state:
  - head (host-producer reserve), tail (device-consumer retire). 64-bit to avoid wrap ambiguity.
  - slot_state[C]: ENUM {EMPTY, READY, INFLIGHT, DONE} as uint32_t.
- Control block (host-pinned, mapped or managed):
  - volatile uint64_t host_head; volatile uint64_t dev_tail; volatile uint32_t stop_flag;
  - stats: depth, enq/deq rates, last_error, timestamps.

MVP simplifications:
- Fixed-size payload per slot (e.g., 1–4 KB) — simplifies indexing and avoids a free list.
- Single host-producer, single persistent device-consumer kernel (can scale later).

## 4) Concurrency & Memory Model
- CUDA atomics: Use device atomics for slot claim/release; use host atomics (C11 stdatomic) for host-side counters.
- Fences (key correctness rule):
  - Producer publish: Write payload → __threadfence() → set slot_state=READY (device-side) or order H2D copy before publishing readiness to device via stream-dependence. When host publishes readiness through mapped memory, the GPU consumer must use acquire semantics after seeing READY.
  - Completion: Write results → __threadfence() → set slot_state=DONE → increment tail.
  - Host/device visibility: When crossing the PCIe boundary or host-pinned mappings, use __threadfence_system() on the device side before signaling host-visible flags; on the host side, ensure stream completion (cudaStreamSynchronize or events) before reading device-written host memory [CUDA C++ Programming Guide: Memory Fence Functions].
- Streams/events:
  - Use per-batch streams to pipeline H2D copies and kernels.
  - Persistent kernel polling loop should use backoff (e.g., __nanosleep) to reduce SM burn when idle.

## 5) External API (initial)
- Host API (Track B):
  - init_queue(capacity, slot_bytes)
  - enqueue(const void* data, size_t len, uint64_t* out_msg_id, uint32_t timeout_ms)
  - try_dequeue_result(uint64_t msg_id, void* out_buf, size_t* inout_len)
  - stats(struct QueueStats*)
  - shutdown()
- Kernel plug-in API:
  - A function pointer or functor-like interface invoked by the persistent kernel per READY slot; MVP: a single compiled-in kernel.

## 6) Reliability Semantics
- MVP: At-least-once.
  - If the persistent kernel crashes, host detects watchdog/error and may resubmit READY slots (not marked DONE).
  - Dead-letter policy: if a slot exceeds N retries or T timeout, move to DLQ (host-side list) for inspection.
- Ordering: FIFO per-queue; with multiple GPU workers, ordering not guaranteed across workers (documented).

## 7) Observability & Ops
- Counters: enq_count, deq_count, dropped_full, retries, avg_batch_size, queue_depth.
- Timers: H2D time, kernel time, D2H time (Nsight Systems/Compute markers).
- Health: last CUDA error, last enqueue/consume ts, SM occupancy.

## 8) Performance Targets (initial)
- Enqueue throughput: sized to saturate PCIe H2D and hide transfer with compute via streams.
- GPU utilization: >80% on steady-state workloads sized to SM count.
- Batch sizing: 1K–8K messages (validate experimentally on 4070 Ti Super).

## 9) Track A Details (Redis-backed path)
- Use Redis Lists for MVP (LPUSH/BRPOP) or Streams for consumer groups (XREADGROUP/XACK) [redis.io].
- Reliability caveat: BLPOP removes the element; if consumer dies, message is lost. Use BRPOPLPUSH pattern or Streams with XACK for at-least-once [redis.io/BLPOP, RPOPLPUSH, Streams].
- Optimization: Pipeline pops, use connection pooling. Serialize payloads to fixed-size blobs to align with GPU slot size.

## 10) Risks & Mitigations
- Coherency bugs: Strictly enforce fence patterns; unit tests with randomized producers/consumers and fault injection.
- Persistent kernel starvation: Use cooperative groups or backoff; set watchdog off on Linux or use TCC-like mode where applicable.
- Fragmentation: Avoided by fixed-size slots in MVP; later add variable-sized with paging.
- Host/GPU contention on control block: Align and pad; use 64B cacheline spacing to avoid false sharing.

## 11) Future Work
- Multi-producer on host; multi-consumer persistent kernel grids.
- Multi-GPU and P2P over NVLink/PCIe; GPUDirect RDMA NIC → VRAM path.
- Exactly-once via idempotent processing and dedupe tables (GPU hash-set).
- Persistence (spill to host SSD) and snapshotting.

## 12) References (authoritative)
- Redis:
  - Streams overview: https://redis.io/docs/latest/develop/data-types/streams/
  - XREADGROUP: https://redis.io/docs/latest/commands/xreadgroup/
  - XACK: https://redis.io/docs/latest/commands/xack/
  - BLPOP caveat (loss if client crashes): https://redis.io/docs/latest/commands/blpop/
  - RPOPLPUSH pattern (reliable queue): https://redis.io/docs/latest/commands/rpoplpush/
  - Pub/Sub semantics (at-most-once): https://redis.io/docs/latest/develop/pubsub/
- CUDA (NVIDIA docs):
  - CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - Memory Fence Functions (§10.5): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - Streams & async copies (§6.2.8, §10.27–10.29): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - Atomic Functions (§10.14): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - cuda-samples repo: https://github.com/NVIDIA/cuda-samples (see asyncAPI, simpleIPC, threadFenceReduction)
- GPU:
  - CUDA GPUs (compute capability 8.9 for Ada family): https://developer.nvidia.com/cuda-gpus

---
Notes: This document adopts the user’s Redis+CUDA MVP plan and extends it with a Stage-2 GPU VRAM–resident queue design aligned with CUDA’s memory model and synchronization primitives.
