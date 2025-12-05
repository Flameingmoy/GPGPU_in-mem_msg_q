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

### 4.1 Atomics Strategy
- **libcu++ atomics (CUDA 11+)**: Prefer `cuda::atomic<T>` and `cuda::atomic_ref<T>` from `<cuda/atomic>` for clean memory ordering semantics. These provide explicit `memory_order` parameters (acquire, release, seq_cst).
- **Legacy atomics**: `atomicCAS`, `atomicExch`, `atomicAdd` for device-side operations where libcu++ is not suitable.
- **Host atomics**: C++11 `std::atomic<T>` for host-side counters in control block.

### 4.2 Fence Patterns (Critical Correctness Rules)

| Operation | Pattern | Fence |
|-----------|---------|-------|
| **Producer publish** | Write payload → fence → set READY | `__threadfence()` or stream ordering |
| **Consumer claim** | Read READY → `atomicCAS(READY→INFLIGHT)` | Implicit acquire on atomic |
| **Consumer complete** | Write result → fence → set DONE → advance tail | `__threadfence()` |
| **Host visibility** | Device writes host-visible data → fence | `__threadfence_system()` |

Producer-Consumer Example (validated with NVIDIA docs):
```cuda
// Producer (device-side after H2D copy)
payload[slot] = data;
__threadfence();  // Ensure payload visible before READY
atomicExch(&slot_state[slot], READY);

// Consumer (persistent kernel)
if (atomicCAS(&slot_state[slot], READY, INFLIGHT) == READY) {
    __threadfence();  // Acquire semantics
    process_payload(payload[slot]);
    result[slot] = output;
    __threadfence();  // Ensure result visible before DONE
    atomicExch(&slot_state[slot], DONE);
    atomicAdd(&tail, 1);
}

// Before host-visible signal
__threadfence_system();  // Cross PCIe boundary
atomicExch(&completion_flag, 1);  // Host can read this
```

### 4.3 Avoiding ABA Problem
The slot state machine (EMPTY→READY→INFLIGHT→DONE→EMPTY) avoids ABA because:
- Each state has a unique meaning and can only transition to the next state
- No slot can be reused until it completes the full cycle
- 64-bit head/tail indices avoid wrap ambiguity (2^64 messages before wrap)

### 4.4 Persistent Kernel Design
```cuda
__global__ void persistent_consumer(ControlBlock* ctrl, SlotHeader* headers, 
                                     uint8_t* payloads, uint32_t* states) {
    while (!ctrl->stop_flag) {
        bool found_work = false;
        
        // Scan for READY slots (cooperative across threads)
        for (uint32_t slot = threadIdx.x; slot < capacity; slot += blockDim.x) {
            if (atomicCAS(&states[slot], READY, INFLIGHT) == READY) {
                found_work = true;
                __threadfence();  // Acquire
                
                process_message(&payloads[slot * slot_bytes], headers[slot].len);
                
                __threadfence();  // Release
                atomicExch(&states[slot], DONE);
            }
        }
        
        // Backoff when idle to reduce SM starvation
        if (!found_work) {
            __nanosleep(1000);  // 1 microsecond (requires sm_70+)
        }
    }
}
```

### 4.5 Cooperative Groups (Optional for Multi-Block)
For multi-block persistent kernels, use cooperative groups for grid-wide synchronization:
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void multi_block_consumer(...) {
    cg::grid_group grid = cg::this_grid();
    // ... grid.sync() for grid-wide barrier if needed
}
```
Launch with `cudaLaunchCooperativeKernel()`.

### 4.6 Streams and Pipelining
- Use per-batch CUDA streams to overlap H2D, kernel, D2H
- Pattern for 2 streams:
  ```
  Stream 0: H2D(batch0) → Kernel(batch0) → D2H(batch0)
  Stream 1:              H2D(batch1) → Kernel(batch1) → D2H(batch1)
  ```
- Use `cudaStreamSynchronize` or events to track completion

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
