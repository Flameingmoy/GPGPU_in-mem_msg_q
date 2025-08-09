# CUDA GPU-Resident Message Queue — Use Cases & Test Scenarios

Status: Draft. This document enumerates practical workloads we will use to validate functionality, reliability, and performance across Track A (Redis-backed MVP) and Track B (GPU VRAM–resident queue).

See also: `docs/design.md` (§2.2, §4–§9), `docs/testing.md`, `docs/benchmarking.md`.

## 1) Scope and Assumptions
- Single-GPU, single-node. RTX 4070 Ti Super (CC 8.9), CUDA 12.x.
- Fixed-size payloads per slot for MVP (1–4 KB recommended). At-least-once semantics.
- Track A uses Redis List/Streams; Track B uses the persistent-kernel GPU ring buffer.

## 2) Use Cases

### UC-1: Telemetry/IoT small messages
- Shape: 256–1024 B records; high rate, low compute per message.
- Goal: Sustained 50k–200k msg/s; p95 end-to-end latency < 10 ms under steady-state.
- Track A: Validate ingest/pop pipelines, batching to GPU, overlap via streams.
- Track B: Validate ring throughput and poll loop backoff without SM starvation.
- Acceptance: No loss (at-least-once), throughput within 10% of target, queue depth stable.

### UC-2: Micro-batch analytics (vector ops)
- Shape: 1 KB payloads; micro-batches of 512–4096 messages.
- Goal: High GPU occupancy; overlap H2D/Kernel/D2H ≥ 60%.
- Track A: Batch packing aligns to `slot_bytes`; multi-stream launch.
- Track B: Persistent kernel consumes slots in waves; measure per-batch latency.
- Acceptance: Nsight Systems shows pipeline overlap; p95 latency stable across batch sizes.

### UC-3: Financial ticks with per-key ordering
- Shape: ≤ 512 B; per-symbol FIFO is required, at-least-once.
- Goal: Preserve ordering per key while keeping throughput high.
- Track A: Use Streams/consumer-group or RPOPLPUSH + per-key partitioning.
- Track B: Simulate per-key ordering via hashing to sub-queues or tagged slots (future extension noted).
- Acceptance: For a chosen key set, order never violated in results; retries do not reorder.

### UC-4: Inference pre/post-processing (CPU↔GPU pipeline)
- Shape: 1–4 KB payloads (tokenized text, metadata); moderate compute per message.
- Goal: Keep GPU util > 80% with steady producers; hide transfers via streams.
- Track A: Producer packs and overlaps; measure kernel+copy times.
- Track B: Persistent kernel invokes `process_message` and writes result in-slot.
- Acceptance: Utilization > 80% in steady state; p95 latency within target per payload.

### UC-5: Reliability under faults (at-least-once)
- Shape: 512 B; inject 1–5% kernel failures by msg_id.
- Goal: Retries occur; DLQ populated when exceeding N retries.
- Track A: Use Redis processing list/Streams XACK to re-deliver.
- Track B: READY slots not marked DONE are retried by persistent kernel.
- Acceptance: No message loss; DLQ count = intended failures beyond policy; clear metrics.

### UC-6: Backpressure and overflow behavior
- Shape: 512 B; producer faster than consumer.
- Goal: System applies backpressure; no crash; metrics reflect drops/full.
- Track A: Producer throttles on queue depth (Redis key size) or timeout.
- Track B: `q_enqueue` returns `Q_ERR_FULL` or blocks per timeout.
- Acceptance: Head−tail bounded by capacity; logs/metrics show backpressure, zero corruption.

### UC-7: Cold/warm start and shutdown
- Shape: 1 KB; small batches.
- Goal: Kernel startup time acceptable; graceful `shutdown()` drains work.
- Track A: Worker handles signals; acknowledges in-flight tasks.
- Track B: Persistent kernel observes stop flag after finishing INFLIGHT slots.
- Acceptance: No stranded messages; clean exit; time-to-ready recorded.

### UC-8: Python client integration (package UX)
- Shape: 256–2048 B; Python app submits tasks and polls results.
- Goal: Simple `pip install` experience; stable Python API; backend selectable (Redis vs VRAM).
- Track A: Python client uses Redis (`redis-py`) + GPU worker.
- Track B: Python client calls native extension (pybind11) exposing `enqueue()`/`stats()`.
- Acceptance: Single API used in Python examples for both backends; identical message schema.

## 3) Test Matrix (how we validate)
- Payload sizes: 256 B, 512 B, 1 KB, 2 KB, 4 KB.
- Batches: 256, 1k, 4k, 8k messages.
- Failure rate: 0%, 1%, 5%.
- Streams (Track A): 1, 2, 4, 8.
- Metrics captured: throughput, p50/p95/p99 latency, queue depth, GPU util, H2D/D2H BW.

## 4) Acceptance Criteria (global)
- At-least-once delivery across all tests (no silent loss).
- No data races or invalid accesses (Compute Sanitizer clean).
- GPU util ≥ 80% on steady-state compute-heavy workloads.
- Overlap ≥ 60% for pipeline workloads (Nsight Systems).
- Backpressure works (no unbounded queue growth), error codes surfaced.

## 5) Notes and Future Extensions
- Large payloads (> 4 KB) require variable-size paging or larger `slot_bytes` (Stage-2+).
- Per-key ordering on Track B may use sharded rings or per-key sequence numbers.
- Exactly-once semantics deferred; require idempotency and dedupe tables.
