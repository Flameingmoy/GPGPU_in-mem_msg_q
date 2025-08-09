# CUDA GPU-Resident Message Queue — Testing Plan

Status: Draft. Focus: correctness, reliability (at-least-once), and concurrency safety.

## 1) Test Levels
- Unit (host): ring math, index wrap, state transitions.
- Unit (device): slot claim/release, fences, result publish.
- Integration: end-to-end enqueue→GPU→complete; Redis-backed path.
- Fault-injection: kernel crash/abort, host restart, timeouts/retries.

## 2) Invariants (must-hold)
- 0 ≤ (head - tail) ≤ capacity at all times (mod 2^64 arithmetic).
- Slot state machine: EMPTY -> READY -> INFLIGHT -> DONE -> EMPTY (no skips).
- Publish/complete fence order respected (see `docs/design.md` §4).
- FIFO ordering per-queue (single consumer), unless documented otherwise.

## 3) Host Unit Tests
- Indexing: `(i & (capacity-1))` maps correctly for capacity=2^k.
- Full/empty detection around wrap (randomized sequences).
- Control block counters (atomic increment/read) under contention.

## 4) Device Unit Tests
- Persistent loop claims one READY slot only once (atomicCAS).
- `__threadfence()` before DONE makes payload visible to a verifier kernel.
- `__threadfence_system()` before host-visible signal verified via host read after event.

Tools:
- Compute Sanitizer (racecheck, memcheck):
  - `compute-sanitizer --tool racecheck ./bin/device_tests`
  - `compute-sanitizer --tool memcheck ./bin/device_tests`

## 5) Integration Tests
- Track B: start demo kernel; enqueue N random payloads; verify outputs and ordering; inject delays.
- Track A: Redis worker consumes from list/stream; ack on success; kill/restart worker mid-run; ensure no loss (at-least-once).

## 6) Fault Injection
- Device: trigger a controlled assert in `process_message` for certain msg_ids; expect retry/DLQ.
- Host: simulate enqueue timeout; ensure proper error propagation and metrics.

## 7) Performance/Soak
- Long-run (30–60 min) with steady producer; verify no leaks, counters monotonic, stable latency.
- Measure queue depth oscillations; detect starvation or backpressure bugs.

## 8) CI Hooks (future)
- `ctest -L unit` for host/device unit tests.
- `ctest -L integration` for Track A/B e2e.
- Lint/format, static analysis, and sanitizer jobs.

## 9) Artifacts
- Logs in `logs/` with timestamps.
- JUnit/JSON reports from tests for CI.

## 10) References
- `docs/design.md` §4 (fences), §6 (reliability), §9 (Redis path).
- NVIDIA Compute Sanitizer docs; CUDA Programming Guide.
