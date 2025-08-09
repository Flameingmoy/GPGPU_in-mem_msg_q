# CUDA GPU-Resident Message Queue — API (MVP + Stage-2)

This document defines the public host-facing API and the kernel plug-in interface aligned with `docs/design.md` (§2.2, §5).

Status: Draft (MVP focus). C/C++ signatures shown for clarity; final namespaces and error handling may evolve with the implementation.

## 1) Concepts
- Queue lives in GPU global memory as a ring buffer with fixed-size slots.
- Host enqueues messages (<= slot_bytes). A persistent kernel consumes them and writes results.
- MVP threading: single host producer; one persistent device consumer grid.

See `docs/design.md` for data layout, memory model, and correctness fences.

## 2) Types
```c
// Queue configuration (immutable after init)
typedef struct QueueConfig {
  uint32_t capacity;     // number of slots (power of two)
  uint32_t slot_bytes;   // payload bytes per slot (fixed)
} QueueConfig;

// Opaque handle to a queue instance (host-side)
typedef struct GpuQueue GpuQueue;

typedef struct QueueStats {
  uint64_t enq_count;
  uint64_t deq_count;
  uint64_t dropped_full;
  uint64_t retries;
  uint64_t queue_depth; // head - tail (approx)
} QueueStats;

// Return codes
typedef enum QueueStatus {
  Q_OK = 0,
  Q_ERR_INVALID_ARG = -1,
  Q_ERR_FULL = -2,
  Q_ERR_TIMEOUT = -3,
  Q_ERR_CUDA = -4,
  Q_ERR_SHUTDOWN = -5
} QueueStatus;
```

## 3) Host API (Track B)
```c
// Initialize a GPU-resident queue and start the persistent kernel.
// Returns NULL on failure; use q_last_error() for details.
GpuQueue* q_init(const QueueConfig* cfg);

// Gracefully stop the persistent kernel and free resources.
void q_shutdown(GpuQueue* q);

// Enqueue a message. Copies `len` bytes from host buffer `data` into a free slot.
// On success, returns Q_OK and sets *out_msg_id (monotonic ID). May block up to timeout_ms.
QueueStatus q_enqueue(GpuQueue* q,
                      const void* data,
                      size_t len,
                      uint64_t* out_msg_id,
                      uint32_t timeout_ms);

// Try to read back a result for a specific message ID (if the kernel produces results).
// If available and `*inout_len` is sufficient, copies into out_buf and returns Q_OK.
QueueStatus q_try_dequeue_result(GpuQueue* q,
                                 uint64_t msg_id,
                                 void* out_buf,
                                 size_t* inout_len);

// Snapshot counters and health.
QueueStatus q_stats(GpuQueue* q, QueueStats* out);

// Retrieve human-readable string for last error in the calling thread.
const char* q_last_error(void);
```

Notes:
- `len` must be <= `cfg->slot_bytes`.
- Enqueue may return `Q_ERR_FULL` if the ring buffer is full (no free slots within timeout).
- For high-throughput, batch enqueues and use pinned host memory.

## 4) Kernel plug-in (MVP)
MVP uses a compiled-in device function. Later versions can support a registration mechanism.

```c++
// User-provided device-side processing function signature.
// Implemented by the application and linked with the queue runtime.
struct MsgView {
  const uint8_t* payload; // pointer to slot payload
  uint32_t       len;
  uint64_t       msg_id;
};

struct ResultView {
  uint8_t*  out;      // pointer to slot result area (may alias payload region in MVP)
  uint32_t  cap;      // capacity of result buffer
  uint32_t* out_len;  // device pointer for produced length
};

// Must be defined by the user. Called by the persistent kernel once per READY slot.
__device__ void process_message(const MsgView& in, ResultView* out);
```

Guidelines:
- Device code must write outputs before publishing DONE and follow the fence pattern in `docs/design.md` (§4).
- Keep per-message compute bounded to avoid long busy-waits; use cooperative groups/backoff in the poll loop.

## 5) Memory & Synchronization Contract (summary)
- Host publish (enqueue): copy payload -> ensure order (stream/event) -> publish READY via control block.
- Device complete: write result -> `__threadfence()` -> mark DONE -> advance tail/stats.
- Host-visible updates from device that cross PCIe must use `__threadfence_system()` before signaling host.

Refer to `docs/design.md` §4 for authoritative rules.

## 6) Minimal usage example (host)
```c
QueueConfig cfg{ .capacity = 4096, .slot_bytes = 2048 };
GpuQueue* q = q_init(&cfg);
if (!q) { fprintf(stderr, "init failed: %s\n", q_last_error()); return 1; }

uint8_t payload[512] = {0};
uint64_t msg_id = 0;
QueueStatus st = q_enqueue(q, payload, sizeof(payload), &msg_id, /*timeout_ms=*/10);
if (st != Q_OK) { fprintf(stderr, "enqueue failed: %d\n", st); }

QueueStats s{}; q_stats(q, &s);

q_shutdown(q);
```

## 7) Track A (Redis) surface (high-level)
Track A is a separate worker that:
- Pops N items from Redis (LPUSH/BRPOP or XREADGROUP/XACK).
- Packs to fixed-size blobs aligned to `slot_bytes`.
- Launches GPU kernels on batches using multiple streams.

The exact CLI/SDK will be documented with the implementation of the Redis worker.

## 8) Compatibility
- CUDA Toolkit 12.x; tested on RTX 4070 Ti Super (CC 8.9).
- Linux with recent NVIDIA driver.

## 9) Future extensions
- Multi-producer host API (lock-free MPMC patterns).
- Result subscription callbacks or completion queues.
- Variable-size messages with paging/segmentation.
