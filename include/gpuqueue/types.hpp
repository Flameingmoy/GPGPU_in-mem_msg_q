/**
 * @file types.hpp
 * @brief Core data types for the GPU-resident message queue.
 * 
 * This file defines:
 * - SlotState: State machine for queue slots
 * - SlotHeader: Per-slot metadata
 * - ControlBlock: Shared state between host and device
 * - QueueConfig: Queue configuration parameters
 * - QueueStats: Runtime statistics
 * - Error codes and status types
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>

namespace gpuqueue {

// ============================================================================
// Constants
// ============================================================================

/// Default slot size in bytes (payload only, excluding header)
constexpr size_t DEFAULT_SLOT_BYTES = 1024;

/// Default queue capacity (number of slots, must be power of 2)
constexpr uint32_t DEFAULT_CAPACITY = 4096;

/// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

/// Maximum message payload size
constexpr size_t MAX_PAYLOAD_SIZE = 16 * 1024;  // 16 KB

// ============================================================================
// Slot State Machine
// ============================================================================

/**
 * @brief State machine for queue slots.
 * 
 * Transitions:
 *   EMPTY → READY    (host publishes message)
 *   READY → INFLIGHT (kernel claims for processing)
 *   INFLIGHT → DONE  (kernel completes processing)
 *   DONE → EMPTY     (host reclaims slot after reading result)
 * 
 * This pattern avoids ABA problems because each state has a unique meaning
 * and slots must complete the full cycle before reuse.
 */
enum class SlotState : uint32_t {
    EMPTY = 0,      ///< Slot is free, can be claimed by host for enqueue
    READY = 1,      ///< Message published, waiting for kernel to process
    INFLIGHT = 2,   ///< Kernel is processing this slot
    DONE = 3        ///< Processing complete, result ready for host
};

// ============================================================================
// Slot Header (per-slot metadata)
// ============================================================================

/**
 * @brief Per-slot metadata stored at the beginning of each slot.
 * 
 * Layout: 16 bytes, naturally aligned for GPU access.
 * The payload immediately follows the header in the slot.
 */
struct alignas(16) SlotHeader {
    uint32_t len;       ///< Payload length in bytes (0..slot_bytes)
    uint32_t flags;     ///< Reserved for future use (e.g., priority, type)
    uint64_t msg_id;    ///< Monotonically increasing message ID
};

static_assert(sizeof(SlotHeader) == 16, "SlotHeader must be 16 bytes");
static_assert(alignof(SlotHeader) == 16, "SlotHeader must be 16-byte aligned");

// ============================================================================
// Control Block (shared host/device state)
// ============================================================================

/**
 * @brief Shared control block for host/device synchronization.
 * 
 * This structure is allocated in pinned (page-locked) memory so that:
 * 1. The host can read/write without copying
 * 2. The device can access it via __threadfence_system()
 * 
 * Fields are padded to avoid false sharing between host and device.
 */
struct alignas(CACHE_LINE_SIZE) ControlBlock {
    // === Producer (host) side ===
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> head{0};  ///< Next slot to enqueue
    
    // === Consumer (device) side ===
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> tail{0};  ///< Next slot to consume
    
    // === Control flags ===
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> stop_flag{0};  ///< Signal kernel to exit
    
    // === Statistics (updated by device) ===
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> processed_count{0};  ///< Messages processed
    std::atomic<uint64_t> error_count{0};  ///< Processing errors
    
    // === Timestamps ===
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> last_enqueue_ns{0};
    std::atomic<uint64_t> last_dequeue_ns{0};
};

// Verify control block is properly sized
static_assert(sizeof(ControlBlock) >= 5 * CACHE_LINE_SIZE, 
              "ControlBlock should span multiple cache lines");

// ============================================================================
// Queue Configuration
// ============================================================================

/**
 * @brief Configuration parameters for queue initialization.
 */
struct QueueConfig {
    uint32_t capacity = DEFAULT_CAPACITY;   ///< Number of slots (must be power of 2)
    uint32_t slot_bytes = DEFAULT_SLOT_BYTES;  ///< Payload size per slot
    int device = 0;                          ///< CUDA device ID
    
    // Kernel configuration
    uint32_t num_threads = 256;              ///< Threads per block for persistent kernel
    uint32_t num_blocks = 1;                 ///< Number of blocks (1 for MVP)
    
    // Timeouts
    uint32_t enqueue_timeout_ms = 1000;      ///< Default enqueue timeout
    uint32_t shutdown_timeout_ms = 5000;     ///< Shutdown grace period
    
    /**
     * @brief Validate configuration parameters.
     * @return true if valid, false otherwise
     */
    bool is_valid() const {
        // Capacity must be power of 2
        if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
            return false;
        }
        // Slot size must be reasonable
        if (slot_bytes == 0 || slot_bytes > MAX_PAYLOAD_SIZE) {
            return false;
        }
        // Threads must be valid for CUDA
        if (num_threads == 0 || num_threads > 1024) {
            return false;
        }
        return true;
    }
    
    /**
     * @brief Compute total slot size including header.
     */
    size_t total_slot_size() const {
        return sizeof(SlotHeader) + slot_bytes;
    }
    
    /**
     * @brief Compute total buffer size for all slots.
     */
    size_t total_buffer_size() const {
        return capacity * total_slot_size();
    }
};

// ============================================================================
// Queue Statistics
// ============================================================================

/**
 * @brief Runtime statistics snapshot.
 */
struct QueueStats {
    uint64_t enqueue_count;      ///< Total messages enqueued
    uint64_t dequeue_count;      ///< Total messages dequeued (results retrieved)
    uint64_t processed_count;    ///< Total messages processed by kernel
    uint64_t error_count;        ///< Processing errors
    uint64_t dropped_full;       ///< Messages dropped due to full queue
    uint64_t queue_depth;        ///< Current depth (head - tail)
    uint64_t head;               ///< Current head position
    uint64_t tail;               ///< Current tail position
};

// ============================================================================
// Error Codes
// ============================================================================

/**
 * @brief Status codes returned by queue operations.
 */
enum class QueueStatus : int {
    OK = 0,
    ERR_INVALID_ARG = -1,
    ERR_FULL = -2,
    ERR_TIMEOUT = -3,
    ERR_CUDA = -4,
    ERR_SHUTDOWN = -5,
    ERR_NOT_READY = -6,
    ERR_NOT_FOUND = -7,
    ERR_NOT_INITIALIZED = -8,
    ERR_ALREADY_INITIALIZED = -9,
    ERR_PAYLOAD_TOO_LARGE = -10
};

/**
 * @brief Convert status code to human-readable string.
 */
inline const char* status_to_string(QueueStatus status) {
    switch (status) {
        case QueueStatus::OK: return "OK";
        case QueueStatus::ERR_INVALID_ARG: return "Invalid argument";
        case QueueStatus::ERR_FULL: return "Queue full";
        case QueueStatus::ERR_TIMEOUT: return "Operation timed out";
        case QueueStatus::ERR_CUDA: return "CUDA error";
        case QueueStatus::ERR_SHUTDOWN: return "Queue shutting down";
        case QueueStatus::ERR_NOT_READY: return "Result not ready";
        case QueueStatus::ERR_NOT_FOUND: return "Message not found";
        case QueueStatus::ERR_NOT_INITIALIZED: return "Queue not initialized";
        case QueueStatus::ERR_ALREADY_INITIALIZED: return "Queue already initialized";
        case QueueStatus::ERR_PAYLOAD_TOO_LARGE: return "Payload too large for slot";
        default: return "Unknown error";
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Compute slot index from absolute position.
 * Requires capacity to be a power of two.
 */
inline constexpr uint32_t slot_index(uint64_t pos, uint32_t capacity) {
    return static_cast<uint32_t>(pos & (capacity - 1));
}

/**
 * @brief Compute queue depth (number of items in queue).
 * Works correctly even when head/tail wrap around.
 */
inline constexpr uint64_t queue_depth(uint64_t head, uint64_t tail) {
    return head - tail;
}

/**
 * @brief Check if queue is full.
 */
inline constexpr bool is_full(uint64_t head, uint64_t tail, uint32_t capacity) {
    return queue_depth(head, tail) >= capacity;
}

/**
 * @brief Check if queue is empty.
 */
inline constexpr bool is_empty(uint64_t head, uint64_t tail) {
    return head == tail;
}

} // namespace gpuqueue
