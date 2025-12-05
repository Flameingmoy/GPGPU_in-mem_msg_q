/**
 * @file ring_buffer.hpp
 * @brief GPU-resident ring buffer for the message queue.
 * 
 * The ring buffer consists of:
 * - Device memory for slot data (headers + payloads)
 * - Device memory for slot states (atomic uint32_t array)
 * - Pinned host memory for control block (head, tail, stop_flag)
 */

#pragma once

#include "types.hpp"
#include "memory.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace gpuqueue {

/**
 * @brief GPU-resident ring buffer manager.
 * 
 * This class handles allocation and management of:
 * - Device buffer for slot data (SlotHeader + payload for each slot)
 * - Device buffer for slot states (SlotState enum as uint32_t)
 * - Pinned host buffer for control block
 * - CUDA stream for async operations
 */
class RingBuffer {
public:
    /**
     * @brief Construct a ring buffer with the given configuration.
     * @param config Queue configuration (validated before construction)
     * @throws CudaError if allocation fails
     */
    explicit RingBuffer(const QueueConfig& config);
    
    // Non-copyable, movable
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&) noexcept = default;
    RingBuffer& operator=(RingBuffer&&) noexcept = default;
    
    ~RingBuffer() = default;
    
    // === Accessors ===
    
    /// Get device pointer to slot data buffer
    uint8_t* slot_data() { return d_slot_data_.get(); }
    const uint8_t* slot_data() const { return d_slot_data_.get(); }
    
    /// Get device pointer to slot states array
    uint32_t* slot_states() { return d_slot_states_.get(); }
    const uint32_t* slot_states() const { return d_slot_states_.get(); }
    
    /// Get pointer to control block (pinned host memory)
    ControlBlock* control_block() { return h_control_.get(); }
    const ControlBlock* control_block() const { return h_control_.get(); }
    
    /// Get configuration
    const QueueConfig& config() const { return config_; }
    
    /// Get capacity
    uint32_t capacity() const { return config_.capacity; }
    
    /// Get slot size (header + payload)
    size_t slot_size() const { return config_.total_slot_size(); }
    
    /// Get CUDA stream for async operations
    cudaStream_t stream() const { return stream_.get(); }
    
    // === Slot Access ===
    
    /**
     * @brief Get device pointer to a specific slot's header.
     * @param index Slot index (0..capacity-1)
     */
    SlotHeader* get_slot_header(uint32_t index) {
        return reinterpret_cast<SlotHeader*>(d_slot_data_.get() + index * slot_size());
    }
    
    /**
     * @brief Get device pointer to a specific slot's payload.
     * @param index Slot index (0..capacity-1)
     */
    uint8_t* get_slot_payload(uint32_t index) {
        return d_slot_data_.get() + index * slot_size() + sizeof(SlotHeader);
    }
    
    // === State Management ===
    
    /**
     * @brief Reset all slots to EMPTY state.
     * Should be called during initialization.
     */
    void reset();
    
    /**
     * @brief Get current queue depth.
     */
    uint64_t depth() const {
        return queue_depth(
            h_control_.get()->head.load(std::memory_order_acquire),
            h_control_.get()->tail.load(std::memory_order_acquire)
        );
    }
    
    /**
     * @brief Check if queue is full.
     */
    bool is_full() const {
        return gpuqueue::is_full(
            h_control_.get()->head.load(std::memory_order_acquire),
            h_control_.get()->tail.load(std::memory_order_acquire),
            config_.capacity
        );
    }
    
    /**
     * @brief Check if queue is empty.
     */
    bool is_empty() const {
        return gpuqueue::is_empty(
            h_control_.get()->head.load(std::memory_order_acquire),
            h_control_.get()->tail.load(std::memory_order_acquire)
        );
    }

private:
    QueueConfig config_;
    
    // Device memory
    DeviceBuffer<uint8_t> d_slot_data_;    ///< Slot headers + payloads
    DeviceBuffer<uint32_t> d_slot_states_; ///< Per-slot state (SlotState as uint32_t)
    
    // Pinned host memory for control block (accessible by both host and device)
    PinnedBuffer<ControlBlock> h_control_;
    
    // CUDA stream for async operations
    CudaStream stream_;
};

// ============================================================================
// Implementation
// ============================================================================

inline RingBuffer::RingBuffer(const QueueConfig& config)
    : config_(config)
    , d_slot_data_(config.capacity * config.total_slot_size())
    , d_slot_states_(config.capacity)
    , h_control_(1)
    , stream_()
{
    if (!config.is_valid()) {
        throw std::invalid_argument("Invalid queue configuration");
    }
    
    // Initialize control block
    new (h_control_.get()) ControlBlock{};
    
    // Reset all slots to EMPTY
    reset();
}

inline void RingBuffer::reset() {
    // Zero out slot data
    CUDA_CHECK(cudaMemsetAsync(d_slot_data_.get(), 0, d_slot_data_.bytes(), stream_));
    
    // Set all slot states to EMPTY (0)
    CUDA_CHECK(cudaMemsetAsync(d_slot_states_.get(), 0, d_slot_states_.bytes(), stream_));
    
    // Reset control block counters
    ControlBlock* ctrl = h_control_.get();
    ctrl->head.store(0, std::memory_order_release);
    ctrl->tail.store(0, std::memory_order_release);
    ctrl->stop_flag.store(0, std::memory_order_release);
    ctrl->processed_count.store(0, std::memory_order_release);
    ctrl->error_count.store(0, std::memory_order_release);
    
    // Wait for memsets to complete
    stream_.synchronize();
}

} // namespace gpuqueue
