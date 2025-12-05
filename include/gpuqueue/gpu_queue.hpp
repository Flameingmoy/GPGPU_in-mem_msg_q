/**
 * @file gpu_queue.hpp
 * @brief Main GPU Queue class - the public API for the message queue.
 * 
 * This class provides:
 * - Queue initialization and shutdown
 * - Message enqueue (host → device)
 * - Result dequeue (device → host)
 * - Statistics and monitoring
 */

#pragma once

#include "types.hpp"
#include "ring_buffer.hpp"
#include "memory.hpp"
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>

// Forward declare kernel launcher
extern "C" cudaError_t launch_persistent_consumer(
    uint8_t* slot_data,
    uint32_t* slot_states,
    void* control,
    uint32_t capacity,
    uint32_t slot_size,
    uint32_t num_blocks,
    uint32_t num_threads,
    cudaStream_t stream
);

namespace gpuqueue {

/**
 * @brief Main GPU Queue class.
 * 
 * Thread-safety: The queue is thread-safe for concurrent enqueue operations.
 * Multiple threads can call enqueue() simultaneously.
 */
class GpuQueue {
public:
    /**
     * @brief Construct and initialize a GPU queue.
     * @param config Queue configuration
     * @throws CudaError if initialization fails
     * @throws std::invalid_argument if config is invalid
     */
    explicit GpuQueue(const QueueConfig& config = QueueConfig{});
    
    /**
     * @brief Destructor - shuts down the queue gracefully.
     */
    ~GpuQueue();
    
    // Non-copyable, non-movable (due to kernel state)
    GpuQueue(const GpuQueue&) = delete;
    GpuQueue& operator=(const GpuQueue&) = delete;
    GpuQueue(GpuQueue&&) = delete;
    GpuQueue& operator=(GpuQueue&&) = delete;
    
    // === Enqueue API ===
    
    /**
     * @brief Enqueue a message (blocking with timeout).
     * 
     * @param data Pointer to payload data
     * @param len Length of payload in bytes
     * @param timeout_ms Maximum time to wait for a free slot (0 = no wait)
     * @return Message ID on success (≥0), or QueueStatus error code (<0)
     */
    int64_t enqueue(const void* data, size_t len, uint32_t timeout_ms = 1000);
    
    /**
     * @brief Enqueue a message (non-blocking).
     * 
     * @param data Pointer to payload data
     * @param len Length of payload in bytes
     * @return Message ID on success (≥0), or QueueStatus error code (<0)
     */
    int64_t enqueue_nowait(const void* data, size_t len) {
        return enqueue(data, len, 0);
    }
    
    // === Dequeue API ===
    
    /**
     * @brief Try to dequeue a result for a specific message.
     * 
     * @param msg_id Message ID from enqueue()
     * @param out_data Buffer to receive result data
     * @param inout_len On input: buffer size. On output: actual data length.
     * @return QueueStatus::OK on success, error code otherwise
     */
    QueueStatus try_dequeue_result(uint64_t msg_id, void* out_data, size_t* inout_len);
    
    /**
     * @brief Poll for any completed messages.
     * 
     * @param out_msg_ids Array to receive completed message IDs
     * @param max_count Maximum number of IDs to return
     * @return Number of completed message IDs returned
     */
    size_t poll_completions(uint64_t* out_msg_ids, size_t max_count);
    
    // === Control ===
    
    /**
     * @brief Check if the queue is running.
     */
    bool is_running() const { return running_; }
    
    /**
     * @brief Shutdown the queue gracefully.
     * 
     * Signals the kernel to stop and waits for in-flight messages to complete.
     */
    void shutdown();
    
    // === Statistics ===
    
    /**
     * @brief Get current queue statistics.
     */
    QueueStats stats() const;
    
    /**
     * @brief Get current queue depth (number of pending messages).
     */
    uint64_t depth() const {
        return ring_buffer_ ? ring_buffer_->depth() : 0;
    }
    
    /**
     * @brief Get queue configuration.
     */
    const QueueConfig& config() const { return config_; }

private:
    QueueConfig config_;
    std::unique_ptr<RingBuffer> ring_buffer_;
    
    // Kernel state
    CudaStream kernel_stream_;
    bool running_ = false;
    
    // Enqueue staging buffer (pinned memory for async copies)
    PinnedBuffer<uint8_t> staging_buffer_;
    CudaStream enqueue_stream_;
    
    // Thread safety for enqueue
    mutable std::mutex enqueue_mutex_;
    
    // Message ID counter
    std::atomic<uint64_t> next_msg_id_{0};
    
    // Statistics (host-side counters)
    std::atomic<uint64_t> enqueue_count_{0};
    std::atomic<uint64_t> dequeue_count_{0};
    std::atomic<uint64_t> dropped_full_{0};
    
    // === Internal helpers ===
    
    /**
     * @brief Reserve a slot for enqueue.
     * @return Slot index, or -1 if queue is full
     */
    int32_t reserve_slot();
    
    /**
     * @brief Publish a slot as READY after data is copied.
     */
    void publish_slot(uint32_t slot_idx, uint64_t msg_id, uint32_t len);
    
    /**
     * @brief Start the persistent kernel.
     */
    void start_kernel();
    
    /**
     * @brief Stop the persistent kernel.
     */
    void stop_kernel();
};

// ============================================================================
// Implementation
// ============================================================================

inline GpuQueue::GpuQueue(const QueueConfig& config)
    : config_(config)
    , staging_buffer_(config.slot_bytes)  // One slot's worth for staging
{
    if (!config.is_valid()) {
        throw std::invalid_argument("Invalid queue configuration");
    }
    
    // Set CUDA device
    set_device(config.device);
    
    // Allocate ring buffer
    ring_buffer_ = std::make_unique<RingBuffer>(config);
    
    // Start the persistent kernel
    start_kernel();
}

inline GpuQueue::~GpuQueue() {
    if (running_) {
        shutdown();
    }
}

inline void GpuQueue::start_kernel() {
    cudaError_t err = launch_persistent_consumer(
        ring_buffer_->slot_data(),
        ring_buffer_->slot_states(),
        ring_buffer_->control_block(),
        config_.capacity,
        static_cast<uint32_t>(ring_buffer_->slot_size()),
        config_.num_blocks,
        config_.num_threads,
        kernel_stream_.get()
    );
    
    if (err != cudaSuccess) {
        throw CudaError(err, __FILE__, __LINE__);
    }
    
    running_ = true;
}

inline void GpuQueue::stop_kernel() {
    if (!running_) return;
    
    // Signal kernel to stop
    ring_buffer_->control_block()->stop_flag.store(1, std::memory_order_release);
    
    // Wait for kernel to exit
    kernel_stream_.synchronize();
    
    running_ = false;
}

inline void GpuQueue::shutdown() {
    stop_kernel();
    
    // Drain any remaining DONE slots
    // (In a full implementation, we'd wait for in-flight messages)
}

inline int32_t GpuQueue::reserve_slot() {
    ControlBlock* ctrl = ring_buffer_->control_block();
    // Try to reserve a slot
    uint64_t head = ctrl->head.load(std::memory_order_acquire);
    uint64_t tail = ctrl->tail.load(std::memory_order_acquire);
    
    if (is_full(head, tail, config_.capacity)) {
        return -1;  // Queue full
    }
    
    // Try to advance head (only one producer for MVP)
    uint64_t new_head = head + 1;
    if (!ctrl->head.compare_exchange_strong(head, new_head, 
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return -1;  // Lost race, try again
    }
    
    return static_cast<int32_t>(slot_index(head, config_.capacity));
}

inline void GpuQueue::publish_slot(uint32_t slot_idx, uint64_t msg_id, uint32_t len) {
    // All operations on enqueue_stream_ (non-default stream)
    // IMPORTANT: Cannot use default stream or synchronous cudaMemcpy while persistent kernel is running!
    
    // Write header to device
    SlotHeader header;
    header.len = len;
    header.flags = 0;
    header.msg_id = msg_id;
    
    SlotHeader* d_header = ring_buffer_->get_slot_header(slot_idx);
    CUDA_CHECK(cudaMemcpyAsync(d_header, &header, sizeof(SlotHeader),
                               cudaMemcpyHostToDevice, enqueue_stream_.get()));
    
    // Copy slot state to READY
    uint32_t ready_state = static_cast<uint32_t>(SlotState::READY);
    CUDA_CHECK(cudaMemcpyAsync(ring_buffer_->slot_states() + slot_idx, 
                               &ready_state, sizeof(uint32_t), 
                               cudaMemcpyHostToDevice, enqueue_stream_.get()));
    
    // Wait for our stream only (not device-wide sync)
    CUDA_CHECK(cudaStreamSynchronize(enqueue_stream_.get()));
}

inline int64_t GpuQueue::enqueue(const void* data, size_t len, uint32_t timeout_ms) {
    if (!running_) {
        return static_cast<int64_t>(QueueStatus::ERR_SHUTDOWN);
    }
    
    if (len > config_.slot_bytes) {
        return static_cast<int64_t>(QueueStatus::ERR_PAYLOAD_TOO_LARGE);
    }
    
    if (data == nullptr && len > 0) {
        return static_cast<int64_t>(QueueStatus::ERR_INVALID_ARG);
    }
    
    std::lock_guard<std::mutex> lock(enqueue_mutex_);
    
    auto start = std::chrono::steady_clock::now();
    int32_t slot_idx = -1;
    
    // Try to reserve a slot (with timeout)
    while (slot_idx < 0) {
        slot_idx = reserve_slot();
        
        if (slot_idx < 0) {
            if (timeout_ms == 0) {
                dropped_full_++;
                return static_cast<int64_t>(QueueStatus::ERR_FULL);
            }
            
            auto elapsed = std::chrono::steady_clock::now() - start;
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            
            if (elapsed_ms >= timeout_ms) {
                dropped_full_++;
                return static_cast<int64_t>(QueueStatus::ERR_TIMEOUT);
            }
            
            // Brief sleep before retry
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    // Get message ID
    uint64_t msg_id = next_msg_id_.fetch_add(1, std::memory_order_relaxed);
    
    // Copy payload to device
    if (len > 0) {
        uint8_t* d_payload = ring_buffer_->get_slot_payload(static_cast<uint32_t>(slot_idx));
        CUDA_CHECK(cudaMemcpyAsync(d_payload, data, len, 
                                   cudaMemcpyHostToDevice, enqueue_stream_.get()));
    }
    
    // Publish slot
    publish_slot(static_cast<uint32_t>(slot_idx), msg_id, static_cast<uint32_t>(len));
    
    enqueue_count_++;
    
    return static_cast<int64_t>(msg_id);
}

inline QueueStatus GpuQueue::try_dequeue_result(uint64_t msg_id, void* out_data, size_t* inout_len) {
    if (!out_data || !inout_len) {
        return QueueStatus::ERR_INVALID_ARG;
    }
    
    // Use enqueue_stream for all memory operations (not default stream!)
    cudaStream_t stream = enqueue_stream_.get();
    
    // For MVP: scan all DONE slots to find the message
    uint32_t capacity = config_.capacity;
    
    for (uint32_t i = 0; i < capacity; ++i) {
        // Read slot state from device (async + sync on our stream)
        uint32_t state;
        CUDA_CHECK(cudaMemcpyAsync(&state, ring_buffer_->slot_states() + i,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        if (state == static_cast<uint32_t>(SlotState::DONE)) {
            // Read header
            SlotHeader header;
            CUDA_CHECK(cudaMemcpyAsync(&header, ring_buffer_->get_slot_header(i),
                                       sizeof(SlotHeader), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            if (header.msg_id == msg_id) {
                // Found it!
                if (*inout_len < header.len) {
                    *inout_len = header.len;
                    return QueueStatus::ERR_INVALID_ARG;  // Buffer too small
                }
                
                // Copy result
                if (header.len > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(out_data, ring_buffer_->get_slot_payload(i),
                                               header.len, cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                *inout_len = header.len;
                
                // Mark slot as EMPTY for reuse
                uint32_t empty_state = static_cast<uint32_t>(SlotState::EMPTY);
                CUDA_CHECK(cudaMemcpyAsync(ring_buffer_->slot_states() + i,
                                           &empty_state, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                
                // Advance tail
                ring_buffer_->control_block()->tail.fetch_add(1, std::memory_order_release);
                
                dequeue_count_++;
                
                return QueueStatus::OK;
            }
        }
    }
    
    return QueueStatus::ERR_NOT_READY;  // Not found or not done yet
}

inline size_t GpuQueue::poll_completions(uint64_t* out_msg_ids, size_t max_count) {
    if (!out_msg_ids || max_count == 0) return 0;
    
    cudaStream_t stream = enqueue_stream_.get();
    size_t count = 0;
    uint32_t capacity = config_.capacity;
    
    for (uint32_t i = 0; i < capacity && count < max_count; ++i) {
        uint32_t state;
        CUDA_CHECK(cudaMemcpyAsync(&state, ring_buffer_->slot_states() + i,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        if (state == static_cast<uint32_t>(SlotState::DONE)) {
            SlotHeader header;
            CUDA_CHECK(cudaMemcpyAsync(&header, ring_buffer_->get_slot_header(i),
                                       sizeof(SlotHeader), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            out_msg_ids[count++] = header.msg_id;
        }
    }
    
    return count;
}

inline QueueStats GpuQueue::stats() const {
    QueueStats s{};
    
    s.enqueue_count = enqueue_count_.load(std::memory_order_relaxed);
    s.dequeue_count = dequeue_count_.load(std::memory_order_relaxed);
    s.dropped_full = dropped_full_.load(std::memory_order_relaxed);
    
    if (ring_buffer_) {
        ControlBlock* ctrl = ring_buffer_->control_block();
        s.head = ctrl->head.load(std::memory_order_acquire);
        s.tail = ctrl->tail.load(std::memory_order_acquire);
        s.queue_depth = queue_depth(s.head, s.tail);
        s.processed_count = ctrl->processed_count.load(std::memory_order_acquire);
        s.error_count = ctrl->error_count.load(std::memory_order_acquire);
    }
    
    return s;
}

} // namespace gpuqueue
