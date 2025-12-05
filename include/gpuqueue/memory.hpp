/**
 * @file memory.hpp
 * @brief CUDA memory management utilities for GPUQueue.
 * 
 * Provides RAII wrappers for:
 * - Device memory allocation
 * - Pinned (page-locked) host memory allocation
 * - Async copy operations with CUDA streams
 * - CUDA stream pool for multi-stream pipelining
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

namespace gpuqueue {

// ============================================================================
// Error Handling
// ============================================================================

/**
 * @brief Exception class for CUDA errors.
 */
class CudaError : public std::runtime_error {
public:
    explicit CudaError(cudaError_t err, const char* file, int line)
        : std::runtime_error(format_error(err, file, line)), error_(err) {}
    
    cudaError_t error() const noexcept { return error_; }

private:
    static std::string format_error(cudaError_t err, const char* file, int line) {
        return std::string(file) + ":" + std::to_string(line) + ": " + cudaGetErrorString(err);
    }
    cudaError_t error_;
};

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            throw gpuqueue::CudaError(err, __FILE__, __LINE__);   \
        }                                                         \
    } while (0)

// ============================================================================
// Device Memory
// ============================================================================

/**
 * @brief RAII wrapper for device memory.
 * @tparam T Element type (for pointer arithmetic).
 */
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }
    
    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);  // Don't throw from destructor
        }
    }
    
    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Movable
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    size_t bytes() const noexcept { return size_ * sizeof(T); }
    
    // Copy from host (synchronous)
    void copy_from_host(const T* src, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    // Copy to host (synchronous)
    void copy_to_host(T* dst, size_t count) const {
        CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    // Async copy from host
    void copy_from_host_async(const T* src, size_t count, cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    
    // Async copy to host
    void copy_to_host_async(T* dst, size_t count, cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

private:
    T* ptr_;
    size_t size_;
};

// ============================================================================
// Pinned Host Memory
// ============================================================================

/**
 * @brief RAII wrapper for pinned (page-locked) host memory.
 * @tparam T Element type.
 * 
 * Pinned memory enables faster H2D/D2H transfers and is required for async copies.
 */
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit PinnedBuffer(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
        }
    }
    
    ~PinnedBuffer() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }
    
    // Non-copyable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    
    // Movable
    PinnedBuffer(PinnedBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    size_t bytes() const noexcept { return size_ * sizeof(T); }
    
    T& operator[](size_t idx) { return ptr_[idx]; }
    const T& operator[](size_t idx) const { return ptr_[idx]; }

private:
    T* ptr_;
    size_t size_;
};

// ============================================================================
// CUDA Stream Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for CUDA stream.
 */
class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    explicit CudaStream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    
    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Non-copyable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Movable
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
    bool query() const {
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        throw CudaError(err, __FILE__, __LINE__);
    }

private:
    cudaStream_t stream_;
};

// ============================================================================
// CUDA Event Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for CUDA event.
 */
class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }
    
    explicit CudaEvent(unsigned int flags) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }
    
    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Non-copyable
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // Movable
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    cudaEvent_t get() const noexcept { return event_; }
    operator cudaEvent_t() const noexcept { return event_; }
    
    void record(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
    
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    
    bool query() const {
        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        throw CudaError(err, __FILE__, __LINE__);
    }
    
    /**
     * @brief Get elapsed time between two events in milliseconds.
     */
    static float elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }

private:
    cudaEvent_t event_;
};

// ============================================================================
// Stream Pool
// ============================================================================

/**
 * @brief Pool of CUDA streams for multi-stream pipelining.
 */
class StreamPool {
public:
    explicit StreamPool(size_t count) {
        streams_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            streams_.emplace_back();
        }
    }
    
    size_t size() const noexcept { return streams_.size(); }
    
    CudaStream& operator[](size_t idx) { return streams_[idx]; }
    const CudaStream& operator[](size_t idx) const { return streams_[idx]; }
    
    /**
     * @brief Get stream in round-robin fashion.
     */
    CudaStream& next() {
        CudaStream& s = streams_[next_idx_];
        next_idx_ = (next_idx_ + 1) % streams_.size();
        return s;
    }
    
    /**
     * @brief Synchronize all streams in the pool.
     */
    void synchronize_all() {
        for (auto& s : streams_) {
            s.synchronize();
        }
    }

private:
    std::vector<CudaStream> streams_;
    size_t next_idx_ = 0;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Query device properties for device 0.
 */
inline cudaDeviceProp get_device_properties(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

/**
 * @brief Get compute capability as an integer (e.g., 89 for sm_89).
 */
inline int get_compute_capability(int device = 0) {
    cudaDeviceProp prop = get_device_properties(device);
    return prop.major * 10 + prop.minor;
}

/**
 * @brief Set the current CUDA device.
 */
inline void set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

/**
 * @brief Synchronize the current device.
 */
inline void device_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gpuqueue
