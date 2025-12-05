/**
 * @file queue_kernels.cu
 * @brief CUDA kernels for the GPU-resident message queue.
 * 
 * This file contains:
 * - Persistent consumer kernel that polls for READY slots
 * - Process message device function (user-customizable)
 * - Atomic state transition helpers
 */

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Device-visible types (must match host types)
// ============================================================================

namespace gpuqueue {
namespace device {

/// Slot states (must match SlotState enum in types.hpp)
enum SlotState : uint32_t {
    EMPTY = 0,
    READY = 1,
    INFLIGHT = 2,
    DONE = 3
};

/// Slot header (must match SlotHeader in types.hpp)
struct alignas(16) SlotHeader {
    uint32_t len;
    uint32_t flags;
    uint64_t msg_id;
};

/// Device-visible control block fields
struct DeviceControlBlock {
    uint64_t head;           // Offset 0
    char _pad1[56];          // Pad to cache line
    uint64_t tail;           // Offset 64
    char _pad2[56];
    uint32_t stop_flag;      // Offset 128
    char _pad3[60];
    uint64_t processed_count; // Offset 192
    uint64_t error_count;
};

} // namespace device
} // namespace gpuqueue

using namespace gpuqueue::device;

// ============================================================================
// Process Message (User-Customizable)
// ============================================================================

/**
 * @brief Process a single message (default implementation).
 * 
 * This function is called by the persistent kernel for each READY slot.
 * The default implementation simply copies the payload to the result area.
 * 
 * Users can replace this with custom processing logic.
 * 
 * @param payload Input payload data
 * @param len Input payload length
 * @param result Output buffer for result (same size as payload)
 * @param result_len Output: actual result length
 * @return 0 on success, non-zero on error
 */
__device__ int process_message(
    const uint8_t* payload,
    uint32_t len,
    uint8_t* result,
    uint32_t* result_len
) {
    // Default: simple echo (copy input to output)
    for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        result[i] = payload[i];
    }
    
    // Set result length (only thread 0)
    if (threadIdx.x == 0) {
        *result_len = len;
    }
    
    return 0;  // Success
}

// ============================================================================
// Persistent Consumer Kernel
// ============================================================================

/**
 * @brief Persistent kernel that continuously processes messages.
 * 
 * This kernel runs until stop_flag is set. It:
 * 1. Scans for READY slots
 * 2. Claims slot with atomicCAS(READY → INFLIGHT)
 * 3. Calls process_message()
 * 4. Marks slot as DONE
 * 5. Backs off with __nanosleep() when idle
 * 
 * @param slot_data Device buffer containing slot headers + payloads
 * @param slot_states Device array of slot states (uint32_t)
 * @param control Control block (pinned host memory, device-accessible)
 * @param capacity Number of slots
 * @param slot_size Size of each slot (header + payload)
 */
__global__ void persistent_consumer_kernel(
    uint8_t* __restrict__ slot_data,
    volatile uint32_t* __restrict__ slot_states,  // volatile to prevent caching
    volatile DeviceControlBlock* __restrict__ control,
    uint32_t capacity,
    uint32_t slot_size
) {
    // Simple single-thread approach for MVP
    // Only thread 0 does any work
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Main processing loop
    while (true) {
        // Check stop flag (volatile read)
        if (control->stop_flag != 0) {
            break;
        }
        
        // Scan ALL slots sequentially
        for (uint32_t i = 0; i < capacity; i++) {
            // Read current state (volatile ensures fresh read from global memory)
            uint32_t current = slot_states[i];
            
            if (current == READY) {
                // Try to claim this slot atomically
                uint32_t old_state = atomicCAS((uint32_t*)&slot_states[i], READY, INFLIGHT);
                
                if (old_state == READY) {
                    // Successfully claimed this slot
                    // Memory fence to ensure we see the payload written by host
                    __threadfence_system();
                    
                    // (Payload is echo'd in-place, nothing to process for MVP)
                    
                    // Memory fence before marking DONE
                    __threadfence();
                    
                    // Mark slot as DONE
                    atomicExch((uint32_t*)&slot_states[i], DONE);
                    
                    // Update processed count
                    atomicAdd((unsigned long long*)&control->processed_count, 1ULL);
                    
                    // Memory fence to ensure DONE is visible to host
                    __threadfence_system();
                }
            }
        }
        
        // Brief backoff
        #if __CUDA_ARCH__ >= 700
        __nanosleep(100);  // 100 nanoseconds
        #endif
    }
}

// ============================================================================
// Kernel Launch Wrapper (called from host)
// ============================================================================

extern "C" {

/**
 * @brief Launch the persistent consumer kernel.
 */
cudaError_t launch_persistent_consumer(
    uint8_t* slot_data,
    uint32_t* slot_states,
    void* control,
    uint32_t capacity,
    uint32_t slot_size,
    uint32_t num_blocks,
    uint32_t num_threads,
    cudaStream_t stream
) {
    persistent_consumer_kernel<<<num_blocks, num_threads, 0, stream>>>(
        slot_data,
        slot_states,
        reinterpret_cast<volatile DeviceControlBlock*>(control),
        capacity,
        slot_size
    );
    return cudaGetLastError();
}

/**
 * @brief Legacy noop kernel for build verification.
 */
__global__ void gpuqueue_noop_kernel() {
    // Intentionally empty
}

} // extern "C"
