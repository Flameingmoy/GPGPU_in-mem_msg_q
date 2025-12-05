/**
 * @file test_device_query.cu
 * @brief CUDA device query and validation test.
 * 
 * This test verifies:
 * 1. CUDA runtime is accessible
 * 2. GPU device is available
 * 3. Compute capability meets requirements (sm_89 for RTX 4070 Ti Super)
 * 4. Basic kernel execution works
 * 5. H2D/D2H memory transfers work
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Macro to check CUDA errors
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Simple vector add kernel to verify execution
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Test device properties
bool test_device_properties() {
    printf("\n=== Device Properties Test ===\n");
    
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable devices found\n");
        return false;
    }
    
    printf("Found %d CUDA device(s)\n\n", device_count);
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  SM Count: %d\n", prop.multiProcessorCount);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock: %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("\n");
        
        // Check minimum compute capability for this project (sm_89 preferred)
        int cc = prop.major * 10 + prop.minor;
        if (cc >= 89) {
            printf("  ✓ Compute Capability %d.%d meets sm_89 requirement\n", prop.major, prop.minor);
        } else if (cc >= 70) {
            printf("  ⚠ Compute Capability %d.%d is below sm_89 but may work\n", prop.major, prop.minor);
        } else {
            printf("  ✗ Compute Capability %d.%d is too low (need ≥7.0)\n", prop.major, prop.minor);
            return false;
        }
    }
    
    return true;
}

// Test basic kernel execution
bool test_kernel_execution() {
    printf("\n=== Kernel Execution Test ===\n");
    
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "ERROR: Failed to allocate host memory\n");
        return false;
    }
    
    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    printf("  Allocated device memory: 3 x %zu bytes\n", bytes);
    
    // Copy to device (H2D)
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    printf("  H2D transfer complete\n");
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Kernel executed: %d blocks x %d threads\n", blocks, threads_per_block);
    
    // Copy back (D2H)
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("  D2H transfer complete\n");
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            fprintf(stderr, "  ERROR: Mismatch at index %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("  ✓ Results verified: all %d elements correct\n", N);
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    
    return correct;
}

// Test pinned memory and async transfers
bool test_pinned_memory() {
    printf("\n=== Pinned Memory Test ===\n");
    
    const int N = 1024 * 1024;  // 1M elements
    const size_t bytes = N * sizeof(float);
    
    // Allocate pinned host memory
    float* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    printf("  Allocated pinned memory: %.2f MB\n", bytes / (1024.0 * 1024.0));
    
    // Initialize
    for (int i = 0; i < N; ++i) {
        h_pinned[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Create stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Async H2D
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream));
    
    // Async D2H (back to same buffer for verification)
    float* h_verify;
    CUDA_CHECK(cudaMallocHost(&h_verify, bytes));
    CUDA_CHECK(cudaMemcpyAsync(h_verify, d_data, bytes, cudaMemcpyDeviceToHost, stream));
    
    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("  Async H2D → D2H complete\n");
    
    // Verify roundtrip
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_pinned[i] != h_verify[i]) {
            fprintf(stderr, "  ERROR: Roundtrip mismatch at index %d\n", i);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("  ✓ Pinned memory roundtrip verified\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFreeHost(h_verify));
    
    return correct;
}

// Test CUDA events for timing
bool test_cuda_events() {
    printf("\n=== CUDA Events Test ===\n");
    
    const int N = 1024 * 1024 * 4;  // 4M elements
    const size_t bytes = N * sizeof(float);
    
    float *h_data, *d_data;
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Initialize
    memset(h_data, 0, bytes);
    
    // Create events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Time H2D transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float h2d_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
    
    float h2d_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / (h2d_ms / 1000.0);
    printf("  H2D Transfer: %.2f ms (%.2f GB/s)\n", h2d_ms, h2d_bandwidth);
    
    // Time D2H transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
    
    float d2h_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / (d2h_ms / 1000.0);
    printf("  D2H Transfer: %.2f ms (%.2f GB/s)\n", d2h_ms, d2h_bandwidth);
    
    printf("  ✓ CUDA events working correctly\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));
    
    return true;
}

int main() {
    printf("GPUQueue CUDA Device Query & Validation\n");
    printf("========================================\n");
    
    bool all_passed = true;
    
    all_passed &= test_device_properties();
    all_passed &= test_kernel_execution();
    all_passed &= test_pinned_memory();
    all_passed &= test_cuda_events();
    
    printf("\n========================================\n");
    if (all_passed) {
        printf("All tests PASSED ✓\n");
        printf("GPU is ready for GPUQueue development.\n");
        return EXIT_SUCCESS;
    } else {
        printf("Some tests FAILED ✗\n");
        return EXIT_FAILURE;
    }
}
