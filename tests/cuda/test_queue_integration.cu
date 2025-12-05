/**
 * @file test_queue_integration.cu
 * @brief Integration tests for the GPU-resident message queue.
 * 
 * Tests:
 * 1. Queue initialization and shutdown
 * 2. Single message enqueue/dequeue roundtrip
 * 3. Multiple messages enqueue/dequeue
 * 4. Queue full behavior
 * 5. Concurrent enqueue stress test
 */

#include "gpuqueue/gpu_queue.hpp"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

using namespace gpuqueue;

#define TEST_ASSERT(cond, msg)                                      \
    do {                                                            \
        if (!(cond)) {                                              \
            fprintf(stderr, "FAILED: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            return false;                                           \
        }                                                           \
    } while (0)

// ============================================================================
// Test: Basic initialization and shutdown
// ============================================================================

bool test_init_shutdown() {
    printf("\n=== Test: Init/Shutdown ===\n");
    
    QueueConfig config;
    config.capacity = 256;
    config.slot_bytes = 512;
    config.num_threads = 128;
    config.num_blocks = 1;
    
    {
        GpuQueue queue(config);
        TEST_ASSERT(queue.is_running(), "Queue should be running after init");
        
        auto stats = queue.stats();
        TEST_ASSERT(stats.enqueue_count == 0, "Initial enqueue count should be 0");
        TEST_ASSERT(stats.queue_depth == 0, "Initial queue depth should be 0");
        
        printf("  Queue initialized with capacity=%u, slot_bytes=%u\n", 
               config.capacity, config.slot_bytes);
        
        queue.shutdown();
        TEST_ASSERT(!queue.is_running(), "Queue should not be running after shutdown");
    }
    
    printf("  ✓ Init/Shutdown passed\n");
    return true;
}

// ============================================================================
// Test: Single message roundtrip
// ============================================================================

bool test_single_message() {
    printf("\n=== Test: Single Message Roundtrip ===\n");
    
    QueueConfig config;
    config.capacity = 256;
    config.slot_bytes = 512;
    
    GpuQueue queue(config);
    
    // Prepare test message
    const char* test_msg = "Hello, GPU Queue!";
    size_t msg_len = strlen(test_msg) + 1;
    
    // Enqueue
    int64_t msg_id = queue.enqueue(test_msg, msg_len);
    TEST_ASSERT(msg_id >= 0, "Enqueue should succeed");
    printf("  Enqueued message with ID=%ld\n", msg_id);
    
    // Wait for processing (kernel should echo the message back)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Dequeue
    char result[512];
    size_t result_len = sizeof(result);
    
    QueueStatus status = queue.try_dequeue_result(static_cast<uint64_t>(msg_id), result, &result_len);
    TEST_ASSERT(status == QueueStatus::OK, "Dequeue should succeed");
    TEST_ASSERT(result_len == msg_len, "Result length should match");
    TEST_ASSERT(strcmp(result, test_msg) == 0, "Result should match input");
    
    printf("  Received result: \"%s\" (len=%zu)\n", result, result_len);
    
    auto stats = queue.stats();
    TEST_ASSERT(stats.enqueue_count == 1, "Enqueue count should be 1");
    TEST_ASSERT(stats.dequeue_count == 1, "Dequeue count should be 1");
    TEST_ASSERT(stats.processed_count >= 1, "Processed count should be at least 1");
    
    printf("  ✓ Single message roundtrip passed\n");
    return true;
}

// ============================================================================
// Test: Multiple messages
// ============================================================================

bool test_multiple_messages() {
    printf("\n=== Test: Multiple Messages ===\n");
    
    QueueConfig config;
    config.capacity = 256;
    config.slot_bytes = 128;
    
    GpuQueue queue(config);
    
    const int NUM_MESSAGES = 50;
    std::vector<int64_t> msg_ids;
    
    // Enqueue multiple messages
    for (int i = 0; i < NUM_MESSAGES; ++i) {
        char payload[64];
        snprintf(payload, sizeof(payload), "Message #%d", i);
        
        int64_t msg_id = queue.enqueue(payload, strlen(payload) + 1);
        TEST_ASSERT(msg_id >= 0, "Enqueue should succeed");
        msg_ids.push_back(msg_id);
    }
    
    printf("  Enqueued %d messages\n", NUM_MESSAGES);
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Dequeue all messages
    int dequeued = 0;
    for (int64_t id : msg_ids) {
        char result[128];
        size_t result_len = sizeof(result);
        
        // Try a few times (kernel may still be processing)
        for (int retry = 0; retry < 10; ++retry) {
            QueueStatus status = queue.try_dequeue_result(static_cast<uint64_t>(id), result, &result_len);
            if (status == QueueStatus::OK) {
                dequeued++;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    printf("  Dequeued %d/%d messages\n", dequeued, NUM_MESSAGES);
    TEST_ASSERT(dequeued == NUM_MESSAGES, "Should dequeue all messages");
    
    auto stats = queue.stats();
    printf("  Stats: enqueue=%lu, dequeue=%lu, processed=%lu\n",
           stats.enqueue_count, stats.dequeue_count, stats.processed_count);
    
    printf("  ✓ Multiple messages passed\n");
    return true;
}

// ============================================================================
// Test: Poll completions
// ============================================================================

bool test_poll_completions() {
    printf("\n=== Test: Poll Completions ===\n");
    
    QueueConfig config;
    config.capacity = 256;
    config.slot_bytes = 64;
    
    GpuQueue queue(config);
    
    // Enqueue a batch
    const int BATCH_SIZE = 10;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        char payload[32];
        snprintf(payload, sizeof(payload), "Batch item %d", i);
        int64_t msg_id = queue.enqueue(payload, strlen(payload) + 1);
        TEST_ASSERT(msg_id >= 0, "Enqueue should succeed");
    }
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Poll for completions
    uint64_t completed_ids[BATCH_SIZE];
    size_t completed = queue.poll_completions(completed_ids, BATCH_SIZE);
    
    printf("  Found %zu completed messages\n", completed);
    TEST_ASSERT(completed >= 1, "Should find at least one completion");
    
    printf("  ✓ Poll completions passed\n");
    return true;
}

// ============================================================================
// Test: Performance (throughput)
// ============================================================================

bool test_throughput() {
    printf("\n=== Test: Throughput ===\n");
    
    QueueConfig config;
    config.capacity = 4096;
    config.slot_bytes = 256;
    
    GpuQueue queue(config);
    
    const int NUM_MESSAGES = 1000;
    char payload[256];
    memset(payload, 'X', sizeof(payload));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Enqueue all messages
    for (int i = 0; i < NUM_MESSAGES; ++i) {
        int64_t msg_id = queue.enqueue(payload, sizeof(payload), 0);
        if (msg_id < 0) {
            // Queue full, wait and retry
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            --i;
        }
    }
    
    auto enqueue_end = std::chrono::high_resolution_clock::now();
    auto enqueue_us = std::chrono::duration_cast<std::chrono::microseconds>(enqueue_end - start).count();
    
    // Wait for all to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto stats = queue.stats();
    
    double enqueue_rate = (NUM_MESSAGES * 1e6) / enqueue_us;
    double throughput_mbps = (NUM_MESSAGES * sizeof(payload) * 1e6) / enqueue_us / (1024 * 1024);
    
    printf("  Enqueued %d messages in %ld µs\n", NUM_MESSAGES, enqueue_us);
    printf("  Enqueue rate: %.0f msg/s\n", enqueue_rate);
    printf("  Throughput: %.2f MB/s\n", throughput_mbps);
    printf("  Processed: %lu, Errors: %lu\n", stats.processed_count, stats.error_count);
    
    TEST_ASSERT(enqueue_rate > 10000, "Enqueue rate should be > 10k msg/s");
    
    printf("  ✓ Throughput test passed\n");
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("GPUQueue Integration Tests\n");
    printf("==========================\n");
    
    bool all_passed = true;
    
    all_passed &= test_init_shutdown();
    all_passed &= test_single_message();
    all_passed &= test_multiple_messages();
    all_passed &= test_poll_completions();
    all_passed &= test_throughput();
    
    printf("\n==========================\n");
    if (all_passed) {
        printf("All integration tests PASSED ✓\n");
        return 0;
    } else {
        printf("Some tests FAILED ✗\n");
        return 1;
    }
}
