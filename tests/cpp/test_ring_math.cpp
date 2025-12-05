/**
 * @file test_ring_math.cpp
 * @brief Unit tests for ring buffer index math and utilities.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

namespace {

// Ring buffer index math functions (will move to header later)

/**
 * @brief Compute slot index from absolute position using bitwise AND.
 * Requires capacity to be a power of two.
 */
constexpr uint32_t slot_index(uint64_t pos, uint32_t capacity) {
    return static_cast<uint32_t>(pos & (capacity - 1));
}

/**
 * @brief Check if capacity is a power of two.
 */
constexpr bool is_power_of_two(uint32_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Compute queue depth (head - tail) with wrap-around safety.
 */
constexpr uint64_t queue_depth(uint64_t head, uint64_t tail) {
    return head - tail;  // Works correctly due to unsigned wrap-around
}

/**
 * @brief Check if queue is full.
 */
constexpr bool is_full(uint64_t head, uint64_t tail, uint32_t capacity) {
    return queue_depth(head, tail) >= capacity;
}

/**
 * @brief Check if queue is empty.
 */
constexpr bool is_empty(uint64_t head, uint64_t tail) {
    return head == tail;
}

} // anonymous namespace


// =============================================================================
// Power of Two Tests
// =============================================================================

TEST(RingMath, IsPowerOfTwo) {
    // Valid powers of two
    EXPECT_TRUE(is_power_of_two(1));
    EXPECT_TRUE(is_power_of_two(2));
    EXPECT_TRUE(is_power_of_two(4));
    EXPECT_TRUE(is_power_of_two(8));
    EXPECT_TRUE(is_power_of_two(16));
    EXPECT_TRUE(is_power_of_two(256));
    EXPECT_TRUE(is_power_of_two(1024));
    EXPECT_TRUE(is_power_of_two(4096));
    EXPECT_TRUE(is_power_of_two(65536));
    EXPECT_TRUE(is_power_of_two(1 << 20));  // 1MB slots
    
    // Not powers of two
    EXPECT_FALSE(is_power_of_two(0));
    EXPECT_FALSE(is_power_of_two(3));
    EXPECT_FALSE(is_power_of_two(5));
    EXPECT_FALSE(is_power_of_two(6));
    EXPECT_FALSE(is_power_of_two(7));
    EXPECT_FALSE(is_power_of_two(100));
    EXPECT_FALSE(is_power_of_two(1000));
    EXPECT_FALSE(is_power_of_two(1023));
    EXPECT_FALSE(is_power_of_two(1025));
}


// =============================================================================
// Slot Index Tests
// =============================================================================

TEST(RingMath, SlotIndexBasic) {
    const uint32_t capacity = 256;
    
    // Basic indexing
    EXPECT_EQ(slot_index(0, capacity), 0u);
    EXPECT_EQ(slot_index(1, capacity), 1u);
    EXPECT_EQ(slot_index(255, capacity), 255u);
    
    // Wrap-around
    EXPECT_EQ(slot_index(256, capacity), 0u);
    EXPECT_EQ(slot_index(257, capacity), 1u);
    EXPECT_EQ(slot_index(512, capacity), 0u);
    EXPECT_EQ(slot_index(1000, capacity), 1000 % 256);
}

TEST(RingMath, SlotIndexVariousCapacities) {
    // Test multiple power-of-two capacities
    for (uint32_t cap : {256u, 1024u, 4096u, 65536u}) {
        ASSERT_TRUE(is_power_of_two(cap));
        
        for (uint64_t i = 0; i < cap * 3; ++i) {
            EXPECT_EQ(slot_index(i, cap), i % cap)
                << "Failed at i=" << i << ", cap=" << cap;
        }
    }
}

TEST(RingMath, SlotIndexLargePositions) {
    const uint32_t capacity = 4096;
    
    // Test with large positions near UINT64_MAX
    uint64_t large_pos = std::numeric_limits<uint64_t>::max() - 100;
    
    for (int i = 0; i < 200; ++i) {
        uint64_t pos = large_pos + i;
        uint32_t expected = static_cast<uint32_t>(pos % capacity);
        EXPECT_EQ(slot_index(pos, capacity), expected)
            << "Failed at pos=" << pos;
    }
}


// =============================================================================
// Queue Depth Tests
// =============================================================================

TEST(RingMath, QueueDepthBasic) {
    // Normal cases
    EXPECT_EQ(queue_depth(0, 0), 0u);
    EXPECT_EQ(queue_depth(10, 0), 10u);
    EXPECT_EQ(queue_depth(100, 50), 50u);
    EXPECT_EQ(queue_depth(1000, 1000), 0u);
}

TEST(RingMath, QueueDepthWrapAround) {
    // Test wrap-around behavior with 64-bit integers
    // When head wraps around past UINT64_MAX, subtraction still works
    
    uint64_t max_val = std::numeric_limits<uint64_t>::max();
    
    // Near wrap-around
    EXPECT_EQ(queue_depth(max_val, max_val - 100), 100u);
    EXPECT_EQ(queue_depth(max_val, max_val), 0u);
    
    // After wrap-around: head=5, tail=max-5 means depth=11 items enqueued
    // (5 after wrap + 6 before wrap including max itself)
    uint64_t head_wrapped = 5;
    uint64_t tail_before_wrap = max_val - 5;
    uint64_t expected_depth = head_wrapped - tail_before_wrap;  // Wraps to 11
    EXPECT_EQ(queue_depth(head_wrapped, tail_before_wrap), expected_depth);
}


// =============================================================================
// Full/Empty Detection Tests
// =============================================================================

TEST(RingMath, IsEmpty) {
    EXPECT_TRUE(is_empty(0, 0));
    EXPECT_TRUE(is_empty(100, 100));
    EXPECT_TRUE(is_empty(1000000, 1000000));
    
    EXPECT_FALSE(is_empty(1, 0));
    EXPECT_FALSE(is_empty(100, 99));
}

TEST(RingMath, IsFull) {
    const uint32_t capacity = 256;
    
    // Empty queue is not full
    EXPECT_FALSE(is_full(0, 0, capacity));
    
    // Partially filled
    EXPECT_FALSE(is_full(100, 0, capacity));
    EXPECT_FALSE(is_full(255, 0, capacity));
    
    // Full queue
    EXPECT_TRUE(is_full(256, 0, capacity));
    EXPECT_TRUE(is_full(1256, 1000, capacity));
    
    // Overfull (should not happen in practice, but test the math)
    EXPECT_TRUE(is_full(300, 0, capacity));
}

TEST(RingMath, FullEmptyEdgeCases) {
    const uint32_t capacity = 1024;
    
    // Large head/tail values that are equal (empty)
    uint64_t large = std::numeric_limits<uint64_t>::max() - 1000;
    EXPECT_TRUE(is_empty(large, large));
    EXPECT_FALSE(is_full(large, large, capacity));
    
    // Large head, tail behind by capacity (full)
    EXPECT_TRUE(is_full(large + capacity, large, capacity));
    EXPECT_FALSE(is_empty(large + capacity, large));
}


// =============================================================================
// Invariant Tests
// =============================================================================

TEST(RingMath, InvariantDepthBounds) {
    // Invariant: 0 <= depth <= capacity at all times
    const uint32_t capacity = 4096;
    
    // Simulate a sequence of enqueue/dequeue operations
    uint64_t head = 0;
    uint64_t tail = 0;
    
    for (int i = 0; i < 10000; ++i) {
        // Randomly enqueue or dequeue
        if (i % 3 != 0 && queue_depth(head, tail) < capacity) {
            ++head;  // Enqueue
        } else if (!is_empty(head, tail)) {
            ++tail;  // Dequeue
        }
        
        // Check invariant
        uint64_t depth = queue_depth(head, tail);
        EXPECT_GE(depth, 0u);
        EXPECT_LE(depth, capacity)
            << "Invariant violated: depth=" << depth << ", head=" << head << ", tail=" << tail;
    }
}
