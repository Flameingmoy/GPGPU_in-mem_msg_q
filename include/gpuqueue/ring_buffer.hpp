#pragma once
#include <cstddef>

namespace gpuqueue {

// Placeholder configuration for a future device ring buffer.
struct RingBufferConfig {
  int slot_count{0};
  std::size_t slot_size{0};
};

} // namespace gpuqueue
