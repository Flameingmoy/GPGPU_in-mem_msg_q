#include "gpuqueue/queue.hpp"
#include <atomic>

namespace gpuqueue {

namespace {
std::atomic<bool> g_inited{false};
}

int init(int /*device*/) {
  g_inited.store(true, std::memory_order_release);
  return 0;
}

void shutdown() {
  g_inited.store(false, std::memory_order_release);
}

const char* version() {
  return "0.1.0a0";
}

} // namespace gpuqueue
