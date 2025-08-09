#pragma once

namespace gpuqueue {

// Initialize the GPU queue core. For scaffold, this is a no-op that returns 0.
// A non-zero return can be reserved for future error codes.
int init(int device = 0);

// Shutdown and cleanup resources (scaffold: no-op).
void shutdown();

// Return the core version string.
const char* version();

} // namespace gpuqueue
