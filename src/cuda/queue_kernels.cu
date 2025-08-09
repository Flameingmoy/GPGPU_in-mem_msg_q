#include <cuda_runtime.h>

extern "C" __global__ void gpuqueue_noop_kernel() {
  // Intentionally empty: scaffold kernel to verify CUDA build works
}
