cmake_minimum_required(VERSION 3.24)
project(gpuqueue LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default CUDA arch: Ada (sm_89). Allow override via env or cache.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{GPUQUEUE_CUDA_ARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{GPUQUEUE_CUDA_ARCHS})
  else()
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Core library
add_library(gpuqueue_core STATIC
  src/cpp/queue_core.cpp
  src/cuda/queue_kernels.cu
)

set_target_properties(gpuqueue_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gpuqueue_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(gpuqueue_core PUBLIC CUDA::cudart)

# Python extension module
pybind11_add_module(_core
  src/cpp/bindings.cpp
)

target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(_core PRIVATE gpuqueue_core CUDA::cudart)

set_target_properties(_core PROPERTIES
  OUTPUT_NAME "_core"
)

# Install the extension into the Python package directory inside the wheel
install(TARGETS _core
  LIBRARY DESTINATION gpuqueue
  RUNTIME DESTINATION gpuqueue
  ARCHIVE DESTINATION gpuqueue
)
