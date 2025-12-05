cmake_minimum_required(VERSION 3.24)

# Set CUDA architectures BEFORE project() for proper initialization
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{GPUQUEUE_CUDA_ARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{GPUQUEUE_CUDA_ARCHS})
  else()
    # Default: Ada Lovelace (RTX 4070 Ti Super = sm_89)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
endif()

project(gpuqueue LANGUAGES CXX CUDA)

# C++ standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard settings (required for libcu++ atomics)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Debug flags for CUDA (cuda-gdb support)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo")

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Core library
add_library(gpuqueue_core STATIC
  src/cpp/queue_core.cpp
  src/cuda/queue_kernels.cu
)

set_target_properties(gpuqueue_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  # Note: CUDA_SEPARABLE_COMPILATION has runtime overhead.
  # Only enable if device-side linking is needed (e.g., calling device functions across TUs).
  # CUDA_SEPARABLE_COMPILATION ON
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

# =============================================================================
# Testing
# =============================================================================
option(GPUQUEUE_BUILD_TESTS "Build tests" ON)

if(GPUQUEUE_BUILD_TESTS)
  enable_testing()
  
  # Fetch Google Test
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  # Prevent gtest from overriding compiler/linker settings on Windows
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  
  include(GoogleTest)
  
  # -------------------------------------------------------------------------
  # Standalone CUDA test (no gtest, just verify GPU access)
  # -------------------------------------------------------------------------
  add_executable(test_device_query
    tests/cuda/test_device_query.cu
  )
  target_include_directories(test_device_query PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(test_device_query PRIVATE CUDA::cudart)
  set_target_properties(test_device_query PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
  add_test(NAME DeviceQuery COMMAND test_device_query)
  set_tests_properties(DeviceQuery PROPERTIES LABELS "cuda")
  
  # -------------------------------------------------------------------------
  # Queue Integration Test
  # -------------------------------------------------------------------------
  add_executable(test_queue_integration
    tests/cuda/test_queue_integration.cu
  )
  target_include_directories(test_queue_integration PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(test_queue_integration PRIVATE gpuqueue_core CUDA::cudart)
  set_target_properties(test_queue_integration PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
  add_test(NAME QueueIntegration COMMAND test_queue_integration)
  set_tests_properties(QueueIntegration PROPERTIES LABELS "cuda;integration")
  
  # -------------------------------------------------------------------------
  # C++ Unit Tests with Google Test
  # -------------------------------------------------------------------------
  add_executable(test_ring_math
    tests/cpp/test_ring_math.cpp
  )
  target_include_directories(test_ring_math PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(test_ring_math PRIVATE GTest::gtest_main)
  gtest_discover_tests(test_ring_math PROPERTIES LABELS "unit")
  
endif()
