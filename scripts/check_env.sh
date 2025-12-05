#!/bin/bash
# GPUQueue Environment Verification Script
# Run this to verify your system is ready for development.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  GPUQueue Environment Check"
echo "=========================================="
echo ""

# Track overall status
ALL_PASSED=true

# 1. Check NVIDIA driver
echo -n "Checking NVIDIA driver... "
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$DRIVER_VERSION" ]; then
        # Extract major version
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
        if [ "$DRIVER_MAJOR" -ge 535 ]; then
            echo -e "${GREEN}OK${NC} (v$DRIVER_VERSION)"
        else
            echo -e "${YELLOW}WARNING${NC} (v$DRIVER_VERSION, recommend ≥535)"
        fi
    else
        echo -e "${RED}FAILED${NC} (nvidia-smi found but no driver version)"
        ALL_PASSED=false
    fi
else
    echo -e "${RED}FAILED${NC} (nvidia-smi not found)"
    ALL_PASSED=false
fi

# 2. Check CUDA Toolkit (nvcc)
echo -n "Checking CUDA Toolkit (nvcc)... "
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    NVCC_MAJOR=$(echo "$NVCC_VERSION" | cut -d. -f1)
    NVCC_MINOR=$(echo "$NVCC_VERSION" | cut -d. -f2)
    if [ "$NVCC_MAJOR" -ge 12 ] && [ "$NVCC_MINOR" -ge 6 ]; then
        echo -e "${GREEN}OK${NC} (v$NVCC_VERSION)"
    elif [ "$NVCC_MAJOR" -ge 13 ]; then
        echo -e "${GREEN}OK${NC} (v$NVCC_VERSION)"
    else
        echo -e "${YELLOW}WARNING${NC} (v$NVCC_VERSION, recommend ≥12.6)"
    fi
else
    echo -e "${RED}FAILED${NC} (nvcc not found - install CUDA Toolkit)"
    echo "  Hint: Add /usr/local/cuda/bin to PATH"
    ALL_PASSED=false
fi

# 3. Check GPU and Compute Capability
echo -n "Checking GPU device... "
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        echo -e "${GREEN}OK${NC} ($GPU_NAME)"
    else
        echo -e "${RED}FAILED${NC} (no GPU detected)"
        ALL_PASSED=false
    fi
else
    echo -e "${RED}SKIPPED${NC} (nvidia-smi not available)"
fi

# 4. Check CUDA memory
echo -n "Checking GPU memory... "
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_MEM" ]; then
        echo -e "${GREEN}OK${NC} ($GPU_MEM)"
    else
        echo -e "${YELLOW}WARNING${NC} (could not query memory)"
    fi
else
    echo -e "${YELLOW}SKIPPED${NC}"
fi

# 5. Check CMake
echo -n "Checking CMake... "
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | sed 's/cmake version //')
    CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -ge 3 ] && [ "$CMAKE_MINOR" -ge 24 ]; then
        echo -e "${GREEN}OK${NC} (v$CMAKE_VERSION)"
    else
        echo -e "${YELLOW}WARNING${NC} (v$CMAKE_VERSION, recommend ≥3.24)"
    fi
else
    echo -e "${RED}FAILED${NC} (cmake not found)"
    ALL_PASSED=false
fi

# 6. Check Python
echo -n "Checking Python... "
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | sed 's/Python //')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "${GREEN}OK${NC} (v$PYTHON_VERSION)"
    else
        echo -e "${YELLOW}WARNING${NC} (v$PYTHON_VERSION, recommend ≥3.10)"
    fi
else
    echo -e "${RED}FAILED${NC} (python not found)"
    ALL_PASSED=false
fi

# 7. Check pybind11
echo -n "Checking pybind11... "
if python -c "import pybind11" 2>/dev/null; then
    PYBIND_VERSION=$(python -c "import pybind11; print(pybind11.__version__)" 2>/dev/null)
    echo -e "${GREEN}OK${NC} (v$PYBIND_VERSION)"
else
    echo -e "${YELLOW}WARNING${NC} (not installed - will be fetched by CMake)"
fi

# 8. Check GCC
echo -n "Checking GCC... "
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -1 | sed 's/.*) //')
    echo -e "${GREEN}OK${NC} (v$GCC_VERSION)"
else
    echo -e "${RED}FAILED${NC} (gcc not found)"
    ALL_PASSED=false
fi

echo ""
echo "=========================================="
if [ "$ALL_PASSED" = true ]; then
    echo -e "  ${GREEN}All checks passed!${NC}"
    echo "  Ready to build GPUQueue."
else
    echo -e "  ${RED}Some checks failed.${NC}"
    echo "  Please fix the issues above before building."
fi
echo "=========================================="
echo ""

# Exit with appropriate code
if [ "$ALL_PASSED" = true ]; then
    exit 0
else
    exit 1
fi
