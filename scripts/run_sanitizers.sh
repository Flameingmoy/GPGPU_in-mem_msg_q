#!/bin/bash
# Run CUDA compute-sanitizer checks on GPU queue tests
#
# Usage: ./scripts/run_sanitizers.sh [test_executable]
#
# If no executable is specified, runs on test_queue_integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

# Default test executable
TEST_EXEC="${1:-${BUILD_DIR}/test_queue_integration}"

if [[ ! -f "$TEST_EXEC" ]]; then
    echo "Error: Test executable not found: $TEST_EXEC"
    echo "Build the project first: cmake --build build"
    exit 1
fi

echo "========================================"
echo "CUDA Sanitizer Verification Suite"
echo "========================================"
echo "Test executable: $TEST_EXEC"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

run_sanitizer() {
    local tool=$1
    local desc=$2
    
    echo "----------------------------------------"
    echo "Running: compute-sanitizer --tool $tool"
    echo "Purpose: $desc"
    echo "----------------------------------------"
    
    if compute-sanitizer --tool "$tool" "$TEST_EXEC" 2>&1 | tee /tmp/sanitizer_$tool.log | tail -5; then
        if grep -q "0 errors\|0 hazards" /tmp/sanitizer_$tool.log; then
            echo "✓ $tool PASSED"
            ((PASS_COUNT++))
        else
            echo "✗ $tool FAILED - check /tmp/sanitizer_$tool.log"
            ((FAIL_COUNT++))
        fi
    else
        echo "✗ $tool FAILED to run"
        ((FAIL_COUNT++))
    fi
    echo ""
}

# Run all sanitizers
run_sanitizer "memcheck" "Detect memory access errors (out-of-bounds, use-after-free)"
run_sanitizer "racecheck" "Detect shared memory race conditions"
run_sanitizer "initcheck" "Detect uninitialized device memory access"
run_sanitizer "synccheck" "Detect synchronization errors"

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo ""
    echo "✓ All sanitizer checks PASSED"
    exit 0
else
    echo ""
    echo "✗ Some sanitizer checks FAILED"
    exit 1
fi
