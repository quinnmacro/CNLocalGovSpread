#!/usr/bin/env bash
# ============================================================================
# CNLocalGovSpread v3.0 - CI-friendly test runner
# ============================================================================
# Usage:
#   ./run_tests.sh           # Run all tests (default)
#   ./run_tests.sh quick     # Skip slow tests
#   ./run_tests.sh coverage  # Run with coverage report
#   ./run_tests.sh integration # Run only integration tests
#   ./run_tests.sh dashboard   # Run only dashboard tests
#   ./run_tests.sh smoke     # Quick smoke test (first 5 test files)
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Resolve Python command (prefer python3 for CI environments)
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: python/python3 not found${NC}"
    exit 1
fi

# Resolve pip command
PIP_CMD=""
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Install test dependencies if needed
install_deps() {
    echo -e "${CYAN}Installing test dependencies...${NC}"
    $PIP_CMD install pytest openpyxl -q 2>/dev/null
}

# Run tests with specified markers and options
run_tests() {
    local marker_expr="$1"
    local extra_args="$2"

    echo -e "${CYAN}Running CNLocalGovSpread v3.0 tests...${NC}"
    echo -e "${YELLOW}Marker filter: ${marker_expr:-none}${NC}"

    $PYTHON_CMD -m pytest tests/ \
        --tb=short \
        -v \
        ${marker_expr:+-m "$marker_expr"} \
        $extra_args

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Tests failed with exit code ${exit_code}${NC}"
    fi
    return $exit_code
}

# Parse command
MODE="${1:-all}"

case "$MODE" in
    all)
        install_deps
        run_tests "" ""
        ;;
    quick)
        install_deps
        run_tests "not slow" ""
        ;;
    coverage)
        install_deps
        $PIP_CMD install pytest-cov -q 2>/dev/null
        run_tests "" "--cov=src --cov-report=term-missing --cov-report=html"
        ;;
    integration)
        install_deps
        run_tests "integration" ""
        ;;
    dashboard)
        install_deps
        run_tests "dashboard" ""
        ;;
    smoke)
        install_deps
        echo -e "${CYAN}Running smoke tests (quick subset)...${NC}"
        $PYTHON_CMD -m pytest tests/test_content.py tests/test_styles.py \
            tests/test_export.py tests/test_all.py \
            --tb=short -v -x
        ;;
    count)
        install_deps
        echo -e "${CYAN}Counting tests...${NC}"
        $PYTHON_CMD -m pytest tests/ --collect-only -q 2>&1 | tail -3
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 {all|quick|coverage|integration|dashboard|smoke|count}"
        exit 1
        ;;
esac