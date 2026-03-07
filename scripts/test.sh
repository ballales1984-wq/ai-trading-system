#!/bin/bash

# AI Trading System Test Script
# This script automates the testing process for the entire system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi
}

# Check if Node.js is available
check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js and try again."
        exit 1
    fi
}

# Run Python tests
run_python_tests() {
    print_status "Running Python tests..."
    cd tests

    # Run all Python tests
    python3 -m pytest -v --cov=app --cov-report=html:htmlcov --cov-report=term-missing

    # Check test results
    if [ $? -eq 0 ]; then
        print_success "All Python tests passed!"
    else
        print_error "Some Python tests failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Run Node.js tests
run_node_tests() {
    print_status "Running Node.js tests..."
    cd frontend

    # Install dependencies
    npm install

    # Run tests
    npm test

    # Check test results
    if [ $? -eq 0 ]; then
        print_success "All Node.js tests passed!"
    else
        print_error "Some Node.js tests failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    cd tests

    # Run integration tests
    python3 -m pytest -v test_integration.py

    # Check test results
    if [ $? -eq 0 ]; then
        print_success "All integration tests passed!"
    else
        print_error "Some integration tests failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    cd tests

    # Run performance tests
    python3 -m pytest -v test_performance.py

    # Check test results
    if [ $? -eq 0 ]; then
        print_success "All performance tests passed!"
    else
        print_error "Some performance tests failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Generate test coverage report
generate_coverage_report() {
    print_status "Generating test coverage report..."
    cd tests

    # Generate coverage report
    python3 -m pytest --cov=app --cov-report=html:htmlcov --cov-report=term-missing

    print_success "Test coverage report generated successfully!"
    print_status "View the report at: tests/htmlcov/index.html"

    cd ..
}

# Main test function
run_tests() {
    print_status "Starting AI Trading System test suite..."

    # Check prerequisites
    check_python
    check_node

    # Run tests
    run_python_tests
    run_node_tests
    run_integration_tests
    run_performance_tests
    generate_coverage_report

    print_success "All tests completed successfully!"
}

# Help function
show_help() {
    echo "AI Trading System Test Script"
    echo "Usage: $0 [all|python|node|integration|performance|coverage]"
    echo ""
    echo "Options:"
    echo "  all         Run all tests (default)"
    echo "  python      Run only Python tests"
    echo "  node        Run only Node.js tests"
    echo "  integration Run only integration tests"
    echo "  performance Run only performance tests"
    echo "  coverage    Generate test coverage report"
    echo ""
    echo "Examples:"
    echo "  $0 all       # Run all tests"
    echo "  $0 python    # Run only Python tests"
    echo "  $0 coverage  # Generate coverage report"
}

# Main script logic
if [ $# -eq 0 ]; then
    run_tests
    exit 0
fi

case $1 in
    "all")
        run_tests
        ;;
    "python")
        check_python
        run_python_tests
        ;;
    "node")
        check_node
        run_node_tests
        ;;
    "integration")
        run_integration_tests
        ;;
    "performance")
        run_performance_tests
        ;;
    "coverage")
        generate_coverage_report
        ;;
    *)
        show_help
        exit 1
        ;;
esac