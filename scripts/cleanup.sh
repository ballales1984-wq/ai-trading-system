#!/bin/bash

# AI Trading System Cleanup Script
# This script automates the cleanup process for the entire system

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

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Clean Python environment
clean_python() {
    print_status "Cleaning Python environment..."
    cd app

    # Remove Python cache
    print_status "Removing Python cache..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

    # Remove virtual environment
    print_status "Removing virtual environment..."
    if [ -d "venv" ]; then
        rm -rf venv
    fi

    # Remove pip cache
    print_status "Removing pip cache..."
    pip cache purge

    print_success "Python environment cleaned successfully!"
    cd ..
}

# Clean Node.js environment
clean_node() {
    print_status "Cleaning Node.js environment..."
    cd frontend

    # Remove node_modules
    print_status "Removing node_modules..."
    if [ -d "node_modules" ]; then
        rm -rf node_modules
    fi

    # Remove npm cache
    print_status "Removing npm cache..."
    npm cache clean --force

    # Remove build artifacts
    print_status "Removing build artifacts..."
    if [ -d "dist" ]; then
        rm -rf dist
    fi
    if [ -d "build" ]; then
        rm -rf build
    fi

    print_success "Node.js environment cleaned successfully!"
    cd ..
}

# Clean Docker environment
clean_docker() {
    print_status "Cleaning Docker environment..."
    check_docker

    # Stop all containers
    print_status "Stopping all containers..."
    docker-compose down

    # Remove all containers
    print_status "Removing all containers..."
    docker container prune -f

    # Remove all images
    print_status "Removing all images..."
    docker image prune -a -f

    # Remove all volumes
    print_status "Removing all volumes..."
    docker volume prune -f

    # Remove all networks
    print_status "Removing all networks..."
    docker network prune -f

    print_success "Docker environment cleaned successfully!"
}

# Clean test artifacts
clean_tests() {
    print_status "Cleaning test artifacts..."
    cd tests

    # Remove test cache
    print_status "Removing test cache..."
    if [ -d ".pytest_cache" ]; then
        rm -rf .pytest_cache
    fi

    # Remove coverage reports
    print_status "Removing coverage reports..."
    if [ -d "htmlcov" ]; then
        rm -rf htmlcov
    fi

    # Remove test logs
    print_status "Removing test logs..."
    if [ -f "test.log" ]; then
        rm -f test.log
    fi

    print_success "Test artifacts cleaned successfully!"
    cd ..
}

# Clean documentation
clean_docs() {
    print_status "Cleaning documentation..."
    cd docs

    # Remove built documentation
    print_status "Removing built documentation..."
    if [ -d "site" ]; then
        rm -rf site
    fi

    # Remove API documentation
    print_status "Removing API documentation..."
    if [ -d "api" ]; then
        rm -rf api
    fi

    # Remove frontend documentation
    print_status "Removing frontend documentation..."
    if [ -d "frontend" ]; then
        rm -rf frontend
    fi

    print_success "Documentation cleaned successfully!"
    cd ..
}

# Clean logs
clean_logs() {
    print_status "Cleaning logs..."
    cd logs

    # Remove all log files
    print_status "Removing all log files..."
    if [ -n "$(ls -A . 2>/dev/null)" ]; then
        rm -f *
    fi

    print_success "Logs cleaned successfully!"
    cd ..
}

# Clean temporary files
clean_temp() {
    print_status "Cleaning temporary files..."
    cd ..

    # Remove temporary files
    print_status "Removing temporary files..."
    if [ -f "temp.html" ]; then
        rm -f temp.html
    fi
    if [ -f "temp_orders.json" ]; then
        rm -f temp_orders.json
    fi

    print_success "Temporary files cleaned successfully!"
}

# Clean IDE artifacts
clean_ide() {
    print_status "Cleaning IDE artifacts..."
    cd ..

    # Remove IDE files
    print_status "Removing IDE files..."
    if [ -f ".vscode/settings.json" ]; then
        rm -f .vscode/settings.json
    fi
    if [ -f ".vscode/launch.json" ]; then
        rm -f .vscode/launch.json
    fi
    if [ -f ".vscode/tasks.json" ]; then
        rm -f .vscode/tasks.json
    fi

    print_success "IDE artifacts cleaned successfully!"
}

# Clean everything
clean_all() {
    print_status "Starting full cleanup..."
    check_python
    check_node
    check_docker

    # Clean Python environment
    clean_python

    # Clean Node.js environment
    clean_node

    # Clean Docker environment
    clean_docker

    # Clean test artifacts
    clean_tests

    # Clean documentation
    clean_docs

    # Clean logs
    clean_logs

    # Clean temporary files
    clean_temp

    # Clean IDE artifacts
    clean_ide

    print_success "Full cleanup completed successfully!"
}

# Main cleanup function
cleanup() {
    print_status "Starting AI Trading System cleanup process..."

    # Check prerequisites
    check_python
    check_node
    check_docker

    # Clean Python environment
    clean_python

    # Clean Node.js environment
    clean_node

    # Clean Docker environment
    clean_docker

    # Clean test artifacts
    clean_tests

    # Clean documentation
    clean_docs

    # Clean logs
    clean_logs

    # Clean temporary files
    clean_temp

    # Clean IDE artifacts
    clean_ide

    print_success "Cleanup completed successfully!"
}

# Help function
show_help() {
    echo "AI Trading System Cleanup Script"
    echo "Usage: $0 [all|python|node|docker|tests|docs|logs|temp|ide]"
    echo ""
    echo "Options:"
    echo "  all         Clean everything (default)"
    echo "  python      Clean only Python environment"
    echo "  node        Clean only Node.js environment"
    echo "  docker      Clean only Docker environment"
    echo "  tests       Clean only test artifacts"
    echo "  docs        Clean only documentation"
    echo "  logs        Clean only logs"
    echo "  temp        Clean only temporary files"
    echo "  ide         Clean only IDE artifacts"
    echo ""
    echo "Examples:"
    echo "  $0 all       # Clean everything"
    echo "  $0 python    # Clean only Python environment"
}

# Main script logic
if [ $# -eq 0 ]; then
    cleanup
    exit 0
fi

case $1 in
    "all")
        clean_all
        ;;
    "python")
        check_python
        clean_python
        ;;
    "node")
        check_node
        clean_node
        ;;
    "docker")
        check_docker
        clean_docker
        ;;
    "tests")
        clean_tests
        ;;
    "docs")
        clean_docs
        ;;
    "logs")
        clean_logs
        ;;
    "temp")
        clean_temp
        ;;
    "ide")
        clean_ide
        ;;
    *)
        show_help
        exit 1
        ;;
esac