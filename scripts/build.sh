#!/bin/bash

# AI Trading System Build Script
# This script automates the build process for the entire system

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

# Build Python backend
build_python() {
    print_status "Building Python backend..."
    cd app

    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt

    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_success "Python backend built successfully!"
    else
        print_error "Python backend build failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Build Node.js frontend
build_node() {
    print_status "Building Node.js frontend..."
    cd frontend

    # Install Node.js dependencies
    print_status "Installing Node.js dependencies..."
    npm install

    # Build frontend
    print_status "Building frontend..."
    npm run build

    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_success "Node.js frontend built successfully!"
    else
        print_error "Node.js frontend build failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Build Docker images
build_docker() {
    print_status "Building Docker images..."
    check_docker

    # Build Docker images
    print_status "Building Docker images..."
    docker-compose build

    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_success "Docker images built successfully!"
    else
        print_error "Docker build failed. Check the output for details."
        exit 1
    fi
}

# Build for production
build_production() {
    print_status "Building for production..."
    check_python
    check_node

    # Build Python backend
    build_python

    # Build Node.js frontend
    build_node

    # Build Docker images
    build_docker

    print_success "Production build completed successfully!"
}

# Main build function
build() {
    print_status "Starting AI Trading System build process..."

    # Check prerequisites
    check_python
    check_node

    # Build Python backend
    build_python

    # Build Node.js frontend
    build_node

    # Build Docker images
    build_docker

    print_success "Build completed successfully!"
}

# Help function
show_help() {
    echo "AI Trading System Build Script"
    echo "Usage: $0 [all|python|node|docker|production]"
    echo ""
    echo "Options:"
    echo "  all         Build everything (default)"
    echo "  python      Build only Python backend"
    echo "  node        Build only Node.js frontend"
    echo "  docker      Build only Docker images"
    echo "  production  Build for production environment"
    echo ""
    echo "Examples:"
    echo "  $0 all       # Build everything"
    echo "  $0 production # Build for production"
}

# Main script logic
if [ $# -eq 0 ]; then
    build
    exit 0
fi

case $1 in
    "all")
        build
        ;;
    "python")
        check_python
        build_python
        ;;
    "node")
        check_node
        build_node
        ;;
    "docker")
        check_docker
        build_docker
        ;;
    "production")
        build_production
        ;;
    *)
        show_help
        exit 1
        ;;
esac