#!/bin/bash

# AI Trading System Documentation Script
# This script automates the documentation generation process

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

# Generate API documentation
generate_api_docs() {
    print_status "Generating API documentation..."
    cd app

    # Install pydoc-markdown if not installed
    if ! command -v pydoc-markdown &> /dev/null; then
        print_status "Installing pydoc-markdown..."
        pip install pydoc-markdown
    fi

    # Generate API documentation
    print_status "Generating API documentation..."
    pydoc-markdown --input app --output docs/api --format markdown

    # Check if generation was successful
    if [ $? -eq 0 ]; then
        print_success "API documentation generated successfully!"
    else
        print_error "API documentation generation failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Generate frontend documentation
generate_frontend_docs() {
    print_status "Generating frontend documentation..."
    cd frontend

    # Install documentation tools
    print_status "Installing documentation tools..."
    npm install -g jsdoc

    # Generate frontend documentation
    print_status "Generating frontend documentation..."
    jsdoc -c jsdoc.json -d docs/frontend

    # Check if generation was successful
    if [ $? -eq 0 ]; then
        print_success "Frontend documentation generated successfully!"
    else
        print_error "Frontend documentation generation failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Generate user documentation
generate_user_docs() {
    print_status "Generating user documentation..."
    cd docs

    # Install documentation tools
    print_status "Installing documentation tools..."
    pip install mkdocs-material

    # Build user documentation
    print_status "Building user documentation..."
    mkdocs build

    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_success "User documentation generated successfully!"
        print_status "View the documentation at: docs/site/index.html"
    else
        print_error "User documentation generation failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Generate all documentation
generate_all_docs() {
    print_status "Generating all documentation..."
    check_python
    check_node

    # Generate API documentation
    generate_api_docs

    # Generate frontend documentation
    generate_frontend_docs

    # Generate user documentation
    generate_user_docs

    print_success "All documentation generated successfully!"
}

# Main documentation function
generate_docs() {
    print_status "Starting AI Trading System documentation generation..."

    # Check prerequisites
    check_python
    check_node

    # Generate API documentation
    generate_api_docs

    # Generate frontend documentation
    generate_frontend_docs

    # Generate user documentation
    generate_user_docs

    print_success "Documentation generation completed successfully!"
}

# Help function
show_help() {
    echo "AI Trading System Documentation Script"
    echo "Usage: $0 [all|api|frontend|user|all]"
    echo ""
    echo "Options:"
    echo "  all         Generate all documentation (default)"
    echo "  api         Generate only API documentation"
    echo "  frontend    Generate only frontend documentation"
    echo "  user        Generate only user documentation"
    echo ""
    echo "Examples:"
    echo "  $0 all       # Generate all documentation"
    echo "  $0 api       # Generate only API documentation"
}

# Main script logic
if [ $# -eq 0 ]; then
    generate_docs
    exit 0
fi

case $1 in
    "all")
        generate_all_docs
        ;;
    "api")
        check_python
        generate_api_docs
        ;;
    "frontend")
        check_node
        generate_frontend_docs
        ;;
    "user")
        check_python
        generate_user_docs
        ;;
    *)
        show_help
        exit 1
        ;;
esac