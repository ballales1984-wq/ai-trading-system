#!/bin/bash

# AI Trading System Release Script
# This script automates the release process for the entire system

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

# Check if Git is available
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git and try again."
        exit 1
    fi
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

# Run pre-release checks
run_pre_release_checks() {
    print_status "Running pre-release checks..."

    # Check Git status
    print_status "Checking Git status..."
    if ! git diff --quiet; then
        print_warning "There are uncommitted changes. Please commit or stash them before releasing."
        exit 1
    fi

    # Check if we're on the main branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "main" ]; then
        print_warning "You're not on the main branch. Releases should be made from the main branch."
        exit 1
    fi

    # Check if we're up to date with remote
    print_status "Checking if we're up to date with remote..."
    if ! git diff --quiet origin/main; then
        print_warning "Your local repository is not up to date with the remote. Please pull the latest changes."
        exit 1
    fi

    print_success "Pre-release checks completed successfully!"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    cd tests

    # Run all tests
    python3 -m pytest -v --cov=app --cov-report=html:htmlcov --cov-report=term-missing

    # Check test results
    if [ $? -eq 0 ]; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed. Please fix them before releasing."
        exit 1
    fi

    cd ..
}

# Build the project
build_project() {
    print_status "Building the project..."
    cd scripts

    # Build everything
    ./build.sh all

    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_success "Build completed successfully!"
    else
        print_error "Build failed. Please fix the issues before releasing."
        exit 1
    fi

    cd ..
}

# Update version
update_version() {
    print_status "Updating version..."

    # Get current version
    current_version=$(grep -oP '(?<=version": ")[^"]+' package.json)

    # Ask for new version
    read -p "Enter new version (current is $current_version): " new_version

    # Update version in package.json
    sed -i "s/\"version\": \"$current_version\"/\"version\": \"$new_version\"/" package.json

    # Update version in other files
    sed -i "s/version = '$current_version'/version = '$new_version'/" app/__init__.py
    sed -i "s/version: '$current_version'/version: '$new_version'/" frontend/vite.config.ts

    print_success "Version updated to $new_version!"
}

# Generate release notes
generate_release_notes() {
    print_status "Generating release notes..."

    # Get commit history
    commit_history=$(git log --oneline --since="1 month ago")

    # Generate release notes
    release_notes="## Release Notes\n\n"
    release_notes+="**Version**: $new_version\n"
    release_notes+="**Date**: $(date +'%Y-%m-%d')\n\n"
    release_notes+="### New Features\n\n"
    release_notes+="### Bug Fixes\n\n"
    release_notes+="### Improvements\n\n"
    release_notes+="### Breaking Changes\n\n"
    release_notes+="### Commits\n\n"
    release_notes+="```\n"
    release_notes+="$commit_history\n"
    release_notes+="```\n"

    # Save release notes
    echo -e "$release_notes" > RELEASE_NOTES.md

    print_success "Release notes generated successfully!"
}

# Tag the release
tag_release() {
    print_status "Tagging the release..."
    git tag -a "v$new_version" -m "Release v$new_version"

    # Push tags
    git push origin "v$new_version"

    print_success "Release tagged successfully!"
}

# Build Docker image
build_docker_image() {
    print_status "Building Docker image..."
    check_docker

    # Build Docker image
    docker build -t ai-trading-system:$new_version .

    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully!"
    else
        print_error "Docker build failed. Check the output for details."
        exit 1
    fi
}

# Push Docker image
push_docker_image() {
    print_status "Pushing Docker image..."
    check_docker

    # Login to Docker Hub
    echo "Logging in to Docker Hub..."
    docker login

    # Push image
    docker push ai-trading-system:$new_version

    print_success "Docker image pushed successfully!"
}

# Update documentation
update_documentation() {
    print_status "Updating documentation..."
    cd scripts

    # Generate documentation
    ./docs.sh all

    # Check if generation was successful
    if [ $? -eq 0 ]; then
        print_success "Documentation updated successfully!"
    else
        print_error "Documentation update failed. Check the output for details."
        exit 1
    fi

    cd ..
}

# Main release function
release() {
    print_status "Starting AI Trading System release process..."

    # Check prerequisites
    check_git
    check_python
    check_node
    check_docker

    # Run pre-release checks
    run_pre_release_checks

    # Run tests
    run_tests

    # Build the project
    build_project

    # Update version
    update_version

    # Generate release notes
    generate_release_notes

    # Tag the release
    tag_release

    # Build Docker image
    build_docker_image

    # Push Docker image
    push_docker_image

    # Update documentation
    update_documentation

    print_success "Release completed successfully!"
    print_status "Released version $new_version!"
}

# Help function
show_help() {
    echo "AI Trading System Release Script"
    echo "Usage: $0 [release|version|notes|tag|docker|docs]"
    echo ""
    echo "Options:"
    echo "  release     Run full release process (default)"
    echo "  version     Update version number"
    echo "  notes       Generate release notes"
    echo "  tag         Tag the release"
    echo "  docker      Build and push Docker image"
    echo "  docs        Update documentation"
    echo ""
    echo "Examples:"
    echo "  $0 release   # Run full release process"
    echo "  $0 version   # Update version number"
}

# Main script logic
if [ $# -eq 0 ]; then
    release
    exit 0
fi

case $1 in
    "release")
        release
        ;;
    "version")
        update_version
        ;;
    "notes")
        generate_release_notes
        ;;
    "tag")
        tag_release
        ;;
    "docker")
        build_docker_image
        push_docker_image
        ;;
    "docs")
        update_documentation
        ;;
    *)
        show_help
        exit 1
        ;;
esac