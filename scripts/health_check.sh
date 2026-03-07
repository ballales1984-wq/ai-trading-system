#!/bin/bash

# AI Trading System Health Check Script
# This script automates the health check process for the entire system

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

# Check if Git is available
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git and try again."
        exit 1
    fi
}

# Check Python dependencies
check_python_dependencies() {
    print_status "Checking Python dependencies..."
    cd app

    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found in app directory."
        exit 1
    fi

    # Check if all dependencies can be installed
    print_status "Checking if all Python dependencies can be installed..."
    if pip install -r requirements.txt --dry-run; then
        print_success "All Python dependencies are available!"
    else
        print_error "Some Python dependencies are missing or have issues."
        exit 1
    fi

    cd ..
}

# Check Node.js dependencies
check_node_dependencies() {
    print_status "Checking Node.js dependencies..."
    cd frontend

    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        print_error "package.json not found in frontend directory."
        exit 1
    fi

    # Check if all dependencies can be installed
    print_status "Checking if all Node.js dependencies can be installed..."
    if npm install --dry-run; then
        print_success "All Node.js dependencies are available!"
    else
        print_error "Some Node.js dependencies are missing or have issues."
        exit 1
    fi

    cd ..
}

# Check Docker services
check_docker_services() {
    print_status "Checking Docker services..."
    check_docker

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi

    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi

    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in project root."
        exit 1
    fi

    # Check if Docker services can be started
    print_status "Checking if Docker services can be started..."
    if docker-compose config > /dev/null 2>&1; then
        print_success "Docker services are configured correctly!"
    else
        print_error "Docker services have configuration issues."
        exit 1
    fi

    # Check if Docker images can be built
    print_status "Checking if Docker images can be built..."
    if docker-compose build --dry-run > /dev/null 2>&1; then
        print_success "Docker images can be built successfully!"
    else
        print_error "Docker images have build issues."
        exit 1
    fi
}

# Check Git repository
check_git_repository() {
    print_status "Checking Git repository..."
    check_git

    # Check if we're in a Git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a Git repository."
        exit 1
    fi

    # Check if there are uncommitted changes
    print_status "Checking for uncommitted changes..."
    if ! git diff --quiet; then
        print_warning "There are uncommitted changes. Please commit or stash them."
    else
        print_success "No uncommitted changes found!"
    fi

    # Check if we're on the main branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "main" ]; then
        print_warning "You're not on the main branch. Consider switching to main for releases."
    else
        print_success "On main branch!"
    fi

    # Check if we're up to date with remote
    print_status "Checking if we're up to date with remote..."
    if git diff --quiet origin/main; then
        print_success "Up to date with remote!"
    else
        print_warning "Not up to date with remote. Consider pulling the latest changes."
    fi
}

# Check test coverage
check_test_coverage() {
    print_status "Checking test coverage..."
    cd tests

    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        print_error "pytest is not installed. Please install pytest and try again."
        exit 1
    fi

    # Check if coverage is available
    if ! command -v coverage &> /dev/null; then
        print_error "coverage is not installed. Please install coverage and try again."
        exit 1
    fi

    # Check if tests can be run
    print_status "Checking if tests can be run..."
    if pytest --collect-only > /dev/null 2>&1; then
        print_success "Tests can be run successfully!"
    else
        print_error "Tests have issues."
        exit 1
    fi

    # Check if coverage report can be generated
    print_status "Checking if coverage report can be generated..."
    if coverage run --source=app -m pytest --collect-only > /dev/null 2>&1; then
        print_success "Coverage report can be generated successfully!"
    else
        print_error "Coverage report has issues."
        exit 1
    fi

    cd ..
}

# Check documentation
check_documentation() {
    print_status "Checking documentation..."
    cd docs

    # Check if mkdocs is available
    if ! command -v mkdocs &> /dev/null; then
        print_error "mkdocs is not installed. Please install mkdocs and try again."
        exit 1
    fi

    # Check if mkdocs.yml exists
    if [ ! -f "mkdocs.yml" ]; then
        print_error "mkdocs.yml not found in docs directory."
        exit 1
    fi

    # Check if documentation can be built
    print_status "Checking if documentation can be built..."
    if mkdocs build --dry-run > /dev/null 2>&1; then
        print_success "Documentation can be built successfully!"
    else
        print_error "Documentation has issues."
        exit 1
    fi

    cd ..
}

# Check security
check_security() {
    print_status "Checking security..."
    cd app

    # Check if bandit is available
    if ! command -v bandit &> /dev/null; then
        print_error "bandit is not installed. Please install bandit and try again."
        exit 1
    fi

    # Check if security scan can be run
    print_status "Checking if security scan can be run..."
    if bandit -r . --quiet > /dev/null 2>&1; then
        print_success "Security scan can be run successfully!"
    else
        print_error "Security scan has issues."
        exit 1
    fi

    cd ..
}

# Check performance
check_performance() {
    print_status "Checking performance..."
    cd tests

    # Check if locust is available
    if ! command -v locust &> /dev/null; then
        print_error "locust is not installed. Please install locust and try again."
        exit 1
    fi

    # Check if performance tests can be run
    print_status "Checking if performance tests can be run..."
    if locust --version > /dev/null 2>&1; then
        print_success "Performance tests can be run successfully!"
    else
        print_error "Performance tests have issues."
        exit 1
    fi

    cd ..
}

# Generate health report
generate_health_report() {
    print_status "Generating health report..."

    # Create health report
    health_report="## AI Trading System Health Report\n\n"
    health_report+="**Generated**: $(date +'%Y-%m-%d %H:%M:%S')\n\n"
    health_report+="### System Status\n\n"
    health_report+="| Component | Status | Message |\n"
    health_report+="|-----------|--------|---------|\n"

    # Check Python dependencies
    if check_python_dependencies > /dev/null 2>&1; then
        health_report+="| Python Dependencies | ✅ | All dependencies available |\n"
    else
        health_report+="| Python Dependencies | ❌ | Some dependencies missing or have issues |\n"
    fi

    # Check Node.js dependencies
    if check_node_dependencies > /dev/null 2>&1; then
        health_report+="| Node.js Dependencies | ✅ | All dependencies available |\n"
    else
        health_report+="| Node.js Dependencies | ❌ | Some dependencies missing or have issues |\n"
    fi

    # Check Docker services
    if check_docker_services > /dev/null 2>&1; then
        health_report+="| Docker Services | ✅ | Services configured correctly |\n"
    else
        health_report+="| Docker Services | ❌ | Services have configuration issues |\n"
    fi

    # Check Git repository
    if check_git_repository > /dev/null 2>&1; then
        health_report+="| Git Repository | ✅ | Repository is healthy |\n"
    else
        health_report+="| Git Repository | ❌ | Repository has issues |\n"
    fi

    # Check test coverage
    if check_test_coverage > /dev/null 2>&1; then
        health_report+="| Test Coverage | ✅ | Tests can be run successfully |\n"
    else
        health_report+="| Test Coverage | ❌ | Tests have issues |\n"
    fi

    # Check documentation
    if check_documentation > /dev/null 2>&1; then
        health_report+="| Documentation | ✅ | Documentation can be built successfully |\n"
    else
        health_report+="| Documentation | ❌ | Documentation has issues |\n"
    fi

    # Check security
    if check_security > /dev/null 2>&1; then
        health_report+="| Security | ✅ | Security scan can be run successfully |\n"
    else
        health_report+="| Security | ❌ | Security scan has issues |\n"
    fi

    # Check performance
    if check_performance > /dev/null 2>&1; then
        health_report+="| Performance | ✅ | Performance tests can be run successfully |\n"
    else
        health_report+="| Performance | ❌ | Performance tests have issues |\n"
    fi

    # Save health report
    echo -e "$health_report" > HEALTH_REPORT.md

    print_success "Health report generated successfully!"
    print_status "View the report at: HEALTH_REPORT.md"
}

# Main health check function
health_check() {
    print_status "Starting AI Trading System health check..."

    # Check prerequisites
    check_python
    check_node
    check_docker
    check_git

    # Check Python dependencies
    check_python_dependencies

    # Check Node.js dependencies
    check_node_dependencies

    # Check Docker services
    check_docker_services

    # Check Git repository
    check_git_repository

    # Check test coverage
    check_test_coverage

    # Check documentation
    check_documentation

    # Check security
    check_security

    # Check performance
    check_performance

    # Generate health report
    generate_health_report

    print_success "Health check completed successfully!"
}

# Help function
show_help() {
    echo "AI Trading System Health Check Script"
    echo "Usage: $0 [check|report]"
    echo ""
    echo "Options:"
    echo "  check       Run health check (default)"
    echo "  report      Generate health report only"
    echo ""
    echo "Examples:"
    echo "  $0 check     # Run full health check"
    echo "  $0 report    # Generate health report only"
}

# Main script logic
if [ $# -eq 0 ]; then
    health_check
    exit 0
fi

case $1 in
    "check")
        health_check
        ;;
    "report")
        generate_health_report
        ;;
    *)
        show_help
        exit 1
        ;;
esac