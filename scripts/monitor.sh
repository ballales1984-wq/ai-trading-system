#!/bin/bash

# AI Trading System Monitoring Script
# This script automates the monitoring process for the entire system

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

# Check if curl is available
check_curl() {
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed. Please install curl and try again."
        exit 1
    fi
}

# Monitor system resources
monitor_resources() {
    print_status "Monitoring system resources..."
    check_python
    check_node
    check_docker
    check_curl

    # Create monitoring report
    monitoring_report="## AI Trading System Monitoring Report\n\n"
    monitoring_report+="**Generated**: $(date +'%Y-%m-%d %H:%M:%S')\n\n"
    monitoring_report+="### System Resources\n\n"
    monitoring_report+="| Resource | Status | Value |\n"
    monitoring_report+="|----------|--------|-------|\n"

    # Check CPU usage
    print_status "Checking CPU usage..."
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | \
                sed "s/.*, *\([0-9.]*\)%* id.*/\\1/" | \
                awk '{print 100 - $1"%"}')
    monitoring_report+="| CPU Usage | $(if [ $(echo "$cpu_usage < 80" | bc) -eq 1 ]; then echo "✅"; else echo "⚠️"; fi) | $cpu_usage |\n"

    # Check memory usage
    print_status "Checking memory usage..."
    memory_usage=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
    monitoring_report+="| Memory Usage | $(if [ $(echo "$memory_usage < 80" | bc) -eq 1 ]; then echo "✅"; else echo "⚠️"; fi) | $memory_usage |\n"

    # Check disk usage
    print_status "Checking disk usage..."
    disk_usage=$(df / | awk 'NR==2 {printf "%.1f%%", $5}')
    monitoring_report+="| Disk Usage | $(if [ $(echo "$disk_usage < 90" | bc) -eq 1 ]; then echo "✅"; else echo "⚠️"; fi) | $disk_usage |\n"

    # Check Python processes
    print_status "Checking Python processes..."
    python_processes=$(ps aux | grep python | grep -v grep | wc -l)
    monitoring_report+="| Python Processes | $(if [ $python_processes -lt 10 ]; then echo "✅"; else echo "⚠️"; fi) | $python_processes |\n"

    # Check Node.js processes
    print_status "Checking Node.js processes..."
    node_processes=$(ps aux | grep node | grep -v grep | wc -l)
    monitoring_report+="| Node.js Processes | $(if [ $node_processes -lt 10 ]; then echo "✅"; else echo "⚠️"; fi) | $node_processes |\n"

    # Check Docker containers
    print_status "Checking Docker containers..."
    if docker info > /dev/null 2>&1; then
        docker_containers=$(docker ps -q | wc -l)
        monitoring_report+="| Docker Containers | $(if [ $docker_containers -lt 20 ]; then echo "✅"; else echo "⚠️"; fi) | $docker_containers |\n"
    else
        monitoring_report+="| Docker Containers | ❌ | Docker not running |\n"
    fi

    # Check network connectivity
    print_status "Checking network connectivity..."
    if curl -s --head --request GET http://google.com | grep "200 OK" > /dev/null; then
        monitoring_report+="| Network Connectivity | ✅ | Connected |\n"
    else
        monitoring_report+="| Network Connectivity | ❌ | Not connected |\n"
    fi

    # Check API endpoints
    print_status "Checking API endpoints..."
    if curl -s --head --request GET http://localhost:8000/health | grep "200 OK" > /dev/null; then
        monitoring_report+="| API Health | ✅ | API is healthy |\n"
    else
        monitoring_report+="| API Health | ❌ | API is not healthy |\n"
    fi

    # Check frontend
    print_status "Checking frontend..."
    if curl -s --head --request GET http://localhost:5173 | grep "200 OK" > /dev/null; then
        monitoring_report+="| Frontend | ✅ | Frontend is running |\n"
    else
        monitoring_report+="| Frontend | ❌ | Frontend is not running |\n"
    fi

    # Check database
    print_status "Checking database..."
    if [ -f "app/core/config.py" ]; then
        database_url=$(grep -oP '(?<=DATABASE_URL = ")[^"]+' app/core/config.py)
        if [ -n "$database_url" ]; then
            if [ "${database_url:0:9}" = "postgresql" ]; then
                # Check PostgreSQL database
                if psql "$database_url" -c '\q' > /dev/null 2>&1; then
                    monitoring_report+="| Database | ✅ | Database is accessible |\n"
                else
                    monitoring_report+="| Database | ❌ | Database is not accessible |\n"
                fi
            elif [ "${database_url:0:6}" = "sqlite:" ]; then
                # Check SQLite database
                database_file=$(echo "$database_url" | cut -d/ -f3)
                if [ -f "$database_file" ]; then
                    monitoring_report+="| Database | ✅ | Database file exists |\n"
                else
                    monitoring_report+="| Database | ❌ | Database file not found |\n"
                fi
            else
                monitoring_report+="| Database | ⚠️ | Unknown database type |\n"
            fi
        else
            monitoring_report+="| Database | ⚠️ | Database URL not found |\n"
        fi
    else
        monitoring_report+="| Database | ⚠️ | Database configuration not found |\n"
    fi

    # Check logs
    print_status "Checking logs..."
    if [ -d "logs" ]; then
        log_files=$(find logs -name "*.log" -type f | wc -l)
        monitoring_report+="| Log Files | $(if [ $log_files -lt 100 ]; then echo "✅"; else echo "⚠️"; fi) | $log_files files |\n"
    else
        monitoring_report+="| Log Files | ⚠️ | Logs directory not found |\n"
    fi

    # Check for errors in logs
    print_status "Checking for errors in logs..."
    if [ -d "logs" ]; then
        error_count=$(grep -r "ERROR" logs/ 2>/dev/null | wc -l)
        monitoring_report+="| Error Count | $(if [ $error_count -eq 0 ]; then echo "✅"; else echo "⚠️"; fi) | $error_count errors |\n"
    else
        monitoring_report+="| Error Count | ⚠️ | Logs directory not found |\n"
    fi

    # Check for warnings in logs
    print_status "Checking for warnings in logs..."
    if [ -d "logs" ]; then
        warning_count=$(grep -r "WARNING" logs/ 2>/dev/null | wc -l)
        monitoring_report+="| Warning Count | $(if [ $warning_count -lt 100 ]; then echo "✅"; else echo "⚠️"; fi) | $warning_count warnings |\n"
    else
        monitoring_report+="| Warning Count | ⚠️ | Logs directory not found |\n"
    fi

    # Check recent activity
    print_status "Checking recent activity..."
    if [ -d "logs" ]; then
        recent_activity=$(find logs -name "*.log" -type f -mtime -1 | wc -l)
        monitoring_report+="| Recent Activity | $(if [ $recent_activity -gt 0 ]; then echo "✅"; else echo "⚠️"; fi) | $recent_activity files modified in last 24h |\n"
    else
        monitoring_report+="| Recent Activity | ⚠️ | Logs directory not found |\n"
    fi

    # Save monitoring report
    echo -e "$monitoring_report" > MONITORING_REPORT.md

    print_success "Monitoring report generated successfully!"
    print_status "View the report at: MONITORING_REPORT.md"
}

# Monitor specific component
monitor_component() {
    print_status "Monitoring specific component: $1"

    case $1 in
        "system")
            monitor_resources
            ;;
        "python")
            check_python_dependencies
            ;;
        "node")
            check_node_dependencies
            ;;
        "docker")
            check_docker_services
            ;;
        "git")
            check_git_repository
            ;;
        "tests")
            check_test_coverage
            ;;
        "docs")
            check_documentation
            ;;
        "security")
            check_security
            ;;
        "performance")
            check_performance
            ;;
        *)
            print_error "Invalid component: $1"
            echo "Valid components: system, python, node, docker, git, tests, docs, security, performance"
            exit 1
            ;;
    esac
}

# Main monitoring function
monitor() {
    print_status "Starting AI Trading System monitoring process..."

    # Check prerequisites
    check_python
    check_node
    check_docker
    check_curl

    # Monitor system resources
    monitor_resources

    print_success "Monitoring completed successfully!"
}

# Help function
show_help() {
    echo "AI Trading System Monitoring Script"
    echo "Usage: $0 [all|system|python|node|docker|git|tests|docs|security|performance]"
    echo ""
    echo "Options:"
    echo "  all         Monitor everything (default)"
    echo "  system      Monitor system resources"
    echo "  python      Check Python dependencies"
    echo "  node        Check Node.js dependencies"
    echo "  docker      Check Docker services"
    echo "  git         Check Git repository"
    echo "  tests       Check test coverage"
    echo "  docs        Check documentation"
    echo "  security    Check security"
    echo "  performance Check performance"
    echo ""
    echo "Examples:"
    echo "  $0 all       # Monitor everything"
    echo "  $0 system    # Monitor system resources"
}

# Main script logic
if [ $# -eq 0 ]; then
    monitor
    exit 0
fi

case $1 in
    "all")
        monitor
        ;;
    "system")
        monitor_component "system"
        ;;
    "python")
        monitor_component "python"
        ;;
    "node")
        monitor_component "node"
        ;;
    "docker")
        monitor_component "docker"
        ;;
    "git")
        monitor_component "git"
        ;;
    "tests")
        monitor_component "tests"
        ;;
    "docs")
        monitor_component "docs"
        ;;
    "security")
        monitor_component "security"
        ;;
    "performance")
        monitor_component "performance"
        ;;
    *)
        show_help
        exit 1
        ;;
esac