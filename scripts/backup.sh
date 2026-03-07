#!/bin/bash

# AI Trading System Backup Script
# This script automates the backup process for the entire system

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

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if tar is available
check_tar() {
    if ! command -v tar &> /dev/null; then
        print_error "tar is not installed. Please install tar and try again."
        exit 1
    fi
}

# Backup Git repository
backup_git() {
    print_status "Backing up Git repository..."
    check_git

    # Check if we're in a Git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a Git repository."
        exit 1
    fi

    # Create backup directory
    backup_dir="backups/git"
    mkdir -p "$backup_dir"

    # Create backup file name
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$backup_dir/ai_trading_system_git_backup_$timestamp.tar.gz"

    # Create backup
    print_status "Creating Git repository backup..."
    git bundle create "$backup_file" --all

    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_success "Git repository backup completed successfully!"
        print_status "Backup saved to: $backup_file"
    else
        print_error "Git repository backup failed."
        exit 1
    fi
}

# Backup Python environment
backup_python() {
    print_status "Backing up Python environment..."
    check_python

    # Create backup directory
    backup_dir="backups/python"
    mkdir -p "$backup_dir"

    # Create backup file name
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$backup_dir/ai_trading_system_python_backup_$timestamp.tar.gz"

    # Create backup
    print_status "Creating Python environment backup..."
    cd app

    # Backup requirements.txt
    if [ -f "requirements.txt" ]; then
        cp requirements.txt "$backup_dir/"
    fi

    # Backup virtual environment if it exists
    if [ -d "venv" ]; then
        tar -czf "$backup_file" venv
    fi

    cd ..

    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_success "Python environment backup completed successfully!"
        print_status "Backup saved to: $backup_file"
    else
        print_error "Python environment backup failed."
        exit 1
    fi
}

# Backup Node.js environment
backup_node() {
    print_status "Backing up Node.js environment..."
    check_node

    # Create backup directory
    backup_dir="backups/node"
    mkdir -p "$backup_dir"

    # Create backup file name
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$backup_dir/ai_trading_system_node_backup_$timestamp.tar.gz"

    # Create backup
    print_status "Creating Node.js environment backup..."
    cd frontend

    # Backup package.json
    if [ -f "package.json" ]; then
        cp package.json "$backup_dir/"
    fi

    # Backup node_modules if it exists
    if [ -d "node_modules" ]; then
        tar -czf "$backup_file" node_modules
    fi

    cd ..

    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_success "Node.js environment backup completed successfully!"
        print_status "Backup saved to: $backup_file"
    else
        print_error "Node.js environment backup failed."
        exit 1
    fi
}

# Backup Docker environment
backup_docker() {
    print_status "Backing up Docker environment..."
    check_docker

    # Create backup directory
    backup_dir="backups/docker"
    mkdir -p "$backup_dir"

    # Create backup file name
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$backup_dir/ai_trading_system_docker_backup_$timestamp.tar.gz"

    # Create backup
    print_status "Creating Docker environment backup..."
    cd ..

    # Backup docker-compose.yml
    if [ -f "docker-compose.yml" ]; then
        cp docker-compose.yml "$backup_dir/"
    fi

    # Backup Dockerfiles
    if [ -f "Dockerfile" ]; then
        cp Dockerfile "$backup_dir/"
    fi
    if [ -f "Dockerfile.render.optimized" ]; then
        cp Dockerfile.render.optimized "$backup_dir/"
    fi

    # Backup Docker images
    print_status "Backing up Docker images..."
    docker images -a > "$backup_dir/docker_images.txt"
    docker ps -a > "$backup_dir/docker_containers.txt"

    # Create tar archive of backup directory
    cd backups
    tar -czf "$backup_file" docker

    cd ..

    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_success "Docker environment backup completed successfully!"
        print_status "Backup saved to: $backup_file"
    else
        print_error "Docker environment backup failed."
        exit 1
    fi
}

# Backup database
backup_database() {
    print_status "Backing up database..."
    check_python

    # Create backup directory
    backup_dir="backups/database"
    mkdir -p "$backup_dir"

    # Create backup file name
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$backup_dir/ai_trading_system_database_backup_$timestamp.sql"

    # Create backup
    print_status "Creating database backup..."
    cd app

    # Check if database configuration exists
    if [ ! -f "core/config.py" ]; then
        print_error "Database configuration not found."
        exit 1
    fi

    # Get database URL from configuration
    database_url=$(grep -oP '(?<=DATABASE_URL = ")[^"]+' core/config.py)

    # Check if database URL is valid
    if [ -z "$database_url" ]; then
        print_error "Database URL not found in configuration."
        exit 1
    fi

    # Extract database credentials
    database_type=$(echo "$database_url" | cut -d: -f1)
    if [ "$database_type" = "postgresql" ]; then
        # Backup PostgreSQL database
        print_status "Backing up PostgreSQL database..."
        pg_dump "$database_url" > "$backup_file"
    elif [ "$database_type" = "sqlite" ]; then
        # Backup SQLite database
        print_status "Backing up SQLite database..."
        database_file=$(echo "$database_url" | cut -d/ -f3)
        if [ -f "$database_file" ]; then
            cp "$database_file" "$backup_dir/"
        else
            print_warning "SQLite database file not found."
        fi
    else
        print_error "Unsupported database type: $database_type"
        exit 1
    fi

    cd ..

    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_success "Database backup completed successfully!"
        print_status "Backup saved to: $backup_file"
    else
        print_error "Database backup failed."
        exit 1
    fi
}

# Backup configuration files
backup_config() {
    print_status "Backing up configuration files..."
    check_python

    # Create backup directory
    backup_dir="backups/config"
    mkdir -p "$backup_dir"

    # Create backup file name
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$backup_dir/ai_trading_system_config_backup_$timestamp.tar.gz"

    # Create backup
    print_status "Creating configuration files backup..."
    cd app

    # Backup configuration files
    if [ -f "core/config.py" ]; then
        cp core/config.py "$backup_dir/"
    fi
    if [ -f "core/security.py" ]; then
        cp core/security.py "$backup_dir/"
    fi

    cd ..

    # Backup .env file if it exists
    if [ -f ".env" ]; then
        cp .env "$backup_dir/"
    fi

    # Create tar archive of backup directory
    cd backups
    tar -czf "$backup_file" config

    cd ..

    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_success "Configuration files backup completed successfully!"
        print_status "Backup saved to: $backup_file"
    else
        print_error "Configuration files backup failed."
        exit 1
    fi
}

# Backup logs
backup_logs() {
    print_status "Backing up logs..."
    check_python

    # Create backup directory
    backup_dir="backups/logs"
    mkdir -p "$backup_dir"

    # Create backup file name
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$backup_dir/ai_trading_system_logs_backup_$timestamp.tar.gz"

    # Create backup
    print_status "Creating logs backup..."
    cd logs

    # Backup log files
    if [ -n "$(ls -A . 2>/dev/null)" ]; then
        tar -czf "$backup_file" *
    else
        print_warning "No log files found."
    fi

    cd ..

    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_success "Logs backup completed successfully!"
        print_status "Backup saved to: $backup_file"
    else
        print_error "Logs backup failed."
        exit 1
    fi
}

# Backup everything
backup_all() {
    print_status "Starting full backup..."
    check_git
    check_python
    check_node
    check_docker
    check_tar

    # Create main backup directory
    mkdir -p backups

    # Backup Git repository
    backup_git

    # Backup Python environment
    backup_python

    # Backup Node.js environment
    backup_node

    # Backup Docker environment
    backup_docker

    # Backup database
    backup_database

    # Backup configuration files
    backup_config

    # Backup logs
    backup_logs

    print_success "Full backup completed successfully!"
}

# Main backup function
backup() {
    print_status "Starting AI Trading System backup process..."

    # Check prerequisites
    check_git
    check_python
    check_node
    check_docker
    check_tar

    # Create main backup directory
    mkdir -p backups

    # Backup Git repository
    backup_git

    # Backup Python environment
    backup_python

    # Backup Node.js environment
    backup_node

    # Backup Docker environment
    backup_docker

    # Backup database
    backup_database

    # Backup configuration files
    backup_config

    # Backup logs
    backup_logs

    print_success "Backup completed successfully!"
}

# Help function
show_help() {
    echo "AI Trading System Backup Script"
    echo "Usage: $0 [all|git|python|node|docker|database|config|logs]"
    echo ""
    echo "Options:"
    echo "  all         Backup everything (default)"
    echo "  git         Backup only Git repository"
    echo "  python      Backup only Python environment"
    echo "  node        Backup only Node.js environment"
    echo "  docker      Backup only Docker environment"
    echo "  database    Backup only database"
    echo "  config      Backup only configuration files"
    echo "  logs        Backup only logs"
    echo ""
    echo "Examples:"
    echo "  $0 all       # Backup everything"
    echo "  $0 git       # Backup only Git repository"
}

# Main script logic
if [ $# -eq 0 ]; then
    backup
    exit 0
fi

case $1 in
    "all")
        backup_all
        ;;
    "git")
        backup_git
        ;;
    "python")
        backup_python
        ;;
    "node")
        backup_node
        ;;
    "docker")
        backup_docker
        ;;
    "database")
        backup_database
        ;;
    "config")
        backup_config
        ;;
    "logs")
        backup_logs
        ;;
    *)
        show_help
        exit 1
        ;;
esac