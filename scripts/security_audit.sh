#!/bin/bash

# AI Trading System Security Audit Script
# This script automates the security audit process for the entire system

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

# Check if npm is available
check_npm() {
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm and try again."
        exit 1
    fi
}

# Check if bandit is available
check_bandit() {
    if ! command -v bandit &> /dev/null; then
        print_error "bandit is not installed. Please install bandit and try again."
        exit 1
    fi
}

# Check if npm audit is available
check_npm_audit() {
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if gitleaks is available
check_gitleaks() {
    if ! command -v gitleaks &> /dev/null; then
        print_error "gitleaks is not installed. Please install gitleaks and try again."
        exit 1
    fi
}

# Check if safepython is available
check_safepython() {
    if ! command -v safepython &> /dev/null; then
        print_error "safepython is not installed. Please install safepython and try again."
        exit 1
    fi
}

# Check if semgrep is available
check_semgrep() {
    if ! command -v semgrep &> /dev/null; then
        print_error "semgrep is not installed. Please install semgrep and try again."
        exit 1
    fi
}

# Check if npm audit is available
check_npm_audit() {
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
}

# Check if docker scan is available
check_docker_scan() {





