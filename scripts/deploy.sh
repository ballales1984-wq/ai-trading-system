#!/bin/bash

# AI Trading System Deployment Script
# This script automates the deployment process to various platforms

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

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi
}

# Deploy to Docker
deploy_docker() {
    print_status "Starting Docker deployment..."
    check_docker

    # Build and run Docker containers
    print_status "Building Docker images..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d

    # Wait for services to start
    print_status "Waiting for services to start (2-3 minutes)..."
    sleep 120

    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Docker deployment completed successfully!"
        print_status "Access the system at: http://localhost:8000"
    else
        print_error "Docker deployment failed. Check logs for details."
        exit 1
    fi
}

# Deploy to Render
deploy_render() {
    print_status "Starting Render deployment..."
    check_docker

    # Build Docker image for Render
    print_status "Building Docker image for Render..."
    docker build -f Dockerfile.render.optimized -t ai-trading-system .

    # Deploy to Render (this would be automated with Render CLI)
    print_status "Render deployment requires manual steps:"
    print_status "1. Go to https://render.com"
    print_status "2. Create a new Web Service"
    print_status "3. Connect your GitHub repository"
    print_status "4. Configure environment variables"
    print_status "5. Deploy!"
    print_success "Render deployment guide completed!"
}

# Deploy to Vercel
deploy_vercel() {
    print_status "Starting Vercel deployment..."
    check_docker

    # Build frontend for Vercel
    print_status "Building frontend for Vercel..."
    cd frontend
    npm install
    npm run build
    cd ..

    # Deploy to Vercel (this would be automated with Vercel CLI)
    print_status "Vercel deployment requires manual steps:"
    print_status "1. Install Vercel CLI: npm install -g vercel"
    print_status "2. Run: vercel --prod"
    print_status "3. Connect your GitHub repository"
    print_status "4. Configure build settings"
    print_status "5. Deploy!"
    print_success "Vercel deployment guide completed!"
}

# Main deployment function
deploy() {
    case $1 in
        "docker")
            deploy_docker
            ;;
        "render")
            deploy_render
            ;;
        "vercel")
            deploy_vercel
            ;;
        "all")
            deploy_docker
            deploy_render
            deploy_vercel
            ;;
        *)
            print_error "Invalid deployment target. Use: docker, render, vercel, or all"
            echo "Usage: $0 [docker|render|vercel|all]"
            exit 1
            ;;
    esac
}

# Help function
show_help() {
    echo "AI Trading System Deployment Script"
    echo "Usage: $0 [docker|render|vercel|all]"
    echo ""
    echo "Options:"
    echo "  docker    Deploy to local Docker"
    echo "  render    Deploy to Render.com"
    echo "  vercel    Deploy to Vercel.com"
    echo "  all       Deploy to all platforms"
    echo ""
    echo "Examples:"
    echo "  $0 docker    # Deploy to local Docker"
    echo "  $0 all       # Deploy to all platforms"
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Execute deployment
deploy $1