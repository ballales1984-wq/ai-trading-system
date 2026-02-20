#!/bin/bash
# ============================================================
# AI Trading System - Stable Version Startup Script
# ============================================================
# Version: 1.0.0-stable
# 
# Usage:
#   ./start_stable.sh           - Start all services
#   ./start_stable.sh --build   - Rebuild and start
#   ./start_stable.sh --stop    - Stop all services
#   ./start_stable.sh --logs    - Show logs
#   ./start_stable.sh --status  - Show status

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.stable.yml"
PROJECT_NAME="ai-trading-stable"
VERSION="1.0.0-stable"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "   AI Trading System - Stable Version ${VERSION}"
    echo "============================================================"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed.${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed.${NC}"
        exit 1
    fi
    
    # Check .env file
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}Warning: .env file not found. Creating from template...${NC}"
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${GREEN}Created .env file. Please edit it with your API keys.${NC}"
        else
            echo -e "${RED}Error: .env.example not found. Please create .env manually.${NC}"
            exit 1
        fi
    fi
    
    # Create data directories
    mkdir -p data/{pgdata,redisdata,ml_temp,models,logs,cache}
    
    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Build images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME build --no-cache
    echo -e "${GREEN}Build complete${NC}"
}

# Start services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
    echo -e "${GREEN}Services started${NC}"
    
    echo ""
    echo -e "${BLUE}Services available at:${NC}"
    echo -e "  Dashboard:  ${GREEN}http://localhost:8050${NC}"
    echo -e "  API:        ${GREEN}http://localhost:8000${NC}"
    echo -e "  Database:   ${GREEN}localhost:5432${NC}"
    echo -e "  Redis:      ${GREEN}localhost:6379${NC}"
    echo ""
}

# Stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down
    echo -e "${GREEN}Services stopped${NC}"
}

# Show logs
show_logs() {
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
}

# Show status
show_status() {
    echo -e "${BLUE}Service Status:${NC}"
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
    
    echo ""
    echo -e "${BLUE}Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null || echo "No running containers"
}

# Run resource monitor
run_monitor() {
    echo -e "${YELLOW}Running resource monitor...${NC}"
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME --profile monitoring up -d resource-monitor
}

# Clean up
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v --remove-orphans
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Main script
main() {
    print_banner
    
    case "${1:-start}" in
        --build|-b)
            check_prerequisites
            build_images
            start_services
            ;;
        --stop|-s)
            stop_services
            ;;
        --restart|-r)
            stop_services
            check_prerequisites
            start_services
            ;;
        --logs|-l)
            show_logs
            ;;
        --status|-st)
            show_status
            ;;
        --monitor|-m)
            run_monitor
            ;;
        --clean|-c)
            cleanup
            ;;
        --help|-h)
            echo "Usage: $0 [option]"
            echo ""
            echo "Options:"
            echo "  (none)      Start services (default)"
            echo "  --build     Rebuild and start"
            echo "  --stop      Stop services"
            echo "  --restart   Restart services"
            echo "  --logs      Show logs"
            echo "  --status    Show status"
            echo "  --monitor   Start resource monitor"
            echo "  --clean     Remove all containers and volumes"
            echo "  --help      Show this help"
            ;;
        start|*)
            check_prerequisites
            start_services
            ;;
    esac
}

main "$@"
