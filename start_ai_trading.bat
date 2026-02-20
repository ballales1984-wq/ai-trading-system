@echo off
REM ============================================
REM AI Trading System - Windows Launcher
REM Docker Compose Stable Configuration
REM RAM: 4 GB | ROM: 3 GB
REM ============================================

title AI Trading System

echo.
echo ================================================
echo    AI TRADING SYSTEM - STABLE LAUNCHER
echo    RAM: 4 GB | ROM: 3 GB
echo ================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [OK] Docker is running.
echo.

REM Check for .env file
if not exist ".env" (
    echo [WARNING] .env file not found. Creating template...
    echo.
    echo # AI Trading System - Environment Variables > .env
    echo # API Keys >> .env
    echo BINANCE_API_KEY=your_binance_api_key >> .env
    echo BINANCE_API_SECRET=your_binance_api_secret >> .env
    echo BYBIT_API_KEY=your_bybit_api_key >> .env
    echo BYBIT_API_SECRET=your_bybit_api_secret >> .env
    echo. >> .env
    echo # Database >> .env
    echo DB_USER=trading >> .env
    echo DB_PASSWORD=trading_secret >> .env
    echo DB_NAME=trading_db >> .env
    echo. >> .env
    echo # Trading Mode >> .env
    echo TRADING_MODE=paper >> .env
    echo. >> .env
    echo [CREATED] .env template created. Please edit with your API keys.
    echo.
    notepad .env
    echo.
    echo After editing .env, run this script again.
    pause
    exit /b 0
)

echo [OK] .env file found.
echo.

REM Stop any existing containers
echo [STEP 1] Stopping existing containers...
docker-compose -f docker-compose.stable.yml down 2>nul

REM Start containers
echo [STEP 2] Starting containers...
echo.
echo    - Trading Engine (2 GB RAM)
echo    - Dashboard (512 MB RAM)
echo    - Database (1 GB RAM)
echo    - Redis Cache (512 MB RAM)
echo.

docker-compose -f docker-compose.stable.yml up -d

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start containers.
    echo Check the logs with: docker-compose -f docker-compose.stable.yml logs
    pause
    exit /b 1
)

echo.
echo ================================================
echo    SYSTEM STARTED SUCCESSFULLY!
echo ================================================
echo.
echo    Dashboard:  http://localhost:8050
echo    API:        http://localhost:8000
echo    Database:   localhost:5432
echo    Redis:      localhost:6379
echo.
echo ================================================
echo.

REM Wait for dashboard to be ready
echo Waiting for dashboard to be ready...
timeout /t 10 /nobreak >nul

REM Open dashboard in browser
echo Opening dashboard in browser...
start http://localhost:8050

echo.
echo Press any key to view container logs (Ctrl+C to exit)...
pause >nul

REM Show logs
docker-compose -f docker-compose.stable.yml logs -f
