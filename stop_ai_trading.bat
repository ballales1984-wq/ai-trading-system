@echo off
REM ============================================
REM AI Trading System - Stop Script
REM ============================================

title AI Trading System - Stop

echo.
echo ================================================
echo    AI TRADING SYSTEM - STOPPING CONTAINERS
echo ================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running.
    pause
    exit /b 1
)

echo Stopping all containers...
docker-compose -f docker-compose.stable.yml down

echo.
echo ================================================
echo    ALL CONTAINERS STOPPED
echo ================================================
echo.
echo To start again, run: start_ai_trading.bat
echo.
pause
