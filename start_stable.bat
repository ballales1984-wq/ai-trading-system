@echo off
REM ============================================================
REM AI Trading System - Stable Version Startup Script (Windows)
REM ============================================================
REM Version: 1.0.0-stable
REM
REM Usage:
REM   start_stable.bat           - Start all services
REM   start_stable.bat build     - Rebuild and start
REM   start_stable.bat stop      - Stop all services
REM   start_stable.bat logs      - Show logs
REM   start_stable.bat status    - Show status
REM   start_stable.bat clean     - Remove all containers and volumes

setlocal enabledelayedexpansion

REM Configuration
set COMPOSE_FILE=docker-compose.stable.yml
set PROJECT_NAME=ai-trading-stable
set VERSION=1.0.0-stable

REM Colors (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

REM Print banner
echo.
echo %BLUE%============================================================%NC%
echo    AI Trading System - Stable Version %VERSION%
echo %BLUE%============================================================%NC%
echo.

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: Docker is not installed or not running.%NC%
    exit /b 1
)

REM Check Docker Compose
docker compose version >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: Docker Compose is not installed.%NC%
    exit /b 1
)

REM Handle commands
if "%1"=="" goto start
if "%1"=="start" goto start
if "%1"=="build" goto build
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="logs" goto logs
if "%1"=="status" goto status
if "%1"=="clean" goto clean
if "%1"=="help" goto help
goto help

:start
echo %YELLOW%Starting services...%NC%

REM Check .env file
if not exist ".env" (
    echo %YELLOW%Warning: .env file not found.%NC%
    if exist ".env.example" (
        copy .env.example .env >nul
        echo %GREEN%Created .env file. Please edit it with your API keys.%NC%
    ) else (
        echo %RED%Error: .env.example not found. Please create .env manually.%NC%
        exit /b 1
    )
)

REM Create data directories
if not exist "data" mkdir data
if not exist "data\pgdata" mkdir data\pgdata
if not exist "data\redisdata" mkdir data\redisdata
if not exist "data\ml_temp" mkdir data\ml_temp
if not exist "data\models" mkdir data\models
if not exist "data\logs" mkdir data\logs
if not exist "data\cache" mkdir data\cache

docker compose -f %COMPOSE_FILE% -p %PROJECT_NAME% up -d
if errorlevel 1 (
    echo %RED%Failed to start services.%NC%
    exit /b 1
)

echo.
echo %GREEN%Services started!%NC%
echo.
echo %BLUE%Services available at:%NC%
echo   Dashboard:  %GREEN%http://localhost:8050%NC%
echo   API:        %GREEN%http://localhost:8000%NC%
echo   Database:   %GREEN%localhost:5432%NC%
echo   Redis:      %GREEN%localhost:6379%NC%
echo.
goto end

:build
echo %YELLOW%Building Docker images...%NC%
docker compose -f %COMPOSE_FILE% -p %PROJECT_NAME% build --no-cache
if errorlevel 1 (
    echo %RED%Build failed.%NC%
    exit /b 1
)
echo %GREEN%Build complete.%NC%
goto start

:stop
echo %YELLOW%Stopping services...%NC%
docker compose -f %COMPOSE_FILE% -p %PROJECT_NAME% down
echo %GREEN%Services stopped.%NC%
goto end

:restart
call :stop
call :start
goto end

:logs
docker compose -f %COMPOSE_FILE% -p %PROJECT_NAME% logs -f
goto end

:status
echo %BLUE%Service Status:%NC%
docker compose -f %COMPOSE_FILE% -p %PROJECT_NAME% ps
echo.
echo %BLUE%Resource Usage:%NC%
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>nul
goto end

:clean
echo %YELLOW%Cleaning up...%NC%
docker compose -f %COMPOSE_FILE% -p %PROJECT_NAME% down -v --remove-orphans
echo %GREEN%Cleanup complete.%NC%
goto end

:help
echo Usage: %0 [option]
echo.
echo Options:
echo   (none)      Start services (default)
echo   build       Rebuild and start
echo   stop        Stop services
echo   restart     Restart services
echo   logs        Show logs
echo   status      Show status
echo   clean       Remove all containers and volumes
echo   help        Show this help
goto end

:end
endlocal
