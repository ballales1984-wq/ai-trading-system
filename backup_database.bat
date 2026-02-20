@echo off
REM ============================================
REM AI Trading System - Database Backup Script
REM ============================================

title AI Trading System - Backup

set BACKUP_DIR=backups
set TIMESTAMP=%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo.
echo ================================================
echo    AI TRADING SYSTEM - DATABASE BACKUP
echo ================================================
echo.

REM Create backup directory
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

echo Backup directory: %BACKUP_DIR%
echo Timestamp: %TIMESTAMP%
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running.
    pause
    exit /b 1
)

REM Check if database container is running
docker ps | findstr ai_trading_db >nul
if errorlevel 1 (
    echo [WARNING] Database container is not running.
    echo Starting database container...
    docker-compose -f docker-compose.stable.yml up -d postgres
    timeout /t 10 /nobreak >nul
)

echo Creating database backup...
echo.

REM Create backup using pg_dump
docker exec ai_trading_db pg_dump -U trading -d trading_db -F c -f /var/lib/postgresql/data/backup_%TIMESTAMP%.dump

if errorlevel 1 (
    echo [ERROR] Backup failed.
    pause
    exit /b 1
)

echo.
echo [OK] Database backup created successfully.
echo.

REM Also backup to local directory
echo Copying backup to local directory...
docker cp ai_trading_db:/var/lib/postgresql/data/backup_%TIMESTAMP%.dump %BACKUP_DIR%\backup_%TIMESTAMP%.dump

echo.
echo ================================================
echo    BACKUP COMPLETED
echo ================================================
echo.
echo Backup file: %BACKUP_DIR%\backup_%TIMESTAMP%.dump
echo.

REM List existing backups
echo Existing backups:
echo.
dir /b %BACKUP_DIR%\*.dump 2>nul
echo.

REM Cleanup old backups (keep last 10)
echo Cleaning up old backups (keeping last 10)...
for /f "skip=10 delims=" %%F in ('dir /b /o-d %BACKUP_DIR%\*.dump 2^>nul') do (
    echo Deleting: %%F
    del "%BACKUP_DIR%\%%F"
)

echo.
pause
