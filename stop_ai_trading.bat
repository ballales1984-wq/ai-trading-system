@echo off
REM ============================================
REM AI Trading System - Stop Script
REM Ferma tutti i container
REM ============================================

title Ferma AI Trading System

echo.
echo  ================================================
echo   FERMA AI TRADING SYSTEM
echo  ================================================
echo.

cd /d %~dp0

echo Fermando i container...
docker-compose -f docker-compose.stable.yml down

echo.
echo  ================================================
echo   Sistema fermato
echo  ================================================
echo.
pause
