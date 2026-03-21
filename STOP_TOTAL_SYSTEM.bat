@echo off
title AI Trading System - Total Shutdown
echo ===================================================
echo   ARRESTO SISTEMA AI TRADING - TUTTI I MODULI
echo ===================================================
echo.

echo [1/2] Chiusura processi Python (Backend, Trader, Dashboards)...
taskkill /F /IM python.exe /T >nul 2>&1

echo [2/2] Chiusura processi Node/Vite (Frontend)...
taskkill /F /IM node.exe /T >nul 2>&1

echo.
echo ===================================================
echo   SISTEMA ARRESTATO CON SUCCESSO
echo ===================================================
echo.
pause
