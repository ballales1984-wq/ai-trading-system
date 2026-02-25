@echo off
chcp 65001 >nul
title 🤖 AI Trading System - Setup Collegamento Desktop
color 0A

echo ============================================
echo   🤖 AI TRADING SYSTEM - SETUP DESKTOP
echo ============================================
echo.

echo [📌] Creazione collegamento sul desktop...
echo.

:: Esegui lo script PowerShell
powershell -ExecutionPolicy Bypass -File "%~dp0create_desktop_shortcut.ps1"

if errorlevel 1 (
    echo.
    echo [❌] Errore durante la creazione del collegamento!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   ✅ SETUP COMPLETATO!
echo ============================================
echo.
echo Ora puoi:
echo 1. Chiudere questa finestra
echo 2. Trovare il collegamento "🤖 AI Trading System" sul desktop
echo 3. Fare doppio clic per avviare tutto!
echo.
pause
