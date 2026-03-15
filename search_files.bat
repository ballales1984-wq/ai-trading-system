@echo off
REM AI Trading System - File Search Application
REM Uso: search_files.bat [pattern] [directory]
REM Esempio: search_files.bat *.py . per cercare file Python nella directory corrente

title AI Trading System - Cerca File

set ROOT=c:\ai-trading-system
cd /d %ROOT%

if "%1"=="" (
    echo.
    echo ========================================
    echo   AI TRADING SYSTEM - CERCA FILE
    echo ========================================
    echo Uso: %0 [pattern] [directory]
    echo Esempi:
    echo   %0 *.bat .          ^(cerca tutti i .bat nella root^)
    echo   %0 *.py app         ^(cerca Python in app/^)
    echo   %0 main             . ^(cerca files con 'main'^)
    echo.
    pause
    exit /b 0
)

set PATTERN=%1
set SEARCH_DIR=%2
if "%SEARCH_DIR%"=="" set SEARCH_DIR=.

echo.
echo [CERCA] Pattern: %PATTERN% in %SEARCH_DIR%
echo.

dir /s /b "%SEARCH_DIR%\%PATTERN%" 2>nul | findstr /i "%PATTERN%" || echo Nessun file trovato.

echo.
echo [OK] Ricerca completata.
pause
