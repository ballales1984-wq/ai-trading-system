@echo off
chcp 65001 >nul
title AI Trading System - Avvio Completo

echo.
echo ========================================
echo   AI TRADING SYSTEM - AVVIO COMPLETO
echo ========================================
echo.

REM ========================================
REM 1. Backend API (Porta 8000)
REM ========================================
echo [1/5] Avvio Backend API su porta 8000...
start "Backend API" cmd /k "cd /d c:\ai-trading-system && python -m uvicorn app.main:app --reload --port 8000"
timeout /t 3 /nobreak >nul

REM ========================================
REM 2. Frontend React (Porta 5173)
REM ========================================
echo [2/5] Avvio Frontend React su porta 5173...
start "Frontend React" cmd /k "cd /d c:\ai-trading-system\frontend && npm run dev"
timeout /t 5 /nobreak >nul

REM ========================================
REM 3. Dashboard Python Dash (Porta 8050)
REM ========================================
echo [3/5] Avvio Dashboard Python su porta 8050...
start "Dashboard Python" cmd /k "cd /d c:\ai-trading-system\dashboard && python app.py"
timeout /t 3 /nobreak >nul

REM ========================================
REM 4. AI Assistant Streamlit (Porta 8501)
REM ========================================
echo [4/5] Avvio AI Assistant su porta 8501...
start "AI Assistant" cmd /k "cd /d c:\ai-trading-system && streamlit run ai_financial_dashboard.py --server.port 8501 --server.headless true"

echo.
echo ========================================
echo   SISTEMA AVVIATO!
echo ========================================
echo.
echo Servizi disponibili:
echo   - Backend API:     http://localhost:8000
echo   - Frontend:       http://localhost:5173
echo   - Dashboard:       http://localhost:5173/dashboard
echo   - Python Dash:     http://localhost:8050
echo   - AI Assistant:    http://localhost:8501
echo.
echo ========================================
echo.

REM Hint per l'utente
echo Premi un tasto per uscire...
pause >nul
