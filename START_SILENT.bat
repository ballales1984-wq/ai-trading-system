@echo off
title AI Trading System - Silent Startup
echo ===================================================
echo   AVVIO SISTEMA AI TRADING - MODALITA' SILENZIOSA
echo ===================================================
echo.

:: ============================================================
:: 0. Creazione cartella log se non esiste
:: ============================================================
echo [0/8] Preparazione ambiente di log...
if not exist "data\logs" mkdir "data\logs"

:: Pulizia vecchi log (opzionale - mantieni solo ultimi 7 giorni)
forfiles /P "data\logs" /S /M *.log /D -7 /C "cmd /c del @path" 2>nul

:: ============================================================
:: 1. Pulizia processi precedenti (fantasma/residui)
:: ============================================================
echo [1/8] Pulizia processi residui da esecuzioni precedenti...
echo      - Chiusura Python (uvicorn, streamlit, trader)...
taskkill /F /IM python.exe /T >nul 2>&1
echo      - Chiusura Node (npm, vite)...
taskkill /F /IM node.exe /T >nul 2>&1
echo      - Chiusura cmd.exe residui...
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq BACKEND" /T >nul 2>&1
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq AUTOTRADER" /T >nul 2>&1
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq ML MONITOR" /T >nul 2>&1
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq INVESTOR PORTAL" /T >nul 2>&1
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq AI ASSISTANT" /T >nul 2>&1
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq FRONTEND" /T >nul 2>&1
echo      - Attesa completamento chiusura...
timeout /t 3 /nobreak >nul
echo      [OK] Pulizia completata.

:: ============================================================
:: 2. Avvio Backend API (Porta 8000)
:: ============================================================
echo [2/8] Avvio Backend API (127.0.0.1:8000)...
start "BACKEND" cmd /c "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > data\logs\api_backend.log 2>&1"
timeout /t 15 /nobreak >nul

:: ============================================================
:: 3. Avvio AutoTrader (Loop Mercato)
:: ============================================================
echo [3/8] Avvio AutoTrader (BTC, ETH, SOL)...
start "AUTOTRADER" cmd /c "python main_auto_trader.py --mode live --dry-run --interval 60 --assets BTC/USDT ETH/USDT SOL/USDT > data\logs\auto_trader_live.log 2>&1"
timeout /t 3 /nobreak >nul

:: ============================================================
:: 4. Avvio Monitoraggio ML (Porta 8050)
:: ============================================================
echo [4/8] Avvio Monitoraggio ML (127.0.0.1:8050)...
start "ML MONITOR" cmd /c "python dashboard\dashboard.py > data\logs\ml_monitor.log 2>&1"
timeout /t 3 /nobreak >nul

:: ============================================================
:: 5. Avvio Investor Portal (Porta 8051)
:: ============================================================
echo [5/8] Avvio Investor Portal (127.0.0.1:8051)...
start "INVESTOR PORTAL" cmd /c "python dashboard_investor.py > data\logs\investor_portal.log 2>&1"
timeout /t 3 /nobreak >nul

:: ============================================================
:: 6. Avvio AI Assistant (Porta 8502)
:: ============================================================
echo [6/8] Avvio AI Assistant (127.0.0.1:8502)...
start "AI ASSISTANT" cmd /c "streamlit run ai_financial_dashboard.py --server.port 8502 > data\logs\ai_assistant.log 2>&1"
timeout /t 5 /nobreak >nul

:: ============================================================
:: 7. Avvio Frontend Dashboard (Porta 5173)
:: ============================================================
echo [7/8] Avvio Frontend Dashboard (Porta 5173)...
cd frontend
start "FRONTEND" cmd /c "npm run dev > ..\data\logs\frontend.log 2>&1"
cd ..

:: ============================================================
:: 8. Riepilogo Finale
:: ============================================================
echo.
echo ===================================================
echo   SISTEMA AVVIATO IN MODALITA' SILENZIOSA
echo ===================================================
echo.
echo Tutti i log sono stati reindirizzati nella cartella:
echo   data\logs\
echo.
echo Interfacce disponibili:
echo   - DASHBOARD PRINCIPALE:  http://localhost:5173
echo   - MONITORAGGIO ML:       http://127.0.0.1:8050
echo   - PORTALE INVESTITORI:   http://127.0.0.1:8051
echo   - AI ASSISTANT:          http://127.0.0.1:8502
echo   - API BACKEND:           http://127.0.0.1:8000
echo.
echo Per arrestare il sistema usa: STOP_TOTAL_SYSTEM.bat
echo.
echo Log disponibili in data\logs\:
dir /b data\logs\*.log
echo.
echo ===================================================
echo   Questa finestra rimane aperta per mantenere i servizi attivi.
echo   Per vedere i log in tempo reale, apri nuovi terminali con:
echo   type data\logs\[nome_log].log
echo ===================================================
echo.
pause
