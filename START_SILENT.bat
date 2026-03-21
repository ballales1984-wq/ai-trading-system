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
:: 2. Avvio Backend API in background senza finestra (Porta 8000)
:: ============================================================
echo [2/8] Avvio Backend API (127.0.0.1:8000)...
powershell -Command "Start-Process python -ArgumentList '-m uvicorn app.main:app --host 0.0.0.0 --port 8000' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\api_backend.log' -RedirectStandardError 'data\logs\api_backend_err.log'"
timeout /t 5 /nobreak >nul

:: ============================================================
:: 3. Avvio AutoTrader in background (Loop Mercato)
:: ============================================================
echo [3/8] Avvio AutoTrader (BTC, ETH, SOL)...
powershell -Command "Start-Process python -ArgumentList 'main_auto_trader.py --mode live --dry-run --interval 60 --assets BTC/USDT ETH/USDT SOL/USDT' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\auto_trader_live.log' -RedirectStandardError 'data\logs\auto_trader_err.log'"
timeout /t 3 /nobreak >nul

:: ============================================================
:: 4. Avvio Monitoraggio ML in background (Porta 8050)
:: ============================================================
echo [4/8] Avvio Monitoraggio ML (127.0.0.1:8050)...
powershell -Command "Start-Process python -ArgumentList 'dashboard\dashboard.py' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\ml_monitor.log' -RedirectStandardError 'data\logs\ml_monitor_err.log'"
timeout /t 3 /nobreak >nul

:: ============================================================
:: 5. Avvio Investor Portal in background (Porta 8051)
:: ============================================================
echo [5/8] Avvio Investor Portal (127.0.0.1:8051)...
powershell -Command "Start-Process python -ArgumentList 'dashboard_investor.py' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\investor_portal.log' -RedirectStandardError 'data\logs\investor_portal_err.log'"
timeout /t 3 /nobreak >nul

:: ============================================================
:: 6. Avvio AI Assistant in background (Porta 8502)
:: ============================================================
echo [6/8] Avvio AI Assistant (127.0.0.1:8502)...
powershell -Command "Start-Process streamlit -ArgumentList 'run ai_financial_dashboard.py --server.port 8502' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\ai_assistant.log' -RedirectStandardError 'data\logs\ai_assistant_err.log'"
timeout /t 5 /nobreak >nul

:: ============================================================
:: 7. Avvio Frontend Dashboard in background (Porta 5173)
:: ============================================================
echo [7/8] Avvio Frontend Dashboard (127.0.0.1:5173)...
cd frontend
powershell -Command "Start-Process npm -ArgumentList 'run dev' -WindowStyle Hidden -RedirectStandardOutput '..\data\logs\frontend.log' -RedirectStandardError '..\data\logs\frontend_err.log' -WorkingDirectory '%CD%'"
cd ..

:: ============================================================
:: 8. Attesa per avvio servizi e riepilogo
:: ============================================================
echo [8/8] Attesa completamento avvio servizi...
timeout /t 10 /nobreak >nul

echo.
echo ===================================================
echo   SISTEMA AVVIATO IN MODALITA' SILENZIOSA
echo ===================================================
echo.
echo Tutti i servizi sono stati avviati in background senza finestre.
echo I log sono stati reindirizzati nella cartella:
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
dir /b data\logs\*.log 2>nul
echo.
echo Per monitorare i log in tempo reale, usa:
echo   powershell -Command "Get-Content -Wait -Tail 30 data\logs\[nome_log].log"
echo ===================================================
echo.
echo [NOTA] I processi sono in esecuzione in background.
echo        Non ci sono finestre di terminale aperte.
echo.
pause
