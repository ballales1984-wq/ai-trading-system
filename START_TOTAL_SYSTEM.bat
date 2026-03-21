@echo off
title AI Trading System - Total Startup
echo ===================================================
echo   AVVIO SISTEMA AI TRADING - TUTTI I MODULI
echo ===================================================
echo.

:: 1. Pulizia processi precedenti
echo [1/8] Pulizia processi vecchi...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

:: 2. Avvio Backend API (Porta 8000)
echo [2/8] Avvio Backend API (Porta 8000)...
start "BACKEND - Porta 8000" cmd /c "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
timeout /t 15 /nobreak >nul

:: 3. Avvio AutoTrader (Loop Mercato)
echo [3/8] Avvio AutoTrader (BTC, ETH, SOL)...
start "AUTOTRADER - Live Monitor" cmd /c "python main_auto_trader.py --mode live --dry-run --interval 60 --assets BTC/USDT ETH/USDT SOL/USDT"
timeout /t 3 /nobreak >nul

:: 4. Avvio Monitoraggio ML (Porta 8050)
echo [4/8] Avvio Monitoraggio ML (Porta 8050)...
start "ML MONITOR - Porta 8050" cmd /c "python dashboard/dashboard.py"
timeout /t 3 /nobreak >nul

:: 5. Avvio Investor Portal (Porta 8051)
echo [5/8] Avvio Investor Portal (Porta 8051)...
start "INVESTOR PORTAL - Porta 8051" cmd /c "python dashboard_investor.py"
timeout /t 3 /nobreak >nul

:: 6. Avvio AI Assistant (Porta 8502)
echo [6/8] Avvio AI Financial Assistant (Porta 8502)...
start "AI ASSISTANT - Porta 8502" cmd /c "streamlit run ai_financial_dashboard.py --server.port 8502"
timeout /t 5 /nobreak >nul

:: 7. Avvio Frontend Dashboard (Porta 5173)
echo [7/8] Avvio Frontend Dashboard (Porta 5173)...
cd frontend
start "FRONTEND - Porta 5173" cmd /c "npm run dev"
cd ..

:: 8. Riepilogo Finale
echo.
echo ===================================================
echo   SISTEMA AVVIATO CON SUCCESSO!
echo ===================================================
echo.
echo Puoi accedere alle interfacce qui:
echo - DASHBOARD PRINCIPALE: http://localhost:5173
echo - MONITORAGGIO ML:      http://localhost:8050
echo - PORTALE INVESTITORI:  http://localhost:8051
echo - AI ASSISTANT:         http://localhost:8502
echo.
echo NOTA: Tieni aperte le altre finestre cmd per il funzionamento.
echo Premi un tasto per chiudere questo script di avvio...
pause >nul
