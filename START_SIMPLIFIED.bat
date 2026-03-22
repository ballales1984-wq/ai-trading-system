@echo off
title AI Trading System - Simplified Startup
echo ===================================================
echo   AVVIO SISTEMA AI TRADING - VERSIONE SEMPLIFICATA
echo ===================================================
echo.

:: 1. Pulizia processi precedenti
echo [1/5] Pulizia processi vecchi...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

:: 2. Avvio Backend API (Porta 8000)
echo [2/5] Avvio Backend API (Porta 8000)...
start "BACKEND - Porta 8000" cmd /c "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
timeout /t 15 /nobreak >nul

:: 3. Avvio AutoTrader (Loop Mercato)
echo [3/5] Avvio AutoTrader (BTC, ETH, SOL)...
start "AUTOTRADER - Live Monitor" cmd /c "python main_auto_trader.py --mode live --dry-run --interval 60 --assets BTC/USDT ETH/USDT SOL/USDT"
timeout /t 3 /nobreak >nul

:: 4. Avvio Frontend Dashboard (Porta 5173) - INCLUDE TUTTO!
echo [4/5] Avvio Frontend Dashboard (Porta 5173)...
echo    NOTA: Include ML Monitoring, Investor Portal, AI Assistant
cd frontend
start "FRONTEND - Porta 5173" cmd /c "npm run dev"
cd ..

:: 5. Riepilogo Finale
echo.
echo ===================================================
echo   SISTEMA AVVIATO CON SUCCESSO!
echo ===================================================
echo.
echo Tutti i servizi sono ora accessibili da http://localhost:5173
echo.
echo - DASHBOARD:        http://localhost:5173/dashboard
echo - ML MONITORING:    http://localhost:5173/ml-monitoring
echo - INVESTOR PORTAL: http://localhost:5173/investor-portal
echo - AI ASSISTANT:    http://localhost:5173/ai-assistant
echo.
echo NOTA: Le porte 8050, 8051, 8502 NON sono piu necessarie!
echo.
echo Premi un tasto per chiudere questo script di avvio...
pause >nul
