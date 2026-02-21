@echo off
REM ============================================
REM AI TRADING SYSTEM - PAPER TRADING MODE
REM ============================================
REM DATI REALI + SOLDI FINTI
REM ============================================

echo.
echo ==========================================
echo  AI TRADING SYSTEM - PAPER TRADING
echo  DATI REALI + SOLDI SIMULATI
echo ==========================================
echo.

REM Imposta variabili ambiente
set SIMULATION_MODE=false
set USE_BINANCE_TESTNET=true

echo Configurazione:
echo   SIMULATION_MODE=false (USA DATI REALI)
echo   USE_BINANCE_TESTNET=true (SOLDI FINTI)
echo.

REM Avvia il sistema
python start_paper_trading.py

pause
