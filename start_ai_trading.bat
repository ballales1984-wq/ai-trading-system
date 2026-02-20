@echo off
REM ============================================
REM AI Trading System - Stable Launcher
REM Avvia il sistema come applicazione standalone
REM ============================================

title AI Trading System

echo.
echo  ================================================
echo   AI TRADING SYSTEM - Mini Hedge Fund
echo  ================================================
echo.

REM Salva il percorso corrente
cd /d %~dp0

REM Controlla se Docker e' installato
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Docker non e' installato o non e' in PATH
    echo Installa Docker Desktop da: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Controlla se Docker e' in esecuzione
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Docker non e' in esecuzione
    echo Avvia Docker Desktop e riprova
    pause
    exit /b 1
)

REM Controlla se .env esiste
if not exist .env (
    echo [AVVISO] File .env non trovato
    echo Creando .env di esempio...
    (
        echo # AI Trading System - Configurazione
        echo.
        echo # Binance API
        echo BINANCE_API_KEY=your_api_key
        echo BINANCE_SECRET_KEY=your_secret_key
        echo USE_BINANCE_TESTNET=true
        echo.
        echo # News API
        echo NEWS_API_KEY=your_newsapi_key
        echo.
        echo # Altre API
        echo ALPHA_VANTAGE_API_KEY=your_av_key
    ) > .env
    echo File .env creato. Modifica le API keys prima di riavviare.
    pause
)

echo [1/3] Verifica container esistenti...
docker-compose -f docker-compose.stable.yml ps

echo.
echo [2/3] Avvio container...
docker-compose -f docker-compose.stable.yml up -d

echo.
echo [3/3] Verifica stato...
timeout /t 5 /nobreak >nul
docker-compose -f docker-compose.stable.yml ps

echo.
echo  ================================================
echo   SISTEMA AVVIATO CON SUCCESSO!
echo  ================================================
echo.
echo   Dashboard:  http://localhost:8050
echo   API:        http://localhost:8000
echo   Database:   localhost:5432
echo.
echo  ================================================
echo.
echo Premi un tasto per aprire la Dashboard nel browser...
pause >nul

REM Apri il browser
start http://localhost:8050

echo.
echo Il sistema e' in esecuzione in background.
echo Per fermare: docker-compose -f docker-compose.stable.yml down
echo.
