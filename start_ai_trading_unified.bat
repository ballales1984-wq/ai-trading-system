@echo off
setlocal EnableExtensions
chcp 65001 >nul
title AI Trading System - Avvio Unificato

set "ROOT=%~dp0"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"
set "HEALTH_URL=http://localhost:%BACKEND_PORT%/health"

echo ==========================================
echo  AI Trading System - Avvio Unificato
echo ==========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato nel PATH.
    goto :fail
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] npm non trovato nel PATH.
    goto :fail
)

where ngrok >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] ngrok non trovato nel PATH.
    goto :fail
)

echo [1/5] Verifica backend...
curl -s "%HEALTH_URL%" >nul 2>&1
if errorlevel 1 (
    netstat -ano | findstr /R /C:":%BACKEND_PORT% .*LISTENING" >nul
    if errorlevel 1 (
        echo [INFO] Backend non attivo, avvio backend...
        start "Backend API" cmd /k "cd /d %ROOT% && python -m app.main"
    ) else (
        echo [ERRORE] Porta %BACKEND_PORT% gia in uso da altro processo e /health non risponde.
        echo [INFO] Chiudi il processo sulla porta %BACKEND_PORT% e rilancia.
        goto :fail
    )
) else (
    echo [OK] Backend gia attivo.
)

call :wait_backend
if errorlevel 1 (
    echo [ERRORE] Backend non raggiungibile dopo l'avvio.
    goto :fail
)
echo [OK] Backend pronto.

echo [2/5] Verifica frontend...
netstat -ano | findstr /R /C:":%FRONTEND_PORT% .*LISTENING" >nul
if errorlevel 1 (
    echo [INFO] Frontend non attivo, avvio frontend...
    start "Frontend Dev Server" cmd /k "cd /d %ROOT%frontend && npm run dev"
) else (
    echo [OK] Frontend gia in ascolto sulla porta %FRONTEND_PORT%.
)

echo [3/5] Avvio tunnel ngrok...
tasklist /FI "IMAGENAME eq ngrok.exe" | find /I "ngrok.exe" >nul
if not errorlevel 1 (
    echo [INFO] Chiudo ngrok gia attivo per evitare conflitti...
    taskkill /F /IM ngrok.exe >nul 2>&1
    timeout /t 1 /nobreak >nul
)
start "Ngrok Tunnel" cmd /k "cd /d %ROOT% && ngrok http %BACKEND_PORT% --host-header=localhost:%BACKEND_PORT%"

echo [4/5] Avvio app desktop...
if exist "%ROOT%dist\ai_trading_system.exe" (
    start "" "%ROOT%dist\ai_trading_system.exe"
    echo [OK] App desktop avviata.
) else (
    echo [AVVISO] Eseguibile non trovato: %ROOT%dist\ai_trading_system.exe
)

echo [5/5] Riepilogo
echo Backend : http://localhost:%BACKEND_PORT%
echo Frontend: http://localhost:%FRONTEND_PORT%
echo Ngrok   : controlla finestra "Ngrok Tunnel"
echo.
echo Avvio completato. Chiusura finestra launcher tra 8 secondi...
timeout /t 8 /nobreak >nul
exit /b 0

:wait_backend
for /L %%i in (1,1,30) do (
    curl -s "%HEALTH_URL%" >nul 2>&1
    if not errorlevel 1 exit /b 0
    timeout /t 1 /nobreak >nul
)
exit /b 1

:fail
echo.
echo Avvio interrotto. Chiusura finestra launcher tra 10 secondi...
timeout /t 10 /nobreak >nul
exit /b 1
