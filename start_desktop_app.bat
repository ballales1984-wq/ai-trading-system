@echo off
chcp 65001 >nul
echo ==========================================
echo  AI Trading System - Avvio Desktop + Ngrok
echo ==========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato nel PATH
    pause
    exit /b 1
)

ngrok version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] ngrok non trovato nel PATH
    pause
    exit /b 1
)

echo [1/4] Avvio backend API...
start "Backend API" cmd /k "cd /d %~dp0 && python -m app.main"

echo [INFO] Attendo backend su http://localhost:8000/health ...
set BACKEND_OK=0
for /L %%i in (1,1,20) do (
    curl -s http://localhost:8000/health >nul 2>&1
    if not errorlevel 1 (
        set BACKEND_OK=1
        goto :backend_ready
    )
    timeout /t 1 /nobreak >nul
)

:backend_ready
if "%BACKEND_OK%"=="0" (
    echo [ERRORE] Backend non raggiungibile dopo 20 secondi
    pause
    exit /b 1
)

echo [2/4] Avvio frontend...
start "Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"
timeout /t 5 /nobreak >nul

echo [3/4] Avvio tunnel ngrok per backend API...
tasklist /FI "IMAGENAME eq ngrok.exe" | find /I "ngrok.exe" >nul
if not errorlevel 1 (
    echo [INFO] Trovato ngrok gia attivo, lo chiudo per evitare conflitti endpoint...
    taskkill /F /IM ngrok.exe >nul 2>&1
    timeout /t 1 /nobreak >nul
)
start "Ngrok Tunnel" cmd /k "ngrok http 8000 --host-header=localhost:8000"
timeout /t 5 /nobreak >nul

echo [4/4] Avvio applicazione desktop...
if exist "%~dp0dist\ai_trading_system.exe" (
    start "" "%~dp0dist\ai_trading_system.exe"
) else (
    echo [AVVISO] Eseguibile desktop non trovato: %~dp0dist\ai_trading_system.exe
)

echo.
echo ==========================================
echo  Applicazione avviata con Ngrok!
echo ==========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:5173
echo Ngrok: https:// (vedi finestra ngrok)
echo.
echo NOTE: Copia l'URL ngrok per accesso pubblico
echo.
echo Premi un tasto per chiudere questa finestra
pause >nul
