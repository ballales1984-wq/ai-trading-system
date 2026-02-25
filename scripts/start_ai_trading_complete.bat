@echo off
chcp 65001 >nul
title 🤖 AI Trading System - Avvio Completo
color 0A

echo ============================================
echo   🤖 AI TRADING SYSTEM - AVVIO COMPLETO
echo ============================================
echo.

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [❌ ERRORE] Python non trovato!
    echo Installa Python 3.11+ da https://python.org
    pause
    exit /b 1
)
echo [✅] Python trovato

:: Verifica ngrok
ngrok version >nul 2>&1
if errorlevel 1 (
    echo [❌ ERRORE] ngrok non trovato!
    echo 1. Scarica da: https://ngrok.com/download
    echo 2. Estrai in C:\ngrok
    echo 3. Aggiungi C:\ngrok al PATH di sistema
    echo.
    echo Oppure esegui: start https://ngrok.com/download
    start https://ngrok.com/download
    pause
    exit /b 1
)
echo [✅] ngrok trovato

:: Verifica file .env
if not exist ".env" (
    echo [⚠️  ATTENZIONE] File .env non trovato!
    echo Creo .env di esempio...
    echo BINANCE_API_KEY=your_key_here > .env
    echo BINANCE_SECRET_KEY=your_secret_here >> .env
    echo NEWS_API_KEY=your_news_key_here >> .env
    echo.
    echo [ℹ️] Modifica il file .env con le tue chiavi reali!
    timeout /t 3 >nul
)

:: Installazione dipendenze (se necessario)
echo.
echo [📦] Verifica dipendenze...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [⏳] Installazione dipendenze in corso...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo [❌] Installazione fallita!
        pause
        exit /b 1
    )
    echo [✅] Dipendenze installate
) else (
    echo [✅] Dipendenze già installate
)

:: Avvio backend in nuova finestra
echo.
echo [🚀] Avvio Backend FastAPI su http://localhost:8000...
start "🚀 Backend AI Trading" cmd /k "cd /d %~dp0.. && echo [🚀] Avvio backend... && python -m app.main"

:: Attesa avvio backend
echo [⏳] Attesa avvio backend (5 secondi)...
timeout /t 5 /nobreak >nul

:: Verifica backend
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [⚠️] Backend non risponde ancora, attendo altri 5 secondi...
    timeout /t 5 /nobreak >nul
)

:: Avvio ngrok in nuova finestra
echo.
echo [🌐] Avvio ngrok tunnel...
start "🌐 ngrok Tunnel" cmd /k "cd /d %~dp0.. && echo [🌐] Avvio ngrok... && echo. && echo ============================================ && echo   COPIA QUESTO URL SU VERCEL: && echo ============================================ && echo. && ngrok http 8000"

:: Avvio browser con dashboard
echo.
echo [🌐] Apertura dashboard...
timeout /t 3 >nul
start https://ai-trading-system-kappa.vercel.app/dashboard

:: Messaggio finale
echo.
echo ============================================
echo   ✅ SISTEMA AVVIATO!
echo ============================================
echo.
echo 📱 Frontend:  https://ai-trading-system-kappa.vercel.app/dashboard
echo 🔧 Backend:   http://localhost:8000
echo 📖 API Docs:  http://localhost:8000/docs
echo.
echo 📝 ISTRUZIONI:
echo 1. Copia l'URL HTTPS da ngrok (es: https://abc123.ngrok.io)
echo 2. Vai su https://vercel.com/dashboard
echo 3. Trova il progetto → Settings → Environment Variables
echo 4. Aggiungi: VITE_API_BASE_URL=https://tuo-url.ngrok.io/api/v1
echo 5. Redeploy il frontend
echo.
echo ⚠️  Per fermare tutto: chiudi le finestre del backend e ngrok
echo.
echo [Premi un tasto per chiudere questa finestra...]
pause >nul
