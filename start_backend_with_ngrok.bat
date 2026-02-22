@echo off
chcp 65001 >nul
echo ==========================================
echo  AVVIA BACKEND LOCALE + NGROK
echo ==========================================
echo.

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato! Installa Python 3.11+
    pause
    exit /b 1
)

:: Verifica ngrok
ngrok version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] ngrok non trovato!
    echo Scarica da: https://ngrok.com/download
    echo Estrai in C:\ngrok e aggiungi al PATH
    pause
    exit /b 1
)

echo [1/4] Installazione dipendenze...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERRORE] Installazione dipendenze fallita
    pause
    exit /b 1
)
echo [OK] Dipendenze installate

echo.
echo [2/4] Avvio backend FastAPI su porta 8000...
start "Backend API" cmd /k "cd /d %~dp0 && python -m app.main"
timeout /t 3 /nobreak >nul

echo [3/4] Verifica backend...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [AVVISO] Backend non risponde ancora, attendi...
    timeout /t 3 /nobreak >nul
)

echo [4/4] Avvio ngrok...
echo.
echo ==========================================
echo  CONFIGURA QUESTO URL SU VERCEL:
echo ==========================================
echo.
start "ngrok" cmd /k "ngrok http 8000"
timeout /t 2 /nobreak >nul

echo.
echo ==========================================
echo  ISTRUZIONI:
echo ==========================================
echo 1. Copia l'URL HTTPS da ngrok (es: https://abc123.ngrok.io)
echo 2. Vai su vercel.com → Settings → Environment Variables
echo 3. Aggiungi: VITE_API_BASE_URL=https://tuo-url.ngrok.io/api/v1
echo 4. Redeploy il frontend
echo.
echo 5. Apri il frontend su Vercel e verifica la connessione!
echo.
echo [Premi un tasto per chiudere questa finestra]
echo (I terminali del backend e ngrok rimarranno aperti)
pause >nul
