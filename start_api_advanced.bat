@echo off
chcp 65001 >nul
echo ==========================================
echo  AVVIA API AVANZATA (con Login/Auth)
echo ==========================================
echo.

:: Kill existing process on port 8000
echo [1/2] Ferma eventuali processi esistenti su porta 8000...
for /f "tokens=5" %a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %a >nul 2>&1
)

echo [2/2] Avvia API avanzata su porta 8000...
start "API Avanzata" cmd /k "cd /d c:\ai-trading-system && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo.
echo ==========================================
echo  Fatto! L'API avanzata e' ora in esecuzione
echo  Endpoint disponibili:
echo   - Login: POST /v1/auth/login
echo   - Register: POST /v1/auth/register
echo   - Portfolio: /v1/portfolio/summary
echo   - etc.
echo ==========================================
pause
