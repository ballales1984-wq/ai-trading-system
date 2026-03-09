@echo off
chcp 65001 >nul
echo ==========================================
echo  AVVIA FRONTEND + BACKEND
echo ==========================================
echo.

echo [1/3] Installazione dipendenze frontend (se necessario)...
cd /d c:\ai-trading-system\frontend
if not exist node_modules (
    npm install
)

echo.
echo [2/3] Avvio Backend su porta 8000...
start "Backend API" cmd /k "cd /d c:\ai-trading-system && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 5 /nobreak >nul

echo [3/3] Avvio Frontend su porta 5173...
start "Frontend React" cmd /k "cd /d c:\ai-trading-system\frontend && npm run dev"

echo.
echo ==========================================
echo  Fatto!
echo  - Backend: http://localhost:8000
echo  - Frontend: http://localhost:5173
echo ==========================================
echo.
echo Premi un tasto per uscire...
pause >nul

