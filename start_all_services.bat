@echo off
chcp 65001 >nul
echo ==========================================
echo  AI TRADING SYSTEM - AVVIA TUTTO
echo ==========================================
echo.

REM Check if npm is installed
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRORE] Node.js non installato!
    echo Scaricalo da: https://nodejs.org
    pause
    exit /b 1
)

REM Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRORE] Python non installato!
    pause
    exit /b 1
)

echo [1/4] Avvio Backend API su porta 8000...
start "Backend API" cmd /k "cd /d c:\ai-trading-system && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo [2/4] Avvio Frontend React su porta 5173...
cd /d c:\ai-trading-system\frontend
if not exist node_modules (
    echo     Installo dipendenze frontend...
    call npm install
)
start "Frontend React" cmd /k "cd /d c:\ai-trading-system\frontend && npm run dev"

timeout /t 3 /nobreak >nul

echo [3/4] Avvio Dashboard Python su porta 8050...
start "Dashboard Dash" cmd /k "cd /d c:\ai-trading-system\dashboard && python app.py"

timeout /t 3 /nobreak >nul

echo [4/4] Avvio AI Assistant su porta 8501...
start "AI Assistant" cmd /k "cd /d c:\ai-trading-system && streamlit run ai_financial_dashboard.py --server.port 8501"

echo.
echo ==========================================
echo  TUTTO AVVIATO!
echo ==========================================
echo.
echo  Servizi disponibili:
echo  - Backend API:     http://localhost:8000
echo  - Frontend:        http://localhost:5173
echo  - Dashboard Dash:  http://localhost:8050
echo  - AI Assistant:    http://localhost:8501
echo.
echo  Premi un tasto per uscire (i servizi continueranno)...
pause >nul
