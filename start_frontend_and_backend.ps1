# AI Trading System - Avvio Frontend + Backend
# PowerShell Script

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  AVVIA FRONTEND + BACKEND" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Backend
Write-Host "[1/2] Avvio Backend su porta 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\ai-trading-system; python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" -WindowStyle Normal

Start-Sleep -Seconds 3

# Frontend
Write-Host "[2/2] Avvio Frontend su porta 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\ai-trading-system\frontend; npm run dev" -WindowStyle Normal

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Fatto!" -ForegroundColor Green
Write-Host "  - Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  - Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Green

