# AI Trading System - Avvio Frontend Online con ngrok
# Questo avvia solo il frontend React con tunnel ngrok

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  AVVIA FRONTEND ONLINE (ngrok)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Verifica ngrok
$ngrokCheck = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ngrokCheck) {
    Write-Host "[ERRORE] ngrok non trovato!" -ForegroundColor Red
    Write-Host "Scarica da: https://ngrok.com/download" -ForegroundColor Yellow
    Write-Host "Estrai in C:\ngrok e aggiungi al PATH" -ForegroundColor Yellow
    Read-Host "Premi Invio per uscire"
    exit 1
}

# Verifica Node.js
$nodeCheck = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCheck) {
    Write-Host "[ERRORE] Node.js non trovato!" -ForegroundColor Red
    Read-Host "Premi Invio per uscire"
    exit 1
}

Write-Host "[1/3] Installazione dipendenze frontend..." -ForegroundColor Yellow
Set-Location "c:\ai-trading-system\frontend"
npm install

Write-Host ""
Write-Host "[2/3] Avvio frontend React su porta 5173..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location $path
    npm run dev
} -ArgumentList "c:\ai-trading-system\frontend"

Start-Sleep -Seconds 5

Write-Host "[3/3] Avvio ngrok tunnel..." -ForegroundColor Yellow
# Close existing ngrok if running
$ngrokProcess = Get-Process ngrok -ErrorAction SilentlyContinue
if ($ngrokProcess) {
    Write-Host "Chiudo ngrok esistente..." -ForegroundColor Yellow
    Stop-Process -Name ngrok -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Start ngrok in background and get the URL
$ngrokOutput = ""
$ngrokJob = Start-Job -ScriptBlock {
    ngrok http 5173 --log=stdout 2>&1 | ForEach-Object { 
        Write-Output $_ 
        if ($_ -match "https://[a-z0-9]+\.ngrok-free\.app") {
            $_ -match "(https://[a-z0-9]+\.ngrok-free\.app)" | Out-Null
            $matches[1]
        }
    }
} -ArgumentList "c:\ai-trading-system\frontend"

Start-Sleep -Seconds 8

# Try to get ngrok URL from API
try {
    $ngrokApi = Invoke-RestMethod "http://localhost:4040/api/tunnels" -ErrorAction SilentlyContinue
    if ($ngrokApi) {
        $tunnel = $ngrokApi.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1
        if ($tunnel) {
            $ngrokOutput = $tunnel.public_url
        }
    }
} catch {
    # Fallback - wait a bit more
    Start-Sleep -Seconds 5
    try {
        $ngrokApi = Invoke-RestMethod "http://localhost:4040/api/tunnels" -ErrorAction SilentlyContinue
        $tunnel = $ngrokApi.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1
        if ($tunnel) {
            $ngrokOutput = $tunnel.public_url
        }
    } catch {}
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  FATTO!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Frontend accessibile online su:" -ForegroundColor Cyan
Write-Host $ngrokOutput -ForegroundColor Yellow -BackgroundColor Black
Write-Host ""
Write-Host "Link diretto: $ngrokOutput" -ForegroundColor White
Write-Host ""
Write-Host "NOTE:" -ForegroundColor Yellow
Write-Host "- Il frontend e' accessibile pubblicamente" -ForegroundColor Gray
Write-Host "- L'altra interfaccia (dashboard/) e' per lo sviluppo" -ForegroundColor Gray
Write-Host ""
Write-Host "Premi CTRL+C per fermare tutto" -ForegroundColor Gray
Write-Host ""

# Keep script running
while ($true) {
    Start-Sleep -Seconds 10
}
