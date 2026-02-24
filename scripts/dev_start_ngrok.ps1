param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$Region = "eu",
    [string]$AppDir = "c:\ai-trading-system"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Stopping old uvicorn/ngrok processes..."
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "uvicorn.+app.main:app" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -like "ngrok*" -or $_.CommandLine -match "ngrok.exe http" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Start-Sleep -Seconds 1

Write-Host "[2/5] Starting FastAPI backend..."
Start-Process -FilePath "python" -ArgumentList @(
    "-m", "uvicorn", "app.main:app",
    "--app-dir", $AppDir,
    "--host", $ApiHost,
    "--port", "$Port"
) -WorkingDirectory $AppDir | Out-Null

Write-Host "[3/5] Waiting for backend health..."
$healthy = $false
for ($i = 0; $i -lt 120; $i++) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing "http://$ApiHost`:$Port/health" -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $healthy) {
    throw "Backend did not become healthy on http://$ApiHost`:$Port/health"
}

Write-Host "[4/5] Starting ngrok tunnel..."
Start-Process -FilePath "ngrok" -ArgumentList @("http", "$Port", "--region", $Region) | Out-Null

Write-Host "[5/5] Reading tunnel URL..."
Start-Sleep -Seconds 2
$ngrokUrl = $null
for ($i = 0; $i -lt 20; $i++) {
    try {
        $tunnels = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
        $httpsTunnel = $tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1
        if ($httpsTunnel) {
            $ngrokUrl = $httpsTunnel.public_url
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $ngrokUrl) {
    throw "ngrok URL not available at http://127.0.0.1:4040/api/tunnels"
}

Write-Host ""
Write-Host "Backend: http://$ApiHost`:$Port"
Write-Host "Tunnel:  $ngrokUrl"
Write-Host ""
Write-Host "Quick check:"
try {
    $apiCheck = Invoke-WebRequest -UseBasicParsing "$ngrokUrl/api/v1/orders/emergency/status" -TimeoutSec 5
    Write-Host "GET /api/v1/orders/emergency/status => $($apiCheck.StatusCode)"
} catch {
    Write-Host "GET /api/v1/orders/emergency/status => FAILED ($($_.Exception.Message))"
}
Write-Host ""
Write-Host "Reminder: update vercel.json rewrite if ngrok URL changed."
