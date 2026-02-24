param(
    [switch]$Clean,
    [switch]$OneFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " AI Trading System - First Desktop EXE Build" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python not found in PATH."
}

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "[INFO] Installing PyInstaller..." -ForegroundColor Yellow
    python -m pip install pyinstaller
}

if ($Clean) {
    Write-Host "[INFO] Cleaning build artifacts..." -ForegroundColor Yellow
    foreach ($path in @("build", "dist")) {
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path
        }
    }
}

# Ensure frontend build exists (served by app.main)
if (-not (Test-Path "frontend/dist/index.html")) {
    throw "Missing frontend build at frontend/dist/index.html. Build frontend first (npm run build)."
}

$modeFlag = if ($OneFile) { "--onefile" } else { "--onedir" }
Write-Host "[INFO] Build mode: $modeFlag" -ForegroundColor Yellow

$args = @(
    "--noconfirm",
    "--clean",
    $modeFlag,
    "--name", "ai_trading_desktop",
    "--console",
    "--paths", ".",
    "--add-data", "frontend/dist;frontend/dist",
    "--add-data", "landing;landing",
    "--add-data", "data;data",
    "--hidden-import", "app.main",
    "--hidden-import", "app.api.routes.health",
    "--hidden-import", "app.api.routes.orders",
    "--hidden-import", "app.api.routes.portfolio",
    "--hidden-import", "app.api.routes.strategy",
    "--hidden-import", "app.api.routes.risk",
    "--hidden-import", "app.api.routes.market",
    "--hidden-import", "app.api.routes.waitlist",
    "--hidden-import", "app.api.routes.cache",
    "--hidden-import", "uvicorn",
    "--hidden-import", "fastapi",
    "--hidden-import", "pydantic",
    "scripts/desktop_launcher.py"
)

Write-Host "[INFO] Running PyInstaller..." -ForegroundColor Yellow
& pyinstaller @args

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$exePath = if ($OneFile) {
    "dist/ai_trading_desktop.exe"
} else {
    "dist/ai_trading_desktop/ai_trading_desktop.exe"
}

Write-Host ""
Write-Host "[OK] Build completed." -ForegroundColor Green
Write-Host "[OK] Executable: $exePath" -ForegroundColor Green
Write-Host ""
Write-Host "Run example:" -ForegroundColor White
Write-Host "  $exePath --host 127.0.0.1 --port 8000 --path /dashboard" -ForegroundColor Gray
Write-Host ""
