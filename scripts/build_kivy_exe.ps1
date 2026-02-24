param(
    [switch]$Clean,
    [switch]$OneFile,
    [string]$PythonCmd = "py -3.11"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " AI Trading System - Kivy Desktop EXE Build" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

try {
    $pythonParts = $PythonCmd -split "\s+"
    $pythonExe = $pythonParts[0]
    $pythonArgs = @()
    if ($pythonParts.Length -gt 1) {
        $pythonArgs = $pythonParts[1..($pythonParts.Length - 1)]
    }
    $pyVersion = (& $pythonExe @pythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
} catch {
    throw "Cannot run Python command: $PythonCmd"
}
Write-Host "[INFO] Python version: $pyVersion" -ForegroundColor Yellow

if ($pyVersion -notin @("3.11", "3.12")) {
    throw "Kivy desktop build requires Python 3.11 or 3.12. Current: $pyVersion"
}

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "[INFO] Installing PyInstaller..." -ForegroundColor Yellow
    & $pythonExe @pythonArgs -m pip install pyinstaller
}

Write-Host "[INFO] Installing desktop dependencies (kivy + uvicorn + fastapi)..." -ForegroundColor Yellow
& $pythonExe @pythonArgs -m pip install kivy==2.3.1 fastapi uvicorn pydantic pydantic-settings pyinstaller

if ($Clean) {
    Write-Host "[INFO] Cleaning build artifacts..." -ForegroundColor Yellow
    foreach ($path in @("build", "dist")) {
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path
        }
    }
}

if (-not (Test-Path "frontend/dist/index.html")) {
    throw "Missing frontend build at frontend/dist/index.html. Build frontend first (npm run build)."
}

$modeFlag = if ($OneFile) { "--onefile" } else { "--onedir" }
Write-Host "[INFO] Build mode: $modeFlag" -ForegroundColor Yellow

$workPath = Join-Path "build" ("work_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "[INFO] Work path: $workPath" -ForegroundColor Yellow

# Avoid Kivy file logger permission issues in constrained environments.
$env:KIVY_NO_FILELOG = "1"
$env:KIVY_NO_CONSOLELOG = "1"

$distPath = Join-Path "dist" ("release_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "[INFO] Dist path: $distPath" -ForegroundColor Yellow

$args = @(
    "--noconfirm",
    "--log-level",
    "WARN",
    "--workpath",
    $workPath,
    "--distpath",
    $distPath,
    "ai_trading_kivy_desktop.spec"
)

if ($Clean) {
    $args = @("--clean") + $args
}

Write-Host "[INFO] Running PyInstaller..." -ForegroundColor Yellow
& $pythonExe @pythonArgs -m PyInstaller @args

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$exeCandidates = @(
    (Join-Path $distPath "ai_trading_kivy_desktop.exe"),
    (Join-Path $distPath "ai_trading_kivy_desktop\ai_trading_kivy_desktop.exe")
)
$exePath = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exePath) {
    throw "Build completed but executable not found under expected paths: $($exeCandidates -join ', ')"
}

Write-Host ""
Write-Host "[OK] Kivy build completed." -ForegroundColor Green
Write-Host "[OK] Executable: $exePath" -ForegroundColor Green
Write-Host ""
