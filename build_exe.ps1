# ============================================================
# AI Trading System - Build Standalone Executable (PowerShell)
# ============================================================
# This script creates a standalone Windows executable using PyInstaller
#
# Requirements:
#   - Python 3.10+ installed
#   - pip install pyinstaller
#   - All dependencies from requirements.txt installed
#
# Usage:
#   .\build_exe.ps1              # Build with default settings
#   .\build_exe.ps1 -Clean       # Clean build artifacts first
#   .\build_exe.ps1 -DirMode     # Build as directory (faster startup)
#   .\build_exe.ps1 -Lite        # Lite build without ML libraries
# ============================================================

param(
    [switch]$Clean,
    [switch]$DirMode,
    [switch]$Lite,
    [string]$OutputPath = "dist"
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AI Trading System - Build Standalone Executable" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version 2>&1
Write-Host "[INFO] Using $pythonVersion" -ForegroundColor Green

# Check PyInstaller installation
$pyinstaller = Get-Command pyinstaller -ErrorAction SilentlyContinue
if (-not $pyinstaller) {
    Write-Host "[INFO] PyInstaller not found. Installing..." -ForegroundColor Yellow
    pip install pyinstaller
}

# Clean build artifacts if requested
if ($Clean) {
    Write-Host "[INFO] Cleaning build artifacts..." -ForegroundColor Yellow
    @("build", "dist", "__pycache__") | ForEach-Object {
        if (Test-Path $_) { 
            Remove-Item -Recurse -Force $_ 
        }
    }
    Get-ChildItem -Filter "*.pyc" | Remove-Item -Force
    Write-Host "[INFO] Clean complete." -ForegroundColor Green
    Write-Host ""
}

# Create output directory
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath | Out-Null
}

# Define hidden imports
$hiddenImports = @(
    # Core
    "pandas", "numpy", "scipy", "scipy.stats", "scipy.optimize",
    # Web/API
    "requests", "aiohttp", "websockets", "flask", "flask_cors",
    "dash", "dash.dcc", "dash.html", "dash.dash_table",
    "plotly", "plotly.graph_objs", "plotly.express",
    # Database
    "sqlalchemy", "sqlalchemy.dialects.sqlite", "sqlalchemy.dialects.postgresql",
    # Async
    "asyncio", "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    # FastAPI
    "fastapi", "pydantic", "pydantic_settings",
    # Trading
    "ccxt",
    # Project modules
    "config", "data_collector", "technical_analysis", "sentiment_news",
    "decision_engine", "auto_trader", "trading_simulator", "live_multi_asset", "ml_predictor",
    # App modules
    "app", "app.main", "app.core", "app.core.config", "app.core.logging", "app.core.security",
    "app.database", "app.database.models", "app.database.repository",
    "app.execution", "app.execution.broker_connector", "app.execution.execution_engine",
    "app.portfolio", "app.portfolio.performance", "app.portfolio.optimization",
    "app.risk", "app.risk.risk_engine", "app.risk.hardened_risk_engine",
    "app.strategies", "app.strategies.base_strategy",
    "app.market_data", "app.market_data.data_feed", "app.market_data.websocket_stream",
    # Src modules
    "src", "src.core", "src.core.event_bus", "src.core.state_manager", "src.core.engine",
    "src.external", "src.production", "src.production.broker_interface"
)

# Add ML imports if not lite build
if (-not $Lite) {
    $hiddenImports += @(
        "sklearn", "sklearn.ensemble", "sklearn.linear_model",
        "sklearn.preprocessing", "sklearn.model_selection", "sklearn.metrics",
        "xgboost", "lightgbm", "joblib"
    )
}

# Define excludes
$excludes = @("tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "pylint", "black", "flake8")

# Build PyInstaller command
$pyinstallerArgs = @(
    "--clean",
    "--noconfirm",
    "--name", "ai_trading_system",
    "--console"
)

# Add mode (onefile or onedir)
if ($DirMode) {
    $pyinstallerArgs += "--onedir"
    Write-Host "[INFO] Building in directory mode (faster startup)" -ForegroundColor Yellow
} else {
    $pyinstallerArgs += "--onefile"
    Write-Host "[INFO] Building in single-file mode (slower startup, easier distribution)" -ForegroundColor Yellow
}

# Add data files
if (Test-Path "data") {
    $pyinstallerArgs += @("--add-data", "data;data")
}
if (Test-Path ".env") {
    $pyinstallerArgs += @("--add-data", ".env;.env")
}

# Add hidden imports
foreach ($imp in $hiddenImports) {
    $pyinstallerArgs += @("--hidden-import", $imp)
}

# Add excludes
foreach ($exc in $excludes) {
    $pyinstallerArgs += @("--exclude-module", $exc)
}

# Add entry point
$pyinstallerArgs += "main.py"

# Build the executable
Write-Host ""
Write-Host "[INFO] Building executable..." -ForegroundColor Yellow
Write-Host ""

& pyinstaller $pyinstallerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    Write-Host "Check the error messages above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Show output location
if ($DirMode) {
    $exePath = Join-Path $OutputPath "ai_trading_system\ai_trading_system.exe"
    Write-Host "Executable: $exePath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To run: cd $OutputPath\ai_trading_system && .\ai_trading_system.exe" -ForegroundColor White
} else {
    $exePath = Join-Path $OutputPath "ai_trading_system.exe"
    Write-Host "Executable: $exePath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To run: .\$exePath" -ForegroundColor White
}

Write-Host ""
Write-Host "Usage examples:" -ForegroundColor White
Write-Host "  ai_trading_system.exe --mode menu" -ForegroundColor Gray
Write-Host "  ai_trading_system.exe --mode dashboard" -ForegroundColor Gray
Write-Host "  ai_trading_system.exe --mode signals" -ForegroundColor Gray
Write-Host "  ai_trading_system.exe --mode auto" -ForegroundColor Gray
Write-Host "  ai_trading_system.exe --help" -ForegroundColor Gray
Write-Host ""

# Show file size
if (Test-Path $exePath) {
    $fileSize = (Get-Item $exePath).Length / 1MB
    Write-Host "File size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[INFO] Done!" -ForegroundColor Green
