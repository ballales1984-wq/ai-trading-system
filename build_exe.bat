@echo off
REM ============================================================
REM AI Trading System - Build Standalone Executable
REM ============================================================
REM This script creates a standalone Windows executable using PyInstaller
REM 
REM Requirements:
REM   - Python 3.10+ installed
REM   - pip install pyinstaller
REM   - All dependencies from requirements.txt installed
REM
REM Usage:
REM   build_exe.bat          - Build with default settings
REM   build_exe.bat clean    - Clean build artifacts first
REM   build_exe.bat dir      - Build as directory (faster startup)
REM ============================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   AI Trading System - Build Standalone Executable
echo ============================================================
echo.

REM Parse arguments
set CLEAN_BUILD=0
set DIR_MODE=0
for %%a in (%*) do (
    if "%%a"=="clean" set CLEAN_BUILD=1
    if "%%a"=="dir" set DIR_MODE=1
)

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    exit /b 1
)

REM Check PyInstaller installation
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Clean build artifacts if requested
if %CLEAN_BUILD%==1 (
    echo [INFO] Cleaning build artifacts...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist __pycache__ rmdir /s /q __pycache__
    del /q *.pyc 2>nul
    echo [INFO] Clean complete.
    echo.
)

REM Create version info file
echo [INFO] Creating version info...
(
echo VSVersionInfo^(
echo   ffi=FixedFileInfo^(
echo     filevers=%(2, 0, 0, 0^),
echo     prodvers=%(2, 0, 0, 0^),
echo     mask=0x3f,
echo     flags=0x0,
echo     OS=0x40004,
echo     fileType=0x1,
echo     subtype=0x0,
echo     date=%(0, 0^)
echo   ^),
echo   kids=[
echo     StringFileInfo^(
echo       [
echo         StringTable^(
echo           u'040904B0',
echo           [StringStruct(u'CompanyName', u'AI Trading System'^),
echo            StringStruct(u'FileDescription', u'AI Trading System - Hedge Fund Trading Platform'^),
echo            StringStruct(u'FileVersion', u'2.0.0'^),
echo            StringStruct(u'InternalName', u'ai_trading_system'^),
echo            StringStruct(u'LegalCopyright', u'Copyright (c) 2024'^),
echo            StringStruct(u'OriginalFilename', u'ai_trading_system.exe'^),
echo            StringStruct(u'ProductName', u'AI Trading System'^),
echo            StringStruct(u'ProductVersion', u'2.0.0'^)]^)
echo       ]^),
echo     VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
echo   ]
echo ^)
) > version_info.txt

REM Build the executable
echo [INFO] Building executable...
echo.

if %DIR_MODE%==1 (
    echo [INFO] Building in directory mode (faster startup^)
    REM Modify spec file for directory mode
    pyinstaller --clean --noconfirm ^
        --name ai_trading_system ^
        --console ^
        --onedir ^
        --add-data "data;data" ^
        --add-data ".env;.env" ^
        --hidden-import=pandas ^
        --hidden-import=numpy ^
        --hidden-import=scipy ^
        --hidden-import=sklearn ^
        --hidden-import=dash ^
        --hidden-import=plotly ^
        --hidden-import=ccxt ^
        --hidden-import=sqlalchemy ^
        --hidden-import=requests ^
        --hidden-import=aiohttp ^
        --hidden-import=websockets ^
        --hidden-import=fastapi ^
        --hidden-import=uvicorn ^
        --hidden-import=pydantic ^
        --hidden-import=config ^
        --hidden-import=data_collector ^
        --hidden-import=technical_analysis ^
        --hidden-import=sentiment_news ^
        --hidden-import=decision_engine ^
        --hidden-import=auto_trader ^
        --hidden-import=trading_simulator ^
        --hidden-import=live_multi_asset ^
        --hidden-import=ml_predictor ^
        --hidden-import=app ^
        --hidden-import=app.main ^
        --hidden-import=app.core ^
        --hidden-import=app.database ^
        --hidden-import=app.execution ^
        --hidden-import=app.portfolio ^
        --hidden-import=app.risk ^
        --hidden-import=app.strategies ^
        --hidden-import=src ^
        --hidden-import=src.core ^
        --hidden-import=src.external ^
        --hidden-import=src.production ^
        --exclude-module=tkinter ^
        --exclude-module=matplotlib ^
        --exclude-module=IPython ^
        --exclude-module=jupyter ^
        --exclude-module=pytest ^
        main.py
) else (
    echo [INFO] Building in single-file mode (slower startup, easier distribution^)
    pyinstaller --clean --noconfirm ^
        --name ai_trading_system ^
        --console ^
        --onefile ^
        --add-data "data;data" ^
        --add-data ".env;.env" ^
        --hidden-import=pandas ^
        --hidden-import=numpy ^
        --hidden-import=scipy ^
        --hidden-import=sklearn ^
        --hidden-import=dash ^
        --hidden-import=plotly ^
        --hidden-import=ccxt ^
        --hidden-import=sqlalchemy ^
        --hidden-import=requests ^
        --hidden-import=aiohttp ^
        --hidden-import=websockets ^
        --hidden-import=fastapi ^
        --hidden-import=uvicorn ^
        --hidden-import=pydantic ^
        --hidden-import=config ^
        --hidden-import=data_collector ^
        --hidden-import=technical_analysis ^
        --hidden-import=sentiment_news ^
        --hidden-import=decision_engine ^
        --hidden-import=auto_trader ^
        --hidden-import=trading_simulator ^
        --hidden-import=live_multi_asset ^
        --hidden-import=ml_predictor ^
        --hidden-import=app ^
        --hidden-import=app.main ^
        --hidden-import=app.core ^
        --hidden-import=app.database ^
        --hidden-import=app.execution ^
        --hidden-import=app.portfolio ^
        --hidden-import=app.risk ^
        --hidden-import=app.strategies ^
        --hidden-import=src ^
        --hidden-import=src.core ^
        --hidden-import=src.external ^
        --hidden-import=src.production ^
        --exclude-module=tkinter ^
        --exclude-module=matplotlib ^
        --exclude-module=IPython ^
        --exclude-module=jupyter ^
        --exclude-module=pytest ^
        main.py
)

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo Check the error messages above.
    exit /b 1
)

echo.
echo ============================================================
echo   Build Complete!
echo ============================================================
echo.
echo Output location: dist\
echo.

if %DIR_MODE%==1 (
    echo Executable: dist\ai_trading_system\ai_trading_system.exe
    echo.
    echo To run: cd dist\ai_trading_system ^&^& ai_trading_system.exe
) else (
    echo Executable: dist\ai_trading_system.exe
    echo.
    echo To run: dist\ai_trading_system.exe
)

echo.
echo Usage examples:
echo   ai_trading_system.exe --mode menu
echo   ai_trading_system.exe --mode dashboard
echo   ai_trading_system.exe --mode signals
echo   ai_trading_system.exe --mode auto
echo   ai_trading_system.exe --help
echo.

REM Clean up
del version_info.txt 2>nul

echo [INFO] Done!
pause
