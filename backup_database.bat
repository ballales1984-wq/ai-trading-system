@echo off
REM ============================================
REM AI Trading System - Database Backup
REM Crea backup del database TimescaleDB
REM ============================================

title Backup Database

echo.
echo  ================================================
echo   BACKUP DATABASE - AI Trading System
echo  ================================================
echo.

cd /d %~dp0

REM Crea cartella backup se non esiste
if not exist backups mkdir backups

REM Genera nome file con data
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set BACKUP_FILE=backups\trading_db_%datetime:~0,8%_%datetime:~8,6%.sql

echo Creando backup: %BACKUP_FILE%
echo.

docker exec postgres_stable pg_dump -U user trading_db > %BACKUP_FILE%

if errorlevel 1 (
    echo [ERRORE] Backup fallito. Il container postgres e' in esecuzione?
    pause
    exit /b 1
)

echo.
echo  ================================================
echo   BACKUP COMPLETATO!
echo   File: %BACKUP_FILE%
echo  ================================================
echo.
pause
