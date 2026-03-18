# =============================================================================
# AI Trading System - Create Desktop Shortcut
# =============================================================================
# Questo script crea un collegamento sul Desktop che avvia tutti i servizi

$ErrorActionPreference = "Stop"

# Percorso del progetto
$ProjectRoot = "C:\ai-trading-system"

# Percorso Desktop
$Desktop = [Environment]::GetFolderPath("Desktop")

# Collegamento
$ShortcutPath = Join-Path $Desktop "AI Trading System.lnk"

# Verifica che il file batch esista
$BatchFile = Join-Path $ProjectRoot "start_all_services.bat"
if (-not (Test-Path $BatchFile)) {
    Write-Host "[ERRORE] File batch non trovato: $BatchFile" -ForegroundColor Red
    exit 1
}

# Crea oggetto COM per il collegamento
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)

# Configura il collegamento
$Shortcut.TargetPath = $BatchFile
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Avvia AI Trading System - Tutti i servizi"
$Shortcut.IconLocation = "$env:SystemRoot\system32\shell32.dll,3"

# Salva il collegamento
$Shortcut.Save()

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " COLLEGAMENTO CREATO!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Percorso: $ShortcutPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Il collegamento avviera':" -ForegroundColor Yellow
Write-Host "  - Backend API su porta 8000" -ForegroundColor White
Write-Host "  - Frontend React su porta 5173" -ForegroundColor White
Write-Host "  - Dashboard Dash su porta 8050" -ForegroundColor White
Write-Host "  - AI Assistant su porta 8501" -ForegroundColor White
Write-Host ""
