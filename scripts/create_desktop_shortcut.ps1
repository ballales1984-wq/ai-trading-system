# Creazione collegamento desktop per AI Trading System
# Esegui: powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1

$WshShell = New-Object -comObject WScript.Shell

# Percorso dello script batch
$scriptPath = Join-Path $PSScriptRoot "start_ai_trading_complete.bat"
$projectPath = Split-Path $PSScriptRoot -Parent

# Percorso Desktop
$desktopPath = [Environment]::GetFolderPath("Desktop")

# Nome del collegamento
$shortcutName = "🤖 AI Trading System.lnk"
$shortcutPath = Join-Path $desktopPath $shortcutName

# Crea collegamento
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $scriptPath
$Shortcut.WorkingDirectory = $projectPath
$Shortcut.IconLocation = "C:\Windows\System32\shell32.dll,14" # Icona computer/ret
$Shortcut.Description = "Avvia AI Trading System con backend e ngrok"
$Shortcut.WindowStyle = 1 # Normale

# Salva collegamento
$Shortcut.Save()

Write-Host "Collegamento creato sul desktop: $shortcutName" -ForegroundColor Green
Write-Host "Percorso: $shortcutPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Per avviare il sistema, fai doppio clic sul collegamento sul desktop!" -ForegroundColor Yellow
