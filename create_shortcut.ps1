$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("$env:USERPROFILE\Desktop\AI Trading System (Unified).lnk")
$s.TargetPath = "c:\ai-trading-system\start_ai_trading_unified.bat"
$s.WorkingDirectory = "c:\ai-trading-system"
$s.Save()
Write-Host "Collegamento creato sul desktop!"
