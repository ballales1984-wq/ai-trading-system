$ErrorActionPreference = "SilentlyContinue"

Write-Host "Stopping ngrok..."
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -like "ngrok*" -or $_.CommandLine -match "ngrok.exe http" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Write-Host "Stopping uvicorn app.main:app..."
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "uvicorn.+app.main:app" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Write-Host "Done."
