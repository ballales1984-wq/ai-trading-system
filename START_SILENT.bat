@echo off
:: AI Trading System - Silent Startup
:: Avvio silenzioso senza finestre CMD visibili
:: Utilizza VBScript per nascondere le finestre

echo Starting AI Trading System in silent mode...

:: Crea script VBS helper per nascondere finestre
echo Set WshShell = CreateObject("WScript.Shell") > temp_hidden.vbs
echo WshShell.Run "python main_auto_trader.py --mode live --dry-run --interval 30 --assets BTC/USDT ETH/USDT SOL/USDT", 0, False >> temp_hidden.vbs

:: Avvio AutoTrader in background nascosto
cscript //Nologo temp_hidden.vbs

:: Pulizia file temporaneo
del temp_hidden.vbs

echo.
echo AI Trading System started in silent mode.
echo AutoTrader is running in background.
echo Portfolio: 100000.00 USDT (Dry-Run Mode)
echo.

:: Avvio completato - esci silenziosamente
exit /b 0
