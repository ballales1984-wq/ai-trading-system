#!/usr/bin/env python3
"""
Build script for AI Trading System executable.
Creates a standalone Windows executable using PyInstaller.
"""
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("  AI Trading System - Build Standalone Executable")
    print("=" * 60)
    print()
    
    # Check Python
    print("[INFO] Checking Python installation...")
    result = subprocess.run(["python", "--version"], capture_output=True, text=True)
    print(f"[INFO] {result.stdout.strip()}")
    
    # Check PyInstaller
    print("[INFO] Checking PyInstaller...")
    result = subprocess.run(["pyinstaller", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[INFO] Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Create version info
    print("[INFO] Creating version info...")
    version_info = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(2, 0, 0, 0),
    prodvers=(2, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'AI Trading System'),
           StringStruct(u'FileDescription', u'AI Trading System - Hedge Fund Trading Platform'),
           StringStruct(u'FileVersion', u'2.0.0'),
           StringStruct(u'InternalName', u'ai_trading_system'),
           StringStruct(u'LegalCopyright', u'Copyright (c) 2024'),
           StringStruct(u'OriginalFilename', u'ai_trading_system.exe'),
           StringStruct(u'ProductName', u'AI Trading System'),
           StringStruct(u'ProductVersion', u'2.0.0')]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    with open("version_info.txt", "w") as f:
        f.write(version_info)
    print("[INFO] Version info created.")
    
    # Build executable
    print("[INFO] Building executable...")
    print()
    
    # Check if spec file exists
    if os.path.exists("ai_trading_system.spec"):
        print("[INFO] Using ai_trading_system.spec...")
        subprocess.run(["pyinstaller", "ai_trading_system.spec", "--clean", "--noconfirm"], check=True)
    else:
        print("[INFO] Using main.py...")
        subprocess.run([
            "pyinstaller",
            "--name", "ai_trading_system",
            "--onefile",
            "--windowed",
            "--icon", "icon.ico",
            "--add-data", "config.py;.",
            "--add-data", "data;data",
            "--hidden-import", "pandas",
            "--hidden-import", "numpy",
            "--hidden-import", "sklearn",
            "--hidden-import", "plotly",
            "--hidden-import", "dash",
            "--collect-all", "plotly",
            "--collect-all", "dash",
            "main.py",
            "--clean",
            "--noconfirm"
        ], check=True)
    
    print()
    print("[INFO] Build complete!")
    print("[INFO] Executable location: dist/ai_trading_system.exe")

if __name__ == "__main__":
    main()
