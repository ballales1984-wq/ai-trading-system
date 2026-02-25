"""
Build script per creare l'EXE dell'applicazione desktop.
Usa PyInstaller + Tkinter (incluso in Python, nessuna dipendenza esterna!)
"""

import os
import sys
import subprocess
from pathlib import Path

def build_exe():
    print("=" * 60)
    print(" AI TRADING SYSTEM - BUILD EXE")
    print("=" * 60)
    print()
    
    # Verifica PyInstaller
    try:
        import PyInstaller
        print("[OK] PyInstaller")
    except ImportError:
        print("[..] Installazione PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[OK] PyInstaller installato")
    
    print("[OK] Tkinter (incluso in Python)")
    print()
    print("Creazione EXE in corso...")
    print()
    
    # Configurazione PyInstaller - USA TKINTER VERSION
    main_script = Path(__file__).parent / "main_tkinter.py"
    project_root = Path(__file__).parent.parent
    
    # Opzioni PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "AI_Trading_System_Pro",
        "--onefile",
        "--windowed",
        "--icon", "NONE",
        "--add-data", f"{project_root / 'app'};app",
        "--add-data", f"{project_root / 'src'};src",
        "--add-data", f"{project_root / 'decision_engine'};decision_engine",
        "--add-data", f"{project_root / 'requirements.txt'};.",
        "--hidden-import", "tkinter",
        "--hidden-import", "fastapi",
        "--hidden-import", "uvicorn",
        "--hidden-import", "pydantic",
        "--hidden-import", "sqlalchemy",
        "--hidden-import", "numpy",
        "--hidden-import", "pandas",
        "--hidden-import", "requests",
        "--hidden-import", "websockets",
        "--hidden-import", "ccxt",
        "--hidden-import", "binance",
        "--clean",
        str(main_script)
    ]
    
    try:
        subprocess.check_call(cmd)
        print()
        print("=" * 60)
        print(" BUILD COMPLETATO!")
        print("=" * 60)
        print()
        print(f"EXE: {Path('dist') / 'AI_Trading_System_Pro.exe'}")
        print()
        print("Per distribuire:")
        print("  1. Prendi l'EXE da dist/")
        print("  2. Comprimi in ZIP")
        print("  3. Invia agli utenti paganti")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"Errore build: {e}")
        return False
    
    return True


if __name__ == "__main__":
    if build_exe():
        print()
        input("Premi INVIO per uscire...")
    else:
        sys.exit(1)
