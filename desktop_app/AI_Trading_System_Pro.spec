# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\ai-trading-system\\desktop_app\\main_tkinter.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\ai-trading-system\\app', 'app'), ('C:\\ai-trading-system\\src', 'src'), ('C:\\ai-trading-system\\decision_engine', 'decision_engine'), ('C:\\ai-trading-system\\requirements.txt', '.')],
    hiddenimports=['tkinter', 'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 'numpy', 'pandas', 'requests', 'websockets', 'ccxt', 'binance'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AI_Trading_System_Pro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='NONE',
)
