# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['c:\\ai-trading-system\\desktop_app\\main_tkinter.py'],
    pathex=[],
    binaries=[],
    datas=[('c:\\ai-trading-system\\app', 'app'), ('c:\\ai-trading-system\\src', 'src'), ('c:\\ai-trading-system\\decision_engine', 'decision_engine'), ('c:\\ai-trading-system\\requirements.txt', '.')],
    hiddenimports=['tkinter', 'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 'numpy', 'pandas', 'requests', 'websockets', 'ccxt', 'binance'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'matplotlib', 'scipy', 'tensorflow', 'tensorboard', 'IPython', 'jupyter', 'pytest'],
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
