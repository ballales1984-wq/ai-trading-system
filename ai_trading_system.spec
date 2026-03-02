# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pandas', 'numpy', 'scipy', 'sklearn', 'dash', 'plotly', 'ccxt', 'sqlalchemy', 'requests', 'aiohttp', 'websockets', 'fastapi', 'uvicorn', 'pydantic', 'config', 'data_collector', 'technical_analysis', 'sentiment_news', 'decision_engine', 'auto_trader', 'trading_simulator', 'live_multi_asset', 'ml_predictor'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter', 'pytest'],
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
    name='ai_trading_system',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
