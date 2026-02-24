# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['scripts\\desktop_launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=[('frontend/dist', 'frontend/dist'), ('landing', 'landing'), ('data', 'data')],
    hiddenimports=['app.main', 'app.api.routes.health', 'app.api.routes.orders', 'app.api.routes.portfolio', 'app.api.routes.strategy', 'app.api.routes.risk', 'app.api.routes.market', 'app.api.routes.waitlist', 'app.api.routes.cache', 'uvicorn', 'fastapi', 'pydantic'],
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
    [],
    exclude_binaries=True,
    name='ai_trading_desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ai_trading_desktop',
)
