# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('data', 'data'), ('.env', '.env')],
    hiddenimports=['pandas', 'numpy', 'scipy', 'scipy.stats', 'scipy.optimize', 'requests', 'aiohttp', 'websockets', 'flask', 'flask_cors', 'dash', 'dash.dcc', 'dash.html', 'dash.dash_table', 'plotly', 'plotly.graph_objs', 'plotly.express', 'sqlalchemy', 'sqlalchemy.dialects.sqlite', 'sqlalchemy.dialects.postgresql', 'asyncio', 'uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'fastapi', 'pydantic', 'pydantic_settings', 'ccxt', 'config', 'data_collector', 'technical_analysis', 'sentiment_news', 'decision_engine', 'auto_trader', 'trading_simulator', 'live_multi_asset', 'ml_predictor', 'app', 'app.main', 'app.core', 'app.core.config', 'app.core.logging', 'app.core.security', 'app.database', 'app.database.models', 'app.database.repository', 'app.execution', 'app.execution.broker_connector', 'app.execution.execution_engine', 'app.portfolio', 'app.portfolio.performance', 'app.portfolio.optimization', 'app.risk', 'app.risk.risk_engine', 'app.risk.hardened_risk_engine', 'app.strategies', 'app.strategies.base_strategy', 'app.market_data', 'app.market_data.data_feed', 'app.market_data.websocket_stream', 'src', 'src.core', 'src.core.event_bus', 'src.core.state_manager', 'src.core.engine', 'src.external', 'src.production', 'src.production.broker_interface', 'sklearn', 'sklearn.ensemble', 'sklearn.linear_model', 'sklearn.preprocessing', 'sklearn.model_selection', 'sklearn.metrics', 'xgboost', 'lightgbm', 'joblib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter', 'notebook', 'pytest', 'pylint', 'black', 'flake8'],
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
