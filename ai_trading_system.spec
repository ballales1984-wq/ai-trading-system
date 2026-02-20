# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for AI Trading System
============================================
Creates a standalone executable with all dependencies.

Usage:
    pyinstaller ai_trading_system.spec

Output:
    dist/ai_trading_system.exe (Windows)
    dist/ai_trading_system (Linux/Mac)
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get project root
project_root = os.path.dirname(os.path.abspath(SPEC))

# Collect all data files from packages
datas = []

# Pandas data files
try:
    datas += collect_data_files('pandas')
except:
    pass

# NumPy data files
try:
    datas += collect_data_files('numpy')
except:
    pass

# Plotly data files (important for offline charts)
try:
    datas += collect_data_files('plotly')
except:
    pass

# Dash data files
try:
    datas += collect_data_files('dash')
except:
    pass

# Scikit-learn data files
try:
    datas += collect_data_files('sklearn')
except:
    pass

# Include project data files
data_dirs = ['data', 'config', '.env']
for data_dir in data_dirs:
    if os.path.exists(os.path.join(project_root, data_dir)):
        datas.append((os.path.join(project_root, data_dir), data_dir))

# Hidden imports - modules that PyInstaller might miss
hiddenimports = [
    # Core
    'pandas',
    'numpy',
    'scipy',
    'scipy.stats',
    'scipy.optimize',
    
    # Web/API
    'requests',
    'aiohttp',
    'websockets',
    'flask',
    'flask_cors',
    'dash',
    'dash.dcc',
    'dash.html',
    'dash.dash_table',
    'plotly',
    'plotly.graph_objs',
    'plotly.express',
    
    # ML
    'sklearn',
    'sklearn.ensemble',
    'sklearn.linear_model',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    'sklearn.metrics',
    'xgboost',
    'lightgbm',
    'joblib',
    
    # Deep Learning (optional - makes exe larger)
    # 'torch',
    # 'transformers',
    
    # Technical Analysis
    'talib',
    
    # Trading
    'ccxt',
    
    # Database
    'sqlalchemy',
    'sqlalchemy.dialects.sqlite',
    'sqlalchemy.dialects.postgresql',
    'alembic',
    
    # Async
    'asyncio',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    
    # FastAPI
    'fastapi',
    'pydantic',
    'pydantic_settings',
    
    # Security
    'jose',
    'passlib',
    'bcrypt',
    
    # News/Sentiment
    'newsapi',
    'textblob',
    'tweepy',
    
    # Project modules
    'config',
    'data_collector',
    'technical_analysis',
    'sentiment_news',
    'decision_engine',
    'auto_trader',
    'trading_simulator',
    'live_multi_asset',
    'ml_predictor',
    
    # App modules
    'app',
    'app.main',
    'app.core',
    'app.core.config',
    'app.core.logging',
    'app.core.security',
    'app.api',
    'app.api.routes',
    'app.database',
    'app.database.models',
    'app.database.repository',
    'app.execution',
    'app.execution.broker_connector',
    'app.execution.execution_engine',
    'app.execution.order_manager',
    'app.portfolio',
    'app.portfolio.performance',
    'app.portfolio.optimization',
    'app.risk',
    'app.risk.risk_engine',
    'app.risk.hardened_risk_engine',
    'app.strategies',
    'app.strategies.base_strategy',
    'app.strategies.momentum',
    'app.strategies.mean_reversion',
    'app.market_data',
    'app.market_data.data_feed',
    'app.market_data.websocket_stream',
    
    # Src modules
    'src',
    'src.core',
    'src.core.event_bus',
    'src.core.state_manager',
    'src.core.engine',
    'src.core.portfolio',
    'src.core.execution',
    'src.core.risk',
    'src.external',
    'src.ml_enhanced',
    'src.production',
    'src.production.broker_interface',
]

# Collect all submodules from key packages
for pkg in ['pandas', 'numpy', 'scipy', 'sklearn', 'dash', 'plotly']:
    try:
        hiddenimports.extend(collect_submodules(pkg))
    except:
        pass

# Analysis configuration
a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'pylint',
        'black',
        'flake8',
        # Exclude unused torch modules (if including torch)
        'torch.distributed',
        'torch.testing',
        'torch.ao',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ archive (Python code)
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# Main executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ai_trading_system',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Use UPX compression
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for GUI mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add .ico file path for custom icon
    version='version_info.txt',  # Optional version info
)

# Optional: Create a directory-based distribution (faster startup)
# Uncomment below for directory mode instead of single exe
"""
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ai_trading_system',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ai_trading_system',
)
"""
