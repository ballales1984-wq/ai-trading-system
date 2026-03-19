"""
Connectors Module
==============
Exchange connectors for order execution.
"""

from .binance_connector import BinanceConnector
from .paper_connector import PaperConnector

# Import IBConnector with fallback for Python 3.14 compatibility
# (ib_insync has issues with event loop in Python 3.14)
try:
    from .ib_connector import IBConnector, create_ib_connector
    IB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # ib_insync may not be available or have compatibility issues
    IBConnector = None
    create_ib_connector = None
    IB_AVAILABLE = False

__all__ = [
    "BinanceConnector",
    "PaperConnector",
    "IBConnector",
    "create_ib_connector",
    "IB_AVAILABLE",
]
