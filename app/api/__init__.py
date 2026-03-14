"""
API Routes Module
==============
"""

from .routes.health import router as health
from .routes.orders import router as orders
from .routes.portfolio import router as portfolio
from .routes.strategy import router as strategy
from .routes.market import router as market

__all__ = ["health", "orders", "portfolio", "strategy", "market"]

