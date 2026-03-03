"""
API Routes Module
==============
"""

from app.api.routes.health import router as health
from app.api.routes.orders import router as orders
from app.api.routes.portfolio import router as portfolio
from app.api.routes.strategy import router as strategy
from app.api.routes.market import router as market

__all__ = ["health", "orders", "portfolio", "strategy", "market"]

