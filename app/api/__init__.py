"""
API Routes Module
==============
"""

from app.api.routes.health import router as health # type: ignore
from app.api.routes.orders import router as orders # type: ignore
from app.api.routes.portfolio import router as portfolio # type: ignore
from app.api.routes.strategy import router as strategy # type: ignore
from app.api.routes.market import router as market # type: ignore

__all__ = ["health", "orders", "portfolio", "strategy", "market"]

