"""
API Routes Package
=================
FastAPI route modules for the Hedge Fund Trading System.
"""

from .health import router as health
from .orders import router as orders
from .portfolio import router as portfolio
from .strategy import router as strategy
from .market import router as market
from .waitlist import router as waitlist
from .cache import router as cache
from .news import router as news
from .auth import router as auth
from .risk import router as risk
from .ws import router as ws

__all__ = ["health", "orders", "portfolio", "strategy", "market", "waitlist", "cache", "news", "auth", "risk", "ws"]
