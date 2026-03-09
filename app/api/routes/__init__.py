"""
API Routes Package
=================
FastAPI route modules for the Hedge Fund Trading System.
"""

from app.api.routes.health import router as health
from app.api.routes.orders import router as orders
from app.api.routes.portfolio import router as portfolio
from app.api.routes.strategy import router as strategy
from app.api.routes.market import router as market
from app.api.routes.waitlist import router as waitlist
from app.api.routes.cache import router as cache
from app.api.routes.news import router as news
from app.api.routes.auth import router as auth
from app.api.routes.risk import router as risk

__all__ = ["health", "orders", "portfolio", "strategy", "market", "waitlist", "cache", "news", "auth", "risk"]
