"""
API Routes Package
=================
FastAPI route modules for the Hedge Fund Trading System.
"""

from app.api.routes.health import router as health # type: ignore
from app.api.routes.orders import router as orders # type: ignore
from app.api.routes.portfolio import router as portfolio # type: ignore
from app.api.routes.strategy import router as strategy # type: ignore
from app.api.routes.market import router as market # type: ignore
from app.api.routes.waitlist import router as waitlist # type: ignore
from app.api.routes.cache import router as cache # type: ignore
from app.api.routes.news import router as news # type: ignore
from app.api.routes.auth import router as auth # type: ignore
from app.api.routes.risk import router as risk # type: ignore
from app.api.routes.ws import router as ws # type: ignore
from app.api.routes.agents import router as agents # type: ignore

__all__ = ["health", "orders", "portfolio", "strategy", "market", "waitlist", "cache", "news", "auth", "risk", "ws", "agents"]
