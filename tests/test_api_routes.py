"""
Tests for app/api/routes - API Routes
"""

import pytest


class TestAPIRoutesImports:
    """Test suite for API routes imports."""

    def test_health_router_import(self):
        """Test health router can be imported."""
        from app.api.routes.health import router as health
        
        assert health is not None

    def test_orders_router_import(self):
        """Test orders router can be imported."""
        from app.api.routes.orders import router as orders
        
        assert orders is not None

    def test_portfolio_router_import(self):
        """Test portfolio router can be imported."""
        from app.api.routes.portfolio import router as portfolio
        
        assert portfolio is not None

    def test_strategy_router_import(self):
        """Test strategy router can be imported."""
        from app.api.routes.strategy import router as strategy
        
        assert strategy is not None

    def test_market_router_import(self):
        """Test market router can be imported."""
        from app.api.routes.market import router as market
        
        assert market is not None

    def test_waitlist_router_import(self):
        """Test waitlist router can be imported."""
        from app.api.routes.waitlist import router as waitlist
        
        assert waitlist is not None

    def test_cache_router_import(self):
        """Test cache router can be imported."""
        from app.api.routes.cache import router as cache
        
        assert cache is not None

    def test_news_router_import(self):
        """Test news router can be imported."""
        from app.api.routes.news import router as news
        
        assert news is not None

    def test_auth_router_import(self):
        """Test auth router can be imported."""
        from app.api.routes.auth import router as auth
        
        assert auth is not None


class TestAPIRoutesPackage:
    """Test suite for API routes package."""

    def test_routes_package_import(self):
        """Test routes package can be imported."""
        from app.api import routes
        
        assert routes is not None

    def test_routes_all_export(self):
        """Test __all__ exports."""
        from app.api.routes import __all__
        
        assert "health" in __all__
        assert "orders" in __all__
        assert "portfolio" in __all__
        assert "strategy" in __all__
        assert "market" in __all__
        assert "waitlist" in __all__
        assert "cache" in __all__
        assert "news" in __all__
        assert "auth" in __all__


class TestRouterAttributes:
    """Test suite for router attributes."""

    def test_health_router_has_routes(self):
        """Test health router has routes."""
        from app.api.routes.health import router
        
        # Router should have routes
        assert hasattr(router, 'routes')

    def test_orders_router_has_routes(self):
        """Test orders router has routes."""
        from app.api.routes.orders import router
        
        assert hasattr(router, 'routes')

    def test_portfolio_router_has_routes(self):
        """Test portfolio router has routes."""
        from app.api.routes.portfolio import router
        
        assert hasattr(router, 'routes')

    def test_market_router_has_routes(self):
        """Test market router has routes."""
        from app.api.routes.market import router
        
        assert hasattr(router, 'routes')


class TestAPIRoutesIntegration:
    """Test suite for API routes integration."""

    def test_can_import_all_routes(self):
        """Test all routes can be imported together."""
        from app.api.routes import health, orders, portfolio, strategy, market, waitlist, cache, news, auth
        
        assert health is not None
        assert orders is not None
        assert portfolio is not None
        assert strategy is not None
        assert market is not None
        assert waitlist is not None
        assert cache is not None
        assert news is not None
        assert auth is not None


class TestAPIPackage:
    """Test suite for API package."""

    def test_api_package_import(self):
        """Test API package can be imported."""
        from app import api
        
        assert api is not None

    def test_api_has_routes(self):
        """Test API package has routes."""
        from app.api import routes
        
        assert routes is not None
