"""Extended tests for strategy API routes"""
import pytest

class TestStrategyRoutesExtended:
    def test_strategy_router_exists(self):
        from app.api.routes import strategy
        assert strategy is not None
    
    def test_strategy_has_routes(self):
        from app.api.routes import strategy
        routes = [r.path for r in strategy.routes]
        assert len(routes) > 0
    
    def test_strategy_list_endpoint(self):
        from app.api.routes import strategy
        paths = [r.path for r in strategy.routes]
        assert any("strategy" in p.lower() for p in paths)
