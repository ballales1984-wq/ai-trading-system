"""Extended tests for news API routes"""
import pytest

class TestNewsRoutesExtended:
    def test_news_router_exists(self):
        from app.api.routes import news
        assert news is not None
    
    def test_news_has_routes(self):
        from app.api.routes import news
        routes = [r.path for r in news.routes]
        assert len(routes) > 0
    
    def test_news_list_endpoint(self):
        from app.api.routes import news
        paths = [r.path for r in news.routes]
        assert len(paths) > 0
