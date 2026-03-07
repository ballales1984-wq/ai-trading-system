"""Extended tests for auth API routes"""
import pytest

class TestAuthRoutesExtended:
    def test_auth_router_exists(self):
        from app.api.routes import auth
        assert auth is not None
    
    def test_auth_has_routes(self):
        from app.api.routes import auth
        routes = [r.path for r in auth.routes]
        assert len(routes) > 0
    
    def test_auth_login_endpoint(self):
        from app.api.routes import auth
        paths = [r.path for r in auth.routes]
        assert any("login" in p.lower() for p in paths)
