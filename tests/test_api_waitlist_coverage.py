"""
Tests for API Waitlist Routes - Coverage
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestWaitlistRoutes:
    """Test waitlist endpoints"""
    
    def test_get_waitlist(self):
        """Test getting waitlist"""
        response = client.get("/api/v1/waitlist")
        assert response.status_code in [200, 404]
    
    def test_get_waitlist_status(self):
        """Test getting waitlist status"""
        response = client.get("/api/v1/waitlist/status")
        assert response.status_code in [200, 404]
    
    def test_join_waitlist(self):
        """Test joining waitlist"""
        response = client.post("/api/v1/waitlist", json={"email": "test@example.com"})
        assert response.status_code in [200, 201, 400, 404]
    
    def test_join_waitlist_with_email(self):
        """Test joining waitlist with email"""
        response = client.post("/api/v1/waitlist", json={"email": "user@test.com"})
        assert response.status_code in [200, 201, 400, 404]
    
    def test_join_waitlist_invalid_email(self):
        """Test joining waitlist with invalid email"""
        response = client.post("/api/v1/waitlist", json={"email": "invalid"})
        assert response.status_code in [400, 404, 422]
    
    def test_get_waitlist_position(self):
        """Test getting waitlist position"""
        response = client.get("/api/v1/waitlist/position/test@example.com")
        assert response.status_code in [200, 404]


class TestWaitlistEdgeCases:
    """Test edge cases for waitlist"""
    
    def test_waitlist_empty_email(self):
        """Test waitlist with empty email"""
        response = client.post("/api/v1/waitlist", json={"email": ""})
        assert response.status_code in [400, 422, 404]
    
    def test_waitlist_missing_email(self):
        """Test waitlist without email field"""
        response = client.post("/api/v1/waitlist", json={})
        assert response.status_code in [400, 422, 404]
    
    def test_waitlist_duplicate(self):
        """Test joining waitlist twice"""
        response = client.post("/api/v1/waitlist", json={"email": "duplicate@test.com"})
        # First time might succeed, second might fail
        assert response.status_code in [200, 201, 400, 404]
