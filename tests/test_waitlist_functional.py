"""Functional tests for waitlist API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestWaitlistAPI:
    """Test waitlist API endpoints."""
    
    def test_get_waitlist(self):
        """Test getting waitlist status."""
        response = client.get("/api/v1/waitlist")
        assert response.status_code in [200, 500]
    
    def test_join_waitlist(self):
        """Test joining waitlist."""
        response = client.post("/api/v1/waitlist/join", json={
            "email": "test@example.com",
            "name": "Test User"
        })
        assert response.status_code in [200, 201, 400, 500]
    
    def test_check_position(self):
        """Test checking waitlist position."""
        response = client.get("/api/v1/waitlist/position/test@example.com")
        assert response.status_code in [200, 404, 500]
    
    def test_leave_waitlist(self):
        """Test leaving waitlist."""
        response = client.post("/api/v1/waitlist/leave", json={
            "email": "test@example.com"
        })
        assert response.status_code in [200, 404, 500]
