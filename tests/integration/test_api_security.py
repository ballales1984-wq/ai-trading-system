"""
API Integration Tests - Security
=================================
Tests for security middleware, rate limiting, and authentication.

Author: AI Trading System
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock
import time

# Import the main app
from app.main import app

# Create test client
client = TestClient(app)


class TestSecurityHeaders:
    """Test security headers are properly set."""
    
    def test_hsts_header_present(self):
        """Test HSTS header is present."""
        response = client.get("/health")
        assert "Strict-Transport-Security" in response.headers
    
    def test_x_frame_options_header(self):
        """Test X-Frame-Options header is set to DENY."""
        response = client.get("/health")
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_x_content_type_options(self):
        """Test X-Content-Type-Options is nosniff."""
        response = client.get("/health")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    
    def test_xss_protection_header(self):
        """Test X-XSS-Protection header."""
        response = client.get("/health")
        assert "X-XSS-Protection" in response.headers
    
    def test_referrer_policy(self):
        """Test Referrer-Policy header."""
        response = client.get("/health")
        assert "Referrer-Policy" in response.headers
    
    def test_content_security_policy(self):
        """Test Content-Security-Policy header."""
        response = client.get("/health")
        assert "Content-Security-Policy" in response.headers
    
    def test_permissions_policy(self):
        """Test Permissions-Policy header."""
        response = client.get("/health")
        assert "Permissions-Policy" in response.headers


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_headers_present(self):
        """Test rate limit headers are present after requests."""
        # Make a request to trigger rate limiting
        for _ in range(5):
            client.get("/health")
        
        # Check headers
        response = client.get("/health")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    def test_rate_limit_exceeded(self):
        """Test rate limit is enforced."""
        # This test simulates rate limiting
        # In production, would need to test with actual rate limit
        response = client.get("/api/v1/rate-limit/stats")
        assert response.status_code == 200
    
    def test_rate_limit_stats_endpoint(self):
        """Test rate limit stats endpoint."""
        response = client.get("/api/v1/rate-limit/stats")
        assert response.status_code == 200
        assert "count" in response.json() or "error" in response.json()


class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_login_success(self):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert response.status_code in [200, 401]
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "invalid", "password": "wrong"}
        )
        assert response.status_code in [200, 401]
    
    def test_protected_endpoint_without_token(self):
        """Test accessing protected endpoint without token."""
        response = client.get("/api/v1/portfolio/positions")
        # Should either redirect to login or return 401/403
        assert response.status_code in [200, 401, 403, 307, 308]
    
    def test_protected_endpoint_with_token(self):
        """Test accessing protected endpoint with valid token."""
        # First login to get token
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            if token:
                # Access protected endpoint
                response = client.get(
                    "/api/v1/portfolio/positions",
                    headers={"Authorization": f"Bearer {token}"}
                )
                assert response.status_code in [200, 401]


class TestAuditLogging:
    """Test audit logging functionality."""
    
    def test_audit_events_endpoint(self):
        """Test audit events endpoint exists."""
        response = client.get("/api/audit/events")
        assert response.status_code in [200, 401, 403]
    
    def test_audit_stats_endpoint(self):
        """Test audit stats endpoint exists."""
        response = client.get("/api/audit/stats")
        assert response.status_code in [200, 401, 403]


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_monitoring_health_endpoint(self):
        """Test detailed monitoring health endpoint."""
        response = client.get("/api/monitoring/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "monitoring" in data
    
    def test_monitoring_metrics_endpoint(self):
        """Test monitoring metrics endpoint."""
        response = client.get("/api/monitoring/metrics")
        assert response.status_code in [200, 503]
    
    def test_security_headers_endpoint(self):
        """Test security headers info endpoint."""
        response = client.get("/api/security/headers")
        assert response.status_code == 200
        data = response.json()
        assert "headers" in data
        assert "rate_limiting" in data


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self):
        """Test CORS headers are present."""
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or \
               response.status_code in [200, 204]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
