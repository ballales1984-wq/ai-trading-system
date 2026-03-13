import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_security_headers():
    """Test security headers are present on all responses."""
    response = client.get("/health")
    
    assert response.status_code == 200
    
    security_headers = [
        "Strict-Transport-Security",
        "X-Frame-Options",
        "X-Content-Type-Options", 
        "X-XSS-Protection",
        "Referrer-Policy",
        "Permissions-Policy",
        "Content-Security-Policy",
        "Cross-Origin-Embedder-Policy",
        "Cross-Origin-Opener-Policy",
        "Cross-Origin-Resource-Policy"
    ]
    
    for header in security_headers:
        assert header in response.headers
        assert response.headers[header]  # Non-empty

def test_rate_limit_stats():
    """Test rate-limit/stats endpoint."""
    response = client.get("/api/v1/rate-limit/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert "count" in data

if __name__ == "__main__":
    pytest.main(["-v", __file__"])

