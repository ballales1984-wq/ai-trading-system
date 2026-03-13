import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.rate_limiter import default_rate_limiter, RateLimitExceeded
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_rate_limit_exceeded():
    """Test rate limiting returns 429."""
    client_id = "127.0.0.1"
    
    # Exhaust rate limit (mock 60 requests in 1 min)
    with patch.object(default_rate_limiter, 'check_rate_limit') as mock_check:
        # First 59 pass
        mock_check.side_effect = [True] * 59 + [RateLimitExceeded("exceeded", 60)]
        
        # First 59 requests succeed
        for i in range(59):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
        
        # 60th request fails
        response = client.get("/api/v1/health")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["error"]
        assert "Retry-After" in response.headers

def test_rate_limit_stats_endpoint():
    """Test rate limiting stats endpoint."""
    response = client.get("/api/v1/rate-limit/stats")
    data = response.json()
    assert response.status_code == 200
    assert "count" in data
    assert isinstance(data["count"], int)

def test_security_headers_present():
    """Test security headers are included in responses."""
    response = client.get("/api/v1/health")
    headers = response.headers
    
    security_headers = [
        "strict-transport-security",
        "x-frame-options",
        "x-content-type-options",
        "x-xss-protection",
        "referrer-policy",
        "content-security-policy",
        "cross-origin-embedder-policy"
    ]
    
    for header in security_headers:
        assert header.lower() in [h.lower() for h in headers.keys()]
        assert headers.getlist(header)[0]

def test_audit_logging():
    """Test audit events endpoint (basic access)."""
    response = client.get("/api/audit/events?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert "total" in data

@pytest.mark.asyncio
async def test_monitoring_metrics():
    """Test monitoring endpoints."""
    response = client.get("/api/monitoring/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests" in data
    assert "errors" in data
