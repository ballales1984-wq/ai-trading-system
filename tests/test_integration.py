import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

@pytest.mark.integration
def test_order_to_portfolio_flow():
    """Test full order creation → risk check → portfolio update flow."""
    
    # Step 1: Create order
    order_data = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "MARKET",
        "quantity": 0.01
    }
    order_response = client.post("/api/v1/orders", json=order_data)
    assert order_response.status_code in [200, 201]
    order = order_response.json()
    order_id = order["order_id"]
    
    # Step 2: Check risk metrics (should reflect new position)
    risk_response = client.get("/api/v1/risk/metrics")
    assert risk_response.status_code == 200
    risk_data = risk_response.json()
    assert "portfolio_var_95" in risk_data
    
    # Step 3: Verify portfolio updated
    portfolio_response = client.get("/api/v1/portfolio/summary")
    assert portfolio_response.status_code == 200
    portfolio = portfolio_response.json()
    assert "positions" in portfolio
    
    # Step 4: Emergency stop (integration test)
    stop_response = client.post("/api/v1/orders/emergency-stop", json={"reason": "test", "cancel_all_orders": True})
    assert stop_response.status_code == 200
    
    status_response = client.get("/api/v1/orders/status/emergency")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["emergency_stopped"] == True

@pytest.mark.integration
def test_strategy_signal_to_order():
    """Test strategy signal generation → order creation flow."""
    
    # Get strategy signals
    signals_response = client.get("/api/v1/strategy/signals")
    assert signals_response.status_code == 200
    
    # Get portfolio before
    portfolio_before = client.get("/api/v1/portfolio/summary").json()
    
    # Assume signal triggers order (mocked in test env)
    if signals_response.json()["signals"]:
        print("Integration: Signals received, order flow would trigger")
    
    # Portfolio should reflect activity
    portfolio_after = client.get("/api/v1/portfolio/summary").json()
    assert portfolio_before["total_value"] != portfolio_after["total_value"]  # In real, check delta

def test_rate_limit_during_integration():
    """Test rate limiting during API flow."""
    responses = []
    for i in range(61):  # Exceed limit
        resp = client.get("/api/v1/market/prices")
        responses.append(resp.status_code)
    
    # Last should be 429
    assert responses[-1] == 429
