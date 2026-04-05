"""
Pytest Configuration and Fixtures
===================================
Test fixtures and configuration for AI Trading System tests.

Author: AI Trading System
"""

import pytest
from fastapi.testclient import TestClient
from typing import Generator


@pytest.fixture
def client() -> TestClient:
    """
    Fixture that provides a FastAPI test client.

    Returns:
        TestClient: FastAPI test client instance
    """
    from app.main import app

    return TestClient(app)


@pytest.fixture
def auth_token(client: TestClient) -> str:
    """
    Fixture that provides an authentication token.

    Returns:
        str: JWT access token
    """
    response = client.post("/api/v1/auth/login", json={"email": "admin", "password": "admin123"})

    if response.status_code == 200:
        data = response.json()
        return data.get("access_token", "")

    return ""


@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    """
    Fixture that provides authentication headers.

    Args:
        auth_token: JWT token from auth_token fixture

    Returns:
        dict: Headers with authorization
    """
    if auth_token:
        return {"Authorization": f"Bearer {auth_token}"}
    return {}


@pytest.fixture
def mock_market_data():
    """
    Fixture that provides mock market data.

    Returns:
        dict: Mock market data
    """
    return {
        "BTC/USDT": {
            "price": 45000.00,
            "volume": 1000000000,
            "change_24h": 2.5,
            "high_24h": 46000.00,
            "low_24h": 44000.00,
        },
        "ETH/USDT": {
            "price": 2500.00,
            "volume": 500000000,
            "change_24h": 1.8,
            "high_24h": 2600.00,
            "low_24h": 2400.00,
        },
    }


@pytest.fixture
def mock_portfolio():
    """
    Fixture that provides mock portfolio data.

    Returns:
        dict: Mock portfolio data
    """
    return {
        "total_value": 100000.00,
        "positions": [
            {
                "symbol": "BTC/USDT",
                "quantity": 1.0,
                "avg_price": 42000.00,
                "current_price": 45000.00,
                "pnl": 3000.00,
                "pnl_pct": 7.14,
            },
            {
                "symbol": "ETH/USDT",
                "quantity": 10.0,
                "avg_price": 2400.00,
                "current_price": 2500.00,
                "pnl": 1000.00,
                "pnl_pct": 4.17,
            },
        ],
        "cash": 50000.00,
    }


@pytest.fixture
def mock_order():
    """
    Fixture that provides mock order data.

    Returns:
        dict: Mock order data
    """
    return {
        "symbol": "BTC/USDT",
        "side": "buy",
        "order_type": "limit",
        "quantity": 0.1,
        "price": 45000.00,
        "status": "pending",
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "security: mark test as security-related")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
