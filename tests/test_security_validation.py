"""
Tests for security fixes in v2.3.2
"""

import pytest
from pydantic import ValidationError
from app.api.routes.auth import LoginRequest, RegisterRequest, RefreshTokenRequest
from app.api.routes.orders import OrderCreate
from app.api.routes.risk import OrderRiskCheckRequest
from app.api.routes.payments import CheckoutSessionRequest
from app.api.routes.agents import ExecuteRequest
from app.api.routes.orders import EmergencyStopRequest


class TestAuthValidation:
    """Test authentication request validation."""

    def test_login_request_valid(self):
        """Test valid login request."""
        req = LoginRequest(email="test@example.com", password="password123")
        assert req.email == "test@example.com"
        assert req.password == "password123"

    def test_login_request_invalid_email(self):
        """Test login with invalid email."""
        with pytest.raises(ValidationError):
            LoginRequest(email="not-an-email", password="password")

    def test_login_request_empty_password(self):
        """Test login with empty password."""
        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com", password="")

    def test_register_request_valid(self):
        """Test valid register request."""
        req = RegisterRequest(email="test@example.com", password="password123")
        assert req.email == "test@example.com"
        assert req.password == "password123"

    def test_register_request_short_password(self):
        """Test register with short password."""
        with pytest.raises(ValidationError):
            RegisterRequest(email="test@example.com", password="short")

    def test_refresh_token_request_valid(self):
        """Test valid refresh token request."""
        req = RefreshTokenRequest(refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test")
        assert req.refresh_token.startswith("eyJ")


class TestOrderValidation:
    """Test order request validation."""

    def test_order_create_valid(self):
        """Test valid order creation."""
        req = OrderCreate(symbol="BTCUSDT", side="BUY", quantity=0.5)
        assert req.symbol == "BTCUSDT"
        assert req.side == "BUY"

    def test_order_create_invalid_side(self):
        """Test order with invalid side."""
        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="HOLD", quantity=0.5)

    def test_order_create_negative_quantity(self):
        """Test order with negative quantity."""
        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="BUY", quantity=-1)

    def test_order_create_too_large_quantity(self):
        """Test order with excessive quantity."""
        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="BUY", quantity=1e10)

    def test_order_create_invalid_broker(self):
        """Test order with invalid broker."""
        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="BUY", quantity=0.5, broker="invalid")


class TestRiskValidation:
    """Test risk request validation."""

    def test_risk_check_valid(self):
        """Test valid risk check request."""
        req = OrderRiskCheckRequest(symbol="ETHUSDT", side="SELL", quantity=1.0, price=2000.0)
        assert req.symbol == "ETHUSDT"

    def test_risk_check_invalid_side(self):
        """Test risk check with invalid side."""
        with pytest.raises(ValidationError):
            OrderRiskCheckRequest(symbol="BTCUSDT", side="SHORT", quantity=1.0, price=50000)


class TestPaymentValidation:
    """Test payment request validation."""

    def test_checkout_session_valid(self):
        """Test valid checkout session."""
        req = CheckoutSessionRequest(price_id="price_123")
        assert req.price_id == "price_123"

    def test_checkout_session_empty(self):
        """Test empty checkout session."""
        req = CheckoutSessionRequest()
        assert req.price_id is None


class TestAgentValidation:
    """Test agent request validation."""

    def test_execute_valid(self):
        """Test valid execute request."""
        req = ExecuteRequest(symbol="BTCUSDT", action="buy", size=0.1)
        assert req.action == "buy"

    def test_execute_invalid_action(self):
        """Test execute with invalid action."""
        with pytest.raises(ValidationError):
            ExecuteRequest(symbol="BTCUSDT", action="trade", size=0.1)

    def test_execute_negative_size(self):
        """Test execute with negative size."""
        with pytest.raises(ValidationError):
            ExecuteRequest(symbol="BTCUSDT", action="buy", size=-1)


class TestEmergencyStopValidation:
    """Test emergency stop validation."""

    def test_emergency_stop_valid(self):
        """Test valid emergency stop."""
        req = EmergencyStopRequest(reason="Market crash")
        assert req.reason == "Market crash"

    def test_emergency_stop_long_reason(self):
        """Test emergency stop with very long reason."""
        long_reason = "x" * 600
        with pytest.raises(ValidationError):
            EmergencyStopRequest(reason=long_reason)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
