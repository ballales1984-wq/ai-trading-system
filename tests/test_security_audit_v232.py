"""
Comprehensive test suite for AI Trading System v2.3.2 Security Fixes
Tests all input validation and security features implemented in the audit.
"""

import pytest
from pydantic import ValidationError, EmailStr
from typing import Optional


# ============================================================================
# AUTH VALIDATION TESTS
# ============================================================================


class TestAuthValidation:
    """Test authentication request validation."""

    def test_login_request_valid(self):
        """Test valid login request."""
        from app.api.routes.auth import LoginRequest

        req = LoginRequest(email="test@example.com", password="password123")
        assert req.email == "test@example.com"
        assert req.password == "password123"

    def test_login_request_invalid_email(self):
        """Test login with invalid email should fail."""
        from app.api.routes.auth import LoginRequest

        with pytest.raises(ValidationError):
            LoginRequest(email="not-an-email", password="password")

    def test_login_request_empty_password(self):
        """Test login with empty password should fail."""
        from app.api.routes.auth import LoginRequest

        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com", password="")

    def test_login_request_too_long_password(self):
        """Test login with password exceeding max length should fail."""
        from app.api.routes.auth import LoginRequest

        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com", password="x" * 201)

    def test_register_request_valid(self):
        """Test valid register request."""
        from app.api.routes.auth import RegisterRequest

        req = RegisterRequest(email="newuser@example.com", password="securepass123")
        assert req.email == "newuser@example.com"
        assert req.password == "securepass123"

    def test_register_request_short_password(self):
        """Test register with password < 8 chars should fail."""
        from app.api.routes.auth import RegisterRequest

        with pytest.raises(ValidationError):
            RegisterRequest(email="test@example.com", password="short")

    def test_register_request_long_password(self):
        """Test register with password > 100 chars should fail."""
        from app.api.routes.auth import RegisterRequest

        with pytest.raises(ValidationError):
            RegisterRequest(email="test@example.com", password="x" * 101)

    def test_refresh_token_request_valid(self):
        """Test valid refresh token request."""
        from app.api.routes.auth import RefreshTokenRequest

        req = RefreshTokenRequest(refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test")
        assert req.refresh_token.startswith("eyJ")

    def test_refresh_token_too_short(self):
        """Test refresh token < 10 chars should fail."""
        from app.api.routes.auth import RefreshTokenRequest

        with pytest.raises(ValidationError):
            RefreshTokenRequest(refresh_token="short")


# ============================================================================
# ORDERS VALIDATION TESTS
# ============================================================================


class TestOrderValidation:
    """Test order request validation."""

    def test_order_create_valid(self):
        """Test valid order creation."""
        from app.api.routes.orders import OrderCreate

        req = OrderCreate(symbol="BTCUSDT", side="BUY", quantity=0.5)
        assert req.symbol == "BTCUSDT"
        assert req.side == "BUY"

    def test_order_create_invalid_side(self):
        """Test order with invalid side should fail."""
        from app.api.routes.orders import OrderCreate

        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="HOLD", quantity=0.5)

    def test_order_create_negative_quantity(self):
        """Test order with negative quantity should fail."""
        from app.api.routes.orders import OrderCreate

        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="BUY", quantity=-1)

    def test_order_create_zero_quantity(self):
        """Test order with zero quantity should fail."""
        from app.api.routes.orders import OrderCreate

        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="BUY", quantity=0)

    def test_order_create_excessive_quantity(self):
        """Test order with quantity > 1e9 should fail."""
        from app.api.routes.orders import OrderCreate

        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="BUY", quantity=1e10)

    def test_order_create_invalid_broker(self):
        """Test order with invalid broker should fail."""
        from app.api.routes.orders import OrderCreate

        with pytest.raises(ValidationError):
            OrderCreate(symbol="BTCUSDT", side="BUY", quantity=0.5, broker="invalid")

    def test_order_create_valid_broker(self):
        """Test order with valid brokers."""
        from app.api.routes.orders import OrderCreate

        for broker in ["binance", "ib", "bybit", "paper"]:
            req = OrderCreate(symbol="BTCUSDT", side="BUY", quantity=0.5, broker=broker)
            assert req.broker == broker


# ============================================================================
# RISK VALIDATION TESTS
# ============================================================================


class TestRiskValidation:
    """Test risk request validation."""

    def test_risk_check_valid(self):
        """Test valid risk check request."""
        from app.api.routes.risk import OrderRiskCheckRequest

        req = OrderRiskCheckRequest(symbol="ETHUSDT", side="SELL", quantity=1.0, price=2000.0)
        assert req.symbol == "ETHUSDT"

    def test_risk_check_invalid_side(self):
        """Test risk check with invalid side should fail."""
        from app.api.routes.risk import OrderRiskCheckRequest

        with pytest.raises(ValidationError):
            OrderRiskCheckRequest(symbol="BTCUSDT", side="SHORT", quantity=1.0, price=50000)

    def test_risk_check_negative_price(self):
        """Test risk check with negative price should fail."""
        from app.api.routes.risk import OrderRiskCheckRequest

        with pytest.raises(ValidationError):
            OrderRiskCheckRequest(symbol="BTCUSDT", side="BUY", quantity=1.0, price=-100)


# ============================================================================
# PAYMENTS VALIDATION TESTS
# ============================================================================


class TestPaymentValidation:
    """Test payment request validation."""

    def test_checkout_session_valid(self):
        """Test valid checkout session."""
        from app.api.routes.payments import CheckoutSessionRequest

        req = CheckoutSessionRequest(price_id="price_123")
        assert req.price_id == "price_123"

    def test_checkout_session_empty(self):
        """Test empty checkout session."""
        from app.api.routes.payments import CheckoutSessionRequest

        req = CheckoutSessionRequest()
        assert req.price_id is None


# ============================================================================
# AGENTS VALIDATION TESTS
# ============================================================================


class TestAgentValidation:
    """Test agent request validation."""

    def test_execute_valid(self):
        """Test valid execute request."""
        from app.api.routes.agents import ExecuteRequest

        req = ExecuteRequest(symbol="BTCUSDT", action="buy", size=0.1)
        assert req.action == "buy"

    def test_execute_invalid_action(self):
        """Test execute with invalid action should fail."""
        from app.api.routes.agents import ExecuteRequest

        with pytest.raises(ValidationError):
            ExecuteRequest(symbol="BTCUSDT", action="trade", size=0.1)

    def test_execute_negative_size(self):
        """Test execute with negative size should fail."""
        from app.api.routes.agents import ExecuteRequest

        with pytest.raises(ValidationError):
            ExecuteRequest(symbol="BTCUSDT", action="buy", size=-1)

    def test_execute_excessive_size(self):
        """Test execute with size > 1e9 should fail."""
        from app.api.routes.agents import ExecuteRequest

        with pytest.raises(ValidationError):
            ExecuteRequest(symbol="BTCUSDT", action="buy", size=1e10)


# ============================================================================
# EMERGENCY STOP VALIDATION TESTS
# ============================================================================


class TestEmergencyStopValidation:
    """Test emergency stop validation."""

    def test_emergency_stop_valid(self):
        """Test valid emergency stop."""
        from app.api.routes.orders import EmergencyStopRequest

        req = EmergencyStopRequest(reason="Market crash")
        assert req.reason == "Market crash"

    def test_emergency_stop_long_reason(self):
        """Test emergency stop with very long reason should fail."""
        from app.api.routes.orders import EmergencyStopRequest

        long_reason = "x" * 600
        with pytest.raises(ValidationError):
            EmergencyStopRequest(reason=long_reason)


# ============================================================================
# SECURITY TESTS
# ============================================================================


class TestSecurityFeatures:
    """Test security features."""

    def test_token_blacklist_exists(self):
        """Test that token blacklist exists in JWT manager."""
        from app.core.security import jwt_manager

        assert hasattr(jwt_manager, "_token_blacklist")
        assert isinstance(jwt_manager._token_blacklist, dict)

    def test_token_blacklist_add(self):
        """Test adding token to blacklist."""
        from app.core.security import JWTManager
        from datetime import timedelta

        manager = JWTManager()

        # Create a test token
        user = manager.create_user("testuser", "testpass", email="test@test.com")
        token = manager.create_access_token(user)

        # Add to blacklist
        result = manager.add_to_blacklist(token)
        assert result is True

        # Check is blacklisted
        assert manager.is_blacklisted(token) is True

    def test_config_debug_default_false(self):
        """Test that debug defaults to False in production."""
        import os

        os.environ.pop("DEBUG", None)

        from app.core.config import settings

        assert settings.debug is False

    def test_production_mode_requires_users(self):
        """Test that production mode requires user env vars."""
        import os

        os.environ["ENVIRONMENT"] = "production"

        # Should not create default users in production
        from app.core import security

        assert security._is_production is True


# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
