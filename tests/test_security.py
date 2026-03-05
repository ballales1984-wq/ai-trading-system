"""
Tests for Security Module
========================
"""

import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSecurityPasswordHashing:
    """Test password hashing functions."""
    
    def test_hash_password_function(self):
        """Test hash_password standalone function."""
        from app.core.security import hash_password
        
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password_function(self):
        """Test verify_password standalone function."""
        from app.core.security import hash_password, verify_password
        
        password = "test_password_123"
        hashed = hash_password(password)
        
        # Correct password
        assert verify_password(password, hashed) is True
        
        # Wrong password
        assert verify_password("wrong_password", hashed) is False
    
    def test_different_hashes_same_password(self):
        """Test that same password produces different hashes (salt)."""
        from app.core.security import hash_password
        
        password = "test_password"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        # Different hashes due to salt
        assert hash1 != hash2


class TestSecurityConfig:
    """Test SecurityConfig class."""
    
    def test_security_config_defaults(self):
        """Test default security config."""
        from app.core.security import SecurityConfig
        
        config = SecurityConfig()
        
        assert config.secret_key is not None
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
    
    def test_security_config_from_env(self):
        """Test creating config from environment."""
        from app.core.security import SecurityConfig
        
        config = SecurityConfig.from_env()
        
        assert config is not None


class TestJWTManager:
    """Test JWTManager class."""
    
    def test_jwt_manager_creation(self):
        """Test JWTManager initialization."""
        from app.core.security import JWTManager
        
        manager = JWTManager()
        
        assert manager is not None
    
    def test_jwt_manager_hash_password(self):
        """Test JWTManager hash_password method."""
        from app.core.security import JWTManager
        
        manager = JWTManager()
        
        hashed = manager.hash_password("test_password")
        
        assert hashed is not None
        assert hashed != "test_password"
    
    def test_jwt_manager_verify_password(self):
        """Test JWTManager verify_password method."""
        from app.core.security import JWTManager
        
        manager = JWTManager()
        
        password = "test_password"
        hashed = manager.hash_password(password)
        
        assert manager.verify_password(password, hashed) is True
        assert manager.verify_password("wrong", hashed) is False


class TestUser:
    """Test User class."""
    
    def test_user_creation(self):
        """Test User creation."""
        from app.core.security import User
        
        user = User(
            user_id="test_id",
            username="testuser",
            email="test@example.com",
            role="trader",
            hashed_password="some_hash"
        )
        
        assert user.user_id == "test_id"
        assert user.username == "testuser"
        assert user.role == "trader"
    
    def test_user_to_dict(self):
        """Test User to_dict method."""
        from app.core.security import User, UserRole
        
        user = User(
            user_id="test_id",
            username="testuser",
            email="test@example.com",
            role=UserRole.TRADER,
            hashed_password="some_hash"
        )
        
        user_dict = user.to_dict()
        
        assert user_dict["username"] == "testuser"
        assert "hashed_password" not in user_dict  # Should not include password


class TestUserRole:
    """Test UserRole enum."""
    
    def test_user_role_values(self):
        """Test UserRole enum values."""
        from app.core.security import UserRole
        
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.TRADER.value == "trader"
        assert UserRole.VIEWER.value == "viewer"


class TestTokenPayload:
    """Test TokenPayload class."""
    
    def test_token_payload_creation(self):
        """Test TokenPayload creation."""
        from app.core.security import TokenPayload
        
        payload = TokenPayload(
            sub="user123",
            username="testuser",
            role="trader",
            exp=datetime.now() + timedelta(hours=1)
        )
        
        assert payload.sub == "user123"
        assert payload.username == "testuser"
        assert payload.role == "trader"
    
    def test_token_payload_to_dict(self):
        """Test TokenPayload to_dict method."""
        from app.core.security import TokenPayload
        
        payload = TokenPayload(
            sub="user123",
            username="testuser",
            role="trader",
            exp=datetime.now() + timedelta(hours=1)
        )
        
        payload_dict = payload.to_dict()
        
        assert payload_dict["sub"] == "user123"
        assert payload_dict["username"] == "testuser"
        assert payload_dict["role"] == "trader"


class TestSubscriptionPlan:
    """Test SubscriptionPlan enum."""
    
    def test_subscription_plan_values(self):
        """Test SubscriptionPlan values."""
        from app.core.security import SubscriptionPlan
        
        assert SubscriptionPlan.FREE.value == "free"
        assert SubscriptionPlan.TRIAL.value == "trial"
        assert SubscriptionPlan.LIFETIME.value == "lifetime"


class TestSubscriptionStatus:
    """Test SubscriptionStatus enum."""
    
    def test_subscription_status_values(self):
        """Test SubscriptionStatus values."""
        from app.core.security import SubscriptionStatus
        
        assert SubscriptionStatus.ACTIVE.value == "active"
        assert SubscriptionStatus.TRIALING.value == "trialing"
        assert SubscriptionStatus.CANCELED.value == "canceled"


class TestSubscription:
    """Test Subscription class."""
    
    def test_subscription_creation(self):
        """Test Subscription creation."""
        from app.core.security import Subscription, SubscriptionPlan, SubscriptionStatus
        
        sub = Subscription(
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE
        )
        
        assert sub.plan == SubscriptionPlan.FREE
    
    def test_subscription_is_active(self):
        """Test subscription is_active."""
        from app.core.security import Subscription, SubscriptionStatus
        
        sub = Subscription(
            status=SubscriptionStatus.ACTIVE
        )
        
        assert sub.is_active() is True
    
    def test_subscription_is_not_active(self):
        """Test subscription is not active when canceled."""
        from app.core.security import Subscription, SubscriptionStatus
        
        sub = Subscription(
            status=SubscriptionStatus.CANCELED
        )
        
        assert sub.is_active() is False


class TestJWTManagerTokens:
    """Test JWT token creation and verification."""
    
    def test_create_access_token(self):
        """Test creating access token."""
        from app.core.security import JWTManager, User, UserRole
        
        manager = JWTManager()
        user = User(
            user_id="test_id",
            username="testuser",
            email="test@example.com",
            role=UserRole.TRADER,
            hashed_password="hash"
        )
        
        token = manager.create_access_token(user)
        
        assert token is not None
        assert len(token) > 0
    
    def test_verify_token(self):
        """Test verifying token."""
        from app.core.security import JWTManager, User, UserRole
        
        manager = JWTManager()
        user = User(
            user_id="test_id",
            username="testuser",
            email="test@example.com",
            role=UserRole.TRADER,
            hashed_password="hash"
        )
        
        token = manager.create_access_token(user)
        payload = manager.verify_token(token)
        
        assert payload is not None
        assert payload.sub == "test_id"
    
    def test_verify_invalid_token(self):
        """Test verifying invalid token."""
        from app.core.security import JWTManager
        
        manager = JWTManager()
        
        payload = manager.verify_token("invalid_token")
        
        assert payload is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
