"""
Test Suite for Security Module
================================
Comprehensive tests for JWT Authentication, Password Hashing, and Subscription Management.
"""

import pytest
from datetime import datetime, timedelta
from app.core.security import (
    SubscriptionPlan,
    SubscriptionStatus,
    Subscription,
    UserRole,
    TokenPayload,
    User,
    SecurityConfig,
    JWTManager,
)


class TestSubscriptionPlan:
    """Tests for SubscriptionPlan enum."""
    
    def test_subscription_plan_values(self):
        """Test subscription plan values."""
        assert SubscriptionPlan.TRIAL.value == "trial"
        assert SubscriptionPlan.FREE.value == "free"
        assert SubscriptionPlan.LIFETIME.value == "lifetime"


class TestSubscriptionStatus:
    """Tests for SubscriptionStatus enum."""
    
    def test_subscription_status_values(self):
        """Test subscription status values."""
        assert SubscriptionStatus.ACTIVE.value == "active"
        assert SubscriptionStatus.TRIALING.value == "trialing"
        assert SubscriptionStatus.CANCELED.value == "canceled"
        assert SubscriptionStatus.EXPIRED.value == "expired"


class TestSubscription:
    """Tests for Subscription dataclass."""
    
    def test_subscription_creation(self):
        """Test subscription creation with defaults."""
        sub = Subscription()
        assert sub.plan == SubscriptionPlan.FREE
        assert sub.status == SubscriptionStatus.ACTIVE
        assert sub.stripe_customer_id is None
    
    def test_subscription_is_trial_active(self):
        """Test trial active check."""
        sub = Subscription()
        sub.plan = SubscriptionPlan.TRIAL
        sub.status = SubscriptionStatus.TRIALING
        sub.trial_end_date = datetime.now() + timedelta(days=7)
        assert sub.is_trial_active() is True
    
    def test_subscription_is_trial_expired(self):
        """Test trial expired check."""
        sub = Subscription()
        sub.plan = SubscriptionPlan.TRIAL
        sub.status = SubscriptionStatus.TRIALING
        sub.trial_end_date = datetime.now() - timedelta(days=1)
        assert sub.is_trial_active() is False
    
    def test_subscription_is_active(self):
        """Test subscription is active."""
        sub = Subscription()
        sub.status = SubscriptionStatus.ACTIVE
        assert sub.is_active() is True
    
    def test_subscription_not_active_canceled(self):
        """Test subscription not active when canceled."""
        sub = Subscription()
        sub.status = SubscriptionStatus.CANCELED
        assert sub.is_active() is False
    
    def test_get_days_remaining(self):
        """Test days remaining calculation."""
        sub = Subscription()
        sub.plan = SubscriptionPlan.TRIAL
        sub.status = SubscriptionStatus.TRIALING
        sub.trial_end_date = datetime.now() + timedelta(days=5)
        assert sub.get_days_remaining() == 5
    
    def test_get_days_remaining_expired(self):
        """Test days remaining when expired."""
        sub = Subscription()
        sub.plan = SubscriptionPlan.TRIAL
        sub.status = SubscriptionStatus.TRIALING
        sub.trial_end_date = datetime.now() - timedelta(days=1)
        assert sub.get_days_remaining() == 0


class TestUserRole:
    """Tests for UserRole enum."""
    
    def test_user_role_values(self):
        """Test user role values."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.TRADER.value == "trader"
        assert UserRole.VIEWER.value == "viewer"
        assert UserRole.API_USER.value == "api_user"


class TestTokenPayload:
    """Tests for TokenPayload dataclass."""
    
    def test_token_payload_creation(self):
        """Test token payload creation."""
        exp = datetime.now() + timedelta(hours=1)
        payload = TokenPayload(
            sub="user_1",
            username="testuser",
            role="admin",
            exp=exp,
        )
        assert payload.sub == "user_1"
        assert payload.username == "testuser"
        assert payload.role == "admin"
    
    def test_token_payload_to_dict(self):
        """Test token payload to dictionary conversion."""
        exp = datetime.now() + timedelta(hours=1)
        payload = TokenPayload(
            sub="user_1",
            username="testuser",
            role="admin",
            exp=exp,
        )
        result = payload.to_dict()
        assert result["sub"] == "user_1"
        assert result["username"] == "testuser"
        assert result["role"] == "admin"
        assert "exp" in result
        assert "iat" in result


class TestUser:
    """Tests for User dataclass."""
    
    def test_user_creation(self):
        """Test user creation."""
        user = User(
            user_id="user_1",
            username="testuser",
            hashed_password="hashed_pwd",
            role=UserRole.TRADER,
        )
        assert user.user_id == "user_1"
        assert user.username == "testuser"
        assert user.role == UserRole.TRADER
        assert user.is_active is True
    
    def test_user_to_dict(self):
        """Test user to dictionary conversion."""
        user = User(
            user_id="user_1",
            username="testuser",
            hashed_password="hashed_pwd",
            role=UserRole.TRADER,
            email="test@example.com",
        )
        result = user.to_dict()
        assert result["user_id"] == "user_1"
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["role"] == "trader"
        assert result["is_active"] is True


class TestSecurityConfig:
    """Tests for SecurityConfig class."""
    
    def test_security_config_defaults(self):
        """Test security config with defaults."""
        config = SecurityConfig()
        assert config.secret_key is not None
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7
    
    def test_security_config_custom(self):
        """Test security config with custom values."""
        config = SecurityConfig(
            secret_key="custom-secret",
            algorithm="HS384",
            access_token_expire_minutes=60,
            refresh_token_expire_days=14,
        )
        assert config.secret_key == "custom-secret"
        assert config.algorithm == "HS384"
        assert config.access_token_expire_minutes == 60
        assert config.refresh_token_expire_days == 14


class TestJWTManager:
    """Tests for JWTManager class."""
    
    def test_jwt_manager_creation(self):
        """Test JWT manager creation."""
        manager = JWTManager()
        assert manager is not None
        assert isinstance(manager._users, dict)
    
    def test_jwt_manager_custom_config(self):
        """Test JWT manager with custom config."""
        config = SecurityConfig(secret_key="test-secret")
        manager = JWTManager(config=config)
        assert manager.config.secret_key == "test-secret"
    
    def test_hash_password(self):
        """Test password hashing."""
        manager = JWTManager()
        hashed = manager.hash_password("testpassword")
        assert hashed != "testpassword"
        assert len(hashed) > 0
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        manager = JWTManager()
        hashed = manager.hash_password("testpassword")
        assert manager.verify_password("testpassword", hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        manager = JWTManager()
        hashed = manager.hash_password("testpassword")
        assert manager.verify_password("wrongpassword", hashed) is False
    
    def test_create_user(self):
        """Test user creation."""
        manager = JWTManager()
        user = manager.create_user(
            username="testuser",
            password="testpassword",
            role=UserRole.TRADER,
        )
        assert user.username == "testuser"
        assert user.role == UserRole.TRADER
        assert user.hashed_password is not None
    
    def test_create_user_with_trial(self):
        """Test user creation with trial."""
        manager = JWTManager()
        user = manager.create_user(
            username="testuser",
            password="testpassword",
            trial_days=7,
        )
        assert user.subscription.plan == SubscriptionPlan.TRIAL
        assert user.subscription.status == SubscriptionStatus.TRIALING
        assert user.subscription.is_trial_active() is True
    
    def test_create_user_with_email(self):
        """Test user creation with email."""
        manager = JWTManager()
        user = manager.create_user(
            username="testuser",
            password="testpassword",
            email="test@example.com",
        )
        assert user.email == "test@example.com"
    
    def test_authenticate_success(self):
        """Test successful authentication."""
        manager = JWTManager()
        manager.create_user(
            username="testuser",
            password="testpassword",
        )
        user = manager.authenticate("testuser", "testpassword")
        assert user is not None
        assert user.username == "testuser"
    
    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password."""
        manager = JWTManager()
        manager.create_user(
            username="testuser",
            password="testpassword",
        )
        user = manager.authenticate("testuser", "wrongpassword")
        assert user is None
    
    def test_authenticate_nonexistent_user(self):
        """Test authentication with non-existent user."""
        manager = JWTManager()
        user = manager.authenticate("nonexistent", "password")
        assert user is None
    
    def test_authenticate_inactive_user(self):
        """Test authentication with inactive user."""
        manager = JWTManager()
        user = manager.create_user(
            username="testuser",
            password="testpassword",
        )
        user.is_active = False
        result = manager.authenticate("testuser", "testpassword")
        assert result is None
    
    def test_create_access_token(self):
        """Test access token creation."""
        manager = JWTManager()
        user = manager.create_user(
            username="testuser",
            password="testpassword",
        )
        token = manager.create_access_token(user)
        assert token is not None
        assert len(token) > 0
    
    def test_create_refresh_token(self):
        """Test refresh token creation."""
        manager = JWTManager()
        user = manager.create_user(
            username="testuser",
            password="testpassword",
        )
        token = manager.create_refresh_token(user)
        assert token is not None
        assert len(token) > 0
    
    def test_decode_token(self):
        """Test token decoding."""
        manager = JWTManager()
        user = manager.create_user(
            username="testuser",
            password="testpassword",
        )
        token = manager.create_access_token(user)
        payload = manager.decode_token(token)
        assert payload is not None
        assert payload["username"] == "testuser"
        assert payload["sub"] == user.user_id
    
    def test_decode_token_invalid(self):
        """Test decoding invalid token."""
        manager = JWTManager()
        payload = manager.decode_token("invalid.token.here")
        assert payload is None
    
    def test_get_user(self):
        """Test getting user by username."""
        manager = JWTManager()
        manager.create_user(username="testuser", password="testpassword")
        user = manager.get_user("testuser")
        assert user is not None
        assert user.username == "testuser"
    
    def test_get_user_not_found(self):
        """Test getting non-existent user."""
        manager = JWTManager()
        user = manager.get_user("nonexistent")
        assert user is None
    
    def test_delete_user(self):
        """Test deleting user."""
        manager = JWTManager()
        manager.create_user(username="testuser", password="testpassword")
        result = manager.delete_user("testuser")
        assert result is True
        assert manager.get_user("testuser") is None
    
    def test_delete_user_not_found(self):
        """Test deleting non-existent user."""
        manager = JWTManager()
        result = manager.delete_user("nonexistent")
        assert result is False
    
    def test_update_user_subscription(self):
        """Test updating user subscription."""
        manager = JWTManager()
        user = manager.create_user(username="testuser", password="testpassword")
        
        sub = Subscription()
        sub.plan = SubscriptionPlan.LIFETIME
        sub.status = SubscriptionStatus.ACTIVE
        
        result = manager.update_user_subscription("testuser", sub)
        assert result is True
        updated_user = manager.get_user("testuser")
        assert updated_user.subscription.plan == SubscriptionPlan.LIFETIME
