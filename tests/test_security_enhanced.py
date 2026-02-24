"""
Tests for Enhanced Security Module
=================================
Tests for the comprehensive security system including password policies,
account security, and advanced features.

Author: AI Trading System
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.security import (
    SecurityConfig, PasswordPolicy, PasswordValidator,
    TwoFactorAuthManager, AuditLogger, JWTManager, User, UserRole
)


class TestPasswordValidator:
    """Test password validation functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = SecurityConfig()
        self.validator = PasswordValidator(self.config)
        
    def test_weak_password_validation(self):
        """Test weak password validation fails."""
        valid, errors = self.validator.validate_password("password")
        assert not valid
        assert len(errors) > 0
        
    def test_strong_password_validation(self):
        """Test strong password passes validation."""
        valid, errors = self.validator.validate_password("Str0ngP@ssw0rd!123")
        assert valid
        assert len(errors) == 0
        
    def test_password_length_validation(self):
        """Test password length validation."""
        config = SecurityConfig(min_password_length=10)
        validator = PasswordValidator(config)
        
        valid, errors = validator.validate_password("Short1!")
        assert not valid
        assert "at least 10 characters" in str(errors)
        
    def test_special_char_validation(self):
        """Test special character requirement."""
        config = SecurityConfig(require_special_chars=True)
        validator = PasswordValidator(config)
        
        valid, errors = validator.validate_password("StrongPassword123")
        assert not valid
        assert "special character" in str(errors)
        
    def test_generate_secure_password(self):
        """Test secure password generation."""
        password = self.validator.generate_secure_password()
        assert len(password) >= self.config.min_password_length
        
        valid, errors = self.validator.validate_password(password)
        assert valid
        assert len(errors) == 0
        
    def test_common_password_detection(self):
        """Test common passwords are rejected."""
        valid, errors = self.validator.validate_password("qwerty123")
        assert not valid
        assert any("common" in error.lower() for error in errors)
        
    def test_password_pattern_detection(self):
        """Test password pattern detection."""
        valid, errors = self.validator.validate_password("1234567890")
        assert not valid
        assert "common patterns" in str(errors)


class TestTwoFactorAuthManager:
    """Test two-factor authentication functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = SecurityConfig()
        self.tfa_manager = TwoFactorAuthManager(self.config)
        
    def test_generate_totp_secret(self):
        """Test TOTP secret generation."""
        secret = self.tfa_manager.generate_totp_secret()
        assert len(secret) > 0
        assert secret.isalnum()
        
    def test_generate_totp_code(self):
        """Test TOTP code generation."""
        secret = "TESTSECRET123456"
        code1 = self.tfa_manager.generate_totp_code(secret)
        code2 = self.tfa_manager.generate_totp_code(secret)
        
        assert len(code1) == 6
        assert code1.isdigit()
        assert code1 == code2  # Same secret should produce same code
        
    def test_verify_totp_code(self):
        """Test TOTP code verification."""
        secret = "TESTSECRET123456"
        code = self.tfa_manager.generate_totp_code(secret)
        
        assert self.tfa_manager.verify_totp_code(secret, code)
        assert not self.tfa_manager.verify_totp_code(secret, "123456")
        
    def test_generate_recovery_codes(self):
        """Test recovery code generation."""
        codes = self.tfa_manager.generate_recovery_codes()
        assert len(codes) == 10
        
        for code in codes:
            assert len(code) == 8
            assert code.isalnum()
        
        # Check all codes are unique
        assert len(set(codes)) == len(codes)


class TestAuditLogger:
    """Test audit logging functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = SecurityConfig()
        self.logger = AuditLogger(self.config)
        
    @patch("app.core.security.logging.getLogger")
    def test_login_logging(self, mock_get_logger):
        """Test login attempt logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = AuditLogger(self.config)
        logger.log_login("testuser", "192.168.1.1", True)
        
        mock_logger.info.assert_called()
        
    @patch("app.core.security.logging.getLogger")
    def test_password_change_logging(self, mock_get_logger):
        """Test password change logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = AuditLogger(self.config)
        logger.log_password_change("testuser", "192.168.1.1")
        
        mock_logger.info.assert_called()
        
    @patch("app.core.security.logging.getLogger")
    def test_account_lock_logging(self, mock_get_logger):
        """Test account lock logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = AuditLogger(self.config)
        logger.log_account_lock("testuser", "192.168.1.1")
        
        mock_logger.warning.assert_called()


class TestJWTManagerEnhanced:
    """Test enhanced JWTManager functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = SecurityConfig(
            login_attempts_threshold=3,
            account_lockout_duration_minutes=5,
            password_history_count=3
        )
        self.jwt_manager = JWTManager(config=self.config)
        
    def test_user_creation(self):
        """Test user creation with validation."""
        user, errors = self.jwt_manager.create_user(
            "testuser", "Str0ngP@ssw0rd!123", UserRole.TRADER
        )
        
        assert user is not None
        assert len(errors) == 0
        assert user.username == "testuser"
        assert user.role == UserRole.TRADER
        
    def test_duplicate_user_creation(self):
        """Test duplicate user creation fails."""
        self.jwt_manager.create_user(
            "testuser", "Str0ngP@ssw0rd!123", UserRole.TRADER
        )
        
        user, errors = self.jwt_manager.create_user(
            "testuser", "AnotherP@ssw0rd!456", UserRole.VIEWER
        )
        
        assert user is None
        assert len(errors) > 0
        
    def test_password_validation_on_creation(self):
        """Test password validation on user creation."""
        user, errors = self.jwt_manager.create_user(
            "invaliduser", "weak", UserRole.VIEWER
        )
        
        assert user is None
        assert len(errors) > 0
        
    def test_password_change(self):
        """Test password change functionality."""
        user, _ = self.jwt_manager.create_user(
            "changetest", "OldP@ssw0rd!123", UserRole.TRADER
        )
        
        success, errors = self.jwt_manager.change_password(
            user, "OldP@ssw0rd!123", "NewP@ssw0rd!456"
        )
        
        assert success
        assert len(errors) == 0
        
    def test_invalid_password_change(self):
        """Test password change with invalid old password fails."""
        user, _ = self.jwt_manager.create_user(
            "invalidtest", "CurrentP@ssw0rd!123", UserRole.TRADER
        )
        
        success, errors = self.jwt_manager.change_password(
            user, "WrongP@ssw0rd!789", "NewP@ssw0rd!456"
        )
        
        assert not success
        assert len(errors) > 0
        
    def test_password_history(self):
        """Test password history prevents reuse."""
        user, _ = self.jwt_manager.create_user(
            "historytest", "FirstP@ssw0rd!123", UserRole.TRADER
        )
        
        # Change to second password
        self.jwt_manager.change_password(
            user, "FirstP@ssw0rd!123", "SecondP@ssw0rd!456"
        )
        
        # Change to third password
        self.jwt_manager.change_password(
            user, "SecondP@ssw0rd!456", "ThirdP@ssw0rd!789"
        )
        
        # Try to reuse first password - should fail
        success, errors = self.jwt_manager.change_password(
            user, "ThirdP@ssw0rd!789", "FirstP@ssw0rd!123"
        )
        
        assert not success
        assert "recent password" in str(errors)
        
    def test_login_attempts_and_locking(self):
        """Test login attempts and account locking."""
        user, _ = self.jwt_manager.create_user(
            "locktest", "LockP@ssw0rd!123", UserRole.TRADER
        )
        
        # First two failed attempts
        for _ in range(2):
            result, status = self.jwt_manager.authenticate_user(
                "locktest", "WrongPassword!456"
            )
            assert status == "invalid_credentials"
            
        assert user.login_attempts == 2
        
        # Third failed attempt should lock the account
        result, status = self.jwt_manager.authenticate_user(
            "locktest", "WrongPassword!456"
        )
        
        assert status == "account_locked"
        assert user.locked_until is not None
        
    def test_authenticate_locked_account(self):
        """Test locked account cannot login."""
        user, _ = self.jwt_manager.create_user(
            "lockeduser", "LockedP@ssw0rd!123", UserRole.TRADER
        )
        
        self.jwt_manager.lock_account(user)
        
        result, status = self.jwt_manager.authenticate_user(
            "lockeduser", "LockedP@ssw0rd!123"
        )
        
        assert status == "account_locked"
        
    def test_password_expiration(self):
        """Test password expiration check."""
        config = SecurityConfig(max_password_age_days=1)
        jwt_manager = JWTManager(config=config)
        
        user, _ = jwt_manager.create_user(
            "expiretest", "ExpireP@ssw0rd!123", UserRole.TRADER
        )
        
        # Set last password change to be in the past
        user.last_password_change = datetime.now() - timedelta(days=2)
        
        assert jwt_manager.is_password_expired(user)
        
    def test_authenticate_user_success(self):
        """Test successful user authentication."""
        user, _ = self.jwt_manager.create_user(
            "successtest", "SuccessP@ssw0rd!123", UserRole.TRADER
        )
        
        result, status = self.jwt_manager.authenticate_user(
            "successtest", "SuccessP@ssw0rd!123"
        )
        
        assert status == "success"
        assert result is not None
        assert result.username == "successtest"
        
    def test_reset_password(self):
        """Test password reset functionality."""
        user, _ = self.jwt_manager.create_user(
            "resettest", "OriginalP@ssw0rd!123", UserRole.TRADER
        )
        
        success, errors = self.jwt_manager.reset_password(
            user, "NewP@ssw0rd!456"
        )
        
        assert success
        assert len(errors) == 0
        
        # Verify new password works
        result, status = self.jwt_manager.authenticate_user(
            "resettest", "NewP@ssw0rd!456"
        )
        
        assert status == "success"


class TestSecurityConfigEnhanced:
    """Test enhanced security configuration."""
    
    def test_password_policy_config(self):
        """Test password policy configuration."""
        config = SecurityConfig(password_policy=PasswordPolicy.STRONG)
        assert config.password_policy == PasswordPolicy.STRONG
        assert config.min_password_length == 12
        
        config = SecurityConfig(password_policy=PasswordPolicy.EXCEPTIONAL)
        assert config.password_policy == PasswordPolicy.EXCEPTIONAL
        assert config.min_password_length == 12
        
    @patch.dict(os.environ, {
        'PASSWORD_POLICY': 'exceptional',
        'MIN_PASSWORD_LENGTH': '16',
        'REQUIRE_SPECIAL_CHARS': 'true',
        'LOGIN_ATTEMPTS_THRESHOLD': '10',
        'ENABLE_2FA': 'true'
    })
    def test_env_config(self):
        """Test configuration from environment variables."""
        config = SecurityConfig.from_env()
        
        assert config.password_policy == PasswordPolicy.EXCEPTIONAL
        assert config.min_password_length == 16
        assert config.require_special_chars is True
        assert config.login_attempts_threshold == 10
        assert config.enable_2fa is True
        
    def test_api_key_config(self):
        """Test API key configuration."""
        config = SecurityConfig(api_key_length=40, api_key_expire_days=365)
        jwt_manager = JWTManager(config=config)
        
        api_key = jwt_manager.generate_api_key()
        assert len(api_key) > 40
        assert api_key.startswith("api_")
        
    def test_cors_config(self):
        """Test CORS configuration."""
        config = SecurityConfig(
            cors_allowed_origins=["https://app.example.com", "https://api.example.com"]
        )
        
        assert len(config.cors_allowed_origins) == 2
        assert "https://app.example.com" in config.cors_allowed_origins


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_complete_user_lifecycle(self):
        """Test complete user security lifecycle."""
        config = SecurityConfig(
            login_attempts_threshold=5,
            password_history_count=3,
            max_password_age_days=90
        )
        jwt_manager = JWTManager(config=config)
        
        # Create user
        user, errors = jwt_manager.create_user(
            "lifecycleuser", "InitialP@ssw0rd!123", UserRole.TRADER
        )
        assert user is not None
        
        # Authenticate
        auth_result, status = jwt_manager.authenticate_user(
            "lifecycleuser", "InitialP@ssw0rd!123"
        )
        assert status == "success"
        
        # Change password
        change_success, change_errors = jwt_manager.change_password(
            user, "InitialP@ssw0rd!123", "ChangedP@ssw0rd!456"
        )
        assert change_success
        
        # Verify new password works
        auth_result, status = jwt_manager.authenticate_user(
            "lifecycleuser", "ChangedP@ssw0rd!456"
        )
        assert status == "success"
        
        # Reset password
        reset_success, reset_errors = jwt_manager.reset_password(
            user, "ResetP@ssw0rd!789"
        )
        assert reset_success
        
        # Verify reset password works
        auth_result, status = jwt_manager.authenticate_user(
            "lifecycleuser", "ResetP@ssw0rd!789"
        )
        assert status == "success"
        
    def test_multiple_security_policies(self):
        """Test different security policies."""
        # Test strong policy
        strong_config = SecurityConfig(password_policy=PasswordPolicy.STRONG)
        strong_validator = PasswordValidator(strong_config)
        
        valid, errors = strong_validator.validate_password("StrongP@ssw0rd!123")
        assert valid
        
        # Test exceptional policy
        exceptional_config = SecurityConfig(
            password_policy=PasswordPolicy.EXCEPTIONAL,
            min_password_length=16
        )
        exceptional_validator = PasswordValidator(exceptional_config)
        
        test_password = "StrongP@ssw0rd!"
        valid, errors = exceptional_validator.validate_password(test_password)
        print(f"Password: '{test_password}'")
        print(f"Length: {len(test_password)}")
        print(f"Valid: {valid}")
        print(f"Errors: {errors}")
        assert not valid  # Too short for exceptional policy
        
        valid, errors = exceptional_validator.validate_password("ExcePtionalP@ssw0rd!1234")
        assert valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
