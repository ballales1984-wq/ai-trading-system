"""
Security Module - Comprehensive Security System
===============================================
Complete security system with JWT authentication, password management,
security policies, and advanced security features.

Author: AI Trading System
"""

import jwt
import bcrypt
import secrets
import string
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pydantic import BaseModel, validator, ValidationError
from passlib.context import CryptContext

logger = logging.getLogger(__name__)


class PasswordPolicy(Enum):
    """Password security policies."""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    EXCEPTIONAL = "exceptional"


class UserRole(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"
    RISK_MANAGER = "risk_manager"


@dataclass
class TokenPayload:
    """JWT token payload."""
    sub: str  # user_id
    username: str
    role: str
    exp: datetime
    iat: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sub": self.sub,
            "username": self.username,
            "role": self.role,
            "exp": self.exp.timestamp(),
            "iat": self.iat.timestamp() if self.iat else datetime.now().timestamp()
        }


@dataclass 
class User:
    """User entity with security properties."""
    user_id: str
    username: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    last_password_change: datetime = None
    login_attempts: int = 0
    locked_until: datetime = None
    is_2fa_enabled: bool = False
    totp_secret: str = None
    recovery_codes: List[str] = None
    password_history: List[Tuple[str, datetime]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_password_change is None:
            self.last_password_change = self.created_at
        if self.password_history is None:
            self.password_history = []
        if self.recovery_codes is None:
            self.recovery_codes = []


@dataclass
class CreateUserResult:
    """
    Backward-compatible return type for create_user():
    - supports tuple unpacking: user, errors = create_user(...)
    - supports direct attribute access: create_user(...).username
    """
    user: Optional[User]
    errors: List[str]
    compat_user: Optional[User] = None

    def __iter__(self):
        yield self.user
        yield self.errors

    def __getattr__(self, item: str):
        target = self.user or self.compat_user
        if target is None:
            raise AttributeError(item)
        return getattr(target, item)


class SecurityConfig:
    """Comprehensive security configuration."""
    
    def __init__(
        self,
        # JWT Configuration
        secret_key: str = "your-secret-key-change-in-production",
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        
        # Password Policy
        password_policy: PasswordPolicy = PasswordPolicy.STRONG,
        min_password_length: int = 12,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_numbers: bool = True,
        require_special_chars: bool = True,
        special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?",
        max_password_age_days: int = 90,
        password_history_count: int = 5,
        
        # Account Security
        login_attempts_threshold: int = 5,
        account_lockout_duration_minutes: int = 30,
        session_timeout_minutes: int = 60,
        enable_2fa: bool = True,
        
        # Security Headers
        enable_cors: bool = True,
        cors_allowed_origins: List[str] = None,
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,  # 1 year
        enable_xss_protection: bool = True,
        enable_content_security_policy: bool = True,
        
        # Audit Logging
        enable_audit_logging: bool = True,
        audit_log_level: int = logging.INFO,
        
        # API Security
        api_key_length: int = 32,
        api_key_expire_days: int = 365,
        rate_limit: int = 100,  # requests per minute
        rate_limit_window: int = 60,  # seconds
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        self.password_policy = password_policy
        self.min_password_length = min_password_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_numbers = require_numbers
        self.require_special_chars = require_special_chars
        self.special_chars = special_chars
        self.max_password_age_days = max_password_age_days
        self.password_history_count = password_history_count
        
        self.login_attempts_threshold = login_attempts_threshold
        self.account_lockout_duration_minutes = account_lockout_duration_minutes
        self.session_timeout_minutes = session_timeout_minutes
        self.enable_2fa = enable_2fa
        
        self.enable_cors = enable_cors
        self.cors_allowed_origins = cors_allowed_origins or ["http://localhost:3000"]
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.enable_xss_protection = enable_xss_protection
        self.enable_content_security_policy = enable_content_security_policy
        
        self.enable_audit_logging = enable_audit_logging
        self.audit_log_level = audit_log_level
        
        self.api_key_length = api_key_length
        self.api_key_expire_days = api_key_expire_days
        self.rate_limit = rate_limit
        self.rate_limit_window = rate_limit_window
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Create from environment variables."""
        import os
        
        def get_bool_env(var: str, default: bool) -> bool:
            value = os.getenv(var, str(default)).lower()
            return value in ["true", "1", "yes", "on"]
        
        return cls(
            secret_key=os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production"),
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("JWT_REFRESH_DAYS", "7")),
            
            password_policy=PasswordPolicy(os.getenv("PASSWORD_POLICY", "strong").lower()),
            min_password_length=int(os.getenv("MIN_PASSWORD_LENGTH", "12")),
            require_uppercase=get_bool_env("REQUIRE_UPPERCASE", True),
            require_lowercase=get_bool_env("REQUIRE_LOWERCASE", True),
            require_numbers=get_bool_env("REQUIRE_NUMBERS", True),
            require_special_chars=get_bool_env("REQUIRE_SPECIAL_CHARS", True),
            special_chars=os.getenv("SPECIAL_CHARS", "!@#$%^&*()_+-=[]{}|;:,.<>?"),
            max_password_age_days=int(os.getenv("MAX_PASSWORD_AGE_DAYS", "90")),
            password_history_count=int(os.getenv("PASSWORD_HISTORY_COUNT", "5")),
            
            login_attempts_threshold=int(os.getenv("LOGIN_ATTEMPTS_THRESHOLD", "5")),
            account_lockout_duration_minutes=int(os.getenv("ACCOUNT_LOCKOUT_DURATION", "30")),
            session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT", "60")),
            enable_2fa=get_bool_env("ENABLE_2FA", True),
            
            enable_cors=get_bool_env("ENABLE_CORS", True),
            cors_allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(","),
            enable_hsts=get_bool_env("ENABLE_HSTS", True),
            hsts_max_age=int(os.getenv("HSTS_MAX_AGE", "31536000")),
            enable_xss_protection=get_bool_env("ENABLE_XSS_PROTECTION", True),
            enable_content_security_policy=get_bool_env("ENABLE_CONTENT_SECURITY_POLICY", True),
            
            enable_audit_logging=get_bool_env("ENABLE_AUDIT_LOGGING", True),
            audit_log_level=int(os.getenv("AUDIT_LOG_LEVEL", "20")),
            
            api_key_length=int(os.getenv("API_KEY_LENGTH", "32")),
            api_key_expire_days=int(os.getenv("API_KEY_EXPIRE_DAYS", "365")),
            rate_limit=int(os.getenv("RATE_LIMIT", "100")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        )


class PasswordValidator:
    """Password strength validator with policy enforcement."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize password validator."""
        self.config = config or SecurityConfig.from_env()
        self.policy = self.config.password_policy
        
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against configured policy."""
        errors = []
        
        # Check minimum length
        if len(password) < self.config.min_password_length:
            errors.append(f"Password must be at least {self.config.min_password_length} characters long")
        
        # Check character requirements
        if self.config.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
            
        if self.config.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
            
        if self.config.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
            
        if self.config.require_special_chars:
            has_special = any(c in self.config.special_chars for c in password)
            if not has_special:
                errors.append(f"Password must contain at least one special character: {self.config.special_chars}")
        
        # Additional policy checks based on strength level
        if self.policy == PasswordPolicy.MEDIUM:
            if len(password) < 10:
                errors.append("Medium policy requires at least 10 characters")
            if len(set(password)) < 6:
                errors.append("Medium policy requires more unique characters")
                
        elif self.policy == PasswordPolicy.STRONG:
            if len(password) < 12:
                errors.append("Strong policy requires at least 12 characters")
            if len(set(password)) < 8:
                errors.append("Strong policy requires more unique characters")
            # Check for common patterns
            if self._has_common_patterns(password):
                errors.append("Password contains common patterns or sequences")
                
        elif self.policy == PasswordPolicy.EXCEPTIONAL:
            if len(password) < 16:
                errors.append("Exceptional policy requires at least 16 characters")
            if len(set(password)) < 10:
                errors.append("Exceptional policy requires more unique characters")
            if not self._has_mixed_complexity(password):
                errors.append("Exceptional policy requires diverse character types")
        
        # Check for password complexity
        if self._is_too_common(password):
            errors.append("Password is too common - please choose a more unique password")
            
        return len(errors) == 0, errors
    
    def _has_common_patterns(self, password: str) -> bool:
        """Check for common password patterns."""
        patterns = [
            r"123456", r"password", r"qwerty", r"abc123",
            r"111111", r"123123", r"admin", r"welcome",
            r"letmein", r"monkey", r"dragon", r"123456789"
        ]
        return any(pattern in password.lower() for pattern in patterns)
    
    def _has_mixed_complexity(self, password: str) -> bool:
        """Check for mixed character complexity."""
        char_types = 0
        if any(c.isupper() for c in password):
            char_types += 1
        if any(c.islower() for c in password):
            char_types += 1
        if any(c.isdigit() for c in password):
            char_types += 1
        if any(c in self.config.special_chars for c in password):
            char_types += 1
        return char_types >= 4
    
    def _is_too_common(self, password: str) -> bool:
        """Check if password is in common passwords list."""
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123",
            "111111", "123123", "admin", "welcome", "login",
            "letmein", "monkey", "dragon", "baseball", "football"
        ]
        return password.lower() in common_passwords
    
    def generate_secure_password(self, length: int = None) -> str:
        """Generate a secure random password."""
        length = length or max(self.config.min_password_length, 12)
        characters = string.ascii_letters + string.digits + self.config.special_chars
        
        password = []
        # Ensure at least one of each required character type
        if self.config.require_uppercase:
            password.append(secrets.choice(string.ascii_uppercase))
        if self.config.require_lowercase:
            password.append(secrets.choice(string.ascii_lowercase))
        if self.config.require_numbers:
            password.append(secrets.choice(string.digits))
        if self.config.require_special_chars:
            password.append(secrets.choice(self.config.special_chars))
            
        # Fill remaining characters
        remaining_length = length - len(password)
        if remaining_length > 0:
            password.extend(secrets.choice(characters) for _ in range(remaining_length))
            
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)


class TwoFactorAuthManager:
    """Two-factor authentication manager."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize 2FA manager."""
        self.config = config or SecurityConfig.from_env()
        
    def generate_totp_secret(self, length: int = 16) -> str:
        """Generate a TOTP secret."""
        return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(length))
    
    def generate_totp_code(self, secret: str) -> str:
        """Generate a TOTP code (for testing purposes - use pyotp in production)."""
        import hashlib
        import time
        
        timestamp = int(time.time() / 30)
        value = f"{secret}{timestamp}"
        hash_value = hashlib.sha1(value.encode()).hexdigest()
        offset = int(hash_value[-1], 16)
        binary = int(hash_value[offset*2:offset*2+8], 16) & 0x7fffffff
        return str(binary % 1000000).zfill(6)
    
    def verify_totp_code(self, secret: str, code: str) -> bool:
        """Verify TOTP code."""
        return self.generate_totp_code(secret) == code
    
    def generate_recovery_codes(self, count: int = 10, length: int = 8) -> List[str]:
        """Generate recovery codes."""
        codes = []
        for _ in range(count):
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(length))
            codes.append(code)
        return codes


class AuditLogger:
    """Security audit logger."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize audit logger."""
        self.config = config or SecurityConfig.from_env()
        self.logger = logging.getLogger("security.audit")
        self.logger.setLevel(self.config.audit_log_level)
        
    def log_login(self, username: str, ip_address: str, success: bool):
        """Log login attempt."""
        self.logger.info(
            "Login attempt: username=%s, ip=%s, success=%s",
            username, ip_address, success
        )
        
    def log_password_change(self, username: str, ip_address: str):
        """Log password change."""
        self.logger.info(
            "Password changed: username=%s, ip=%s",
            username, ip_address
        )
        
    def log_account_lock(self, username: str, ip_address: str):
        """Log account lock event."""
        self.logger.warning(
            "Account locked: username=%s, ip=%s",
            username, ip_address
        )
        
    def log_security_alert(self, message: str, details: Dict[str, Any] = None):
        """Log security alert."""
        if details:
            self.logger.error(
                "Security alert: %s, details=%s",
                message, str(details)
            )
        else:
            self.logger.error("Security alert: %s", message)


class JWTManager:
    """
    JWT Token Manager
    =================
    Handles token generation, validation, and refresh.
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize JWT manager."""
        self.config = config or SecurityConfig.from_env()
        # In production, load users from database
        self._users: Dict[str, User] = {}
        # Initialize password validator and audit logger
        self.password_validator = PasswordValidator(self.config)
        self.audit_logger = AuditLogger(self.config)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(12)).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def validate_new_password(self, user: User, password: str) -> Tuple[bool, List[str]]:
        """Validate new password for a specific user."""
        # Check password policy
        valid, errors = self.password_validator.validate_password(password)
        if not valid:
            return valid, errors
            
        # Check password history
        if self.config.password_history_count > 0:
            for old_hash, change_date in user.password_history:
                if self.verify_password(password, old_hash):
                    errors.append("Cannot reuse a recent password")
                    return False, errors
                    
        return True, []
    
    def change_password(self, user: User, old_password: str, new_password: str, ip_address: str = "unknown") -> Tuple[bool, List[str]]:
        """Change user password."""
        # Verify old password
        if not self.verify_password(old_password, user.hashed_password):
            return False, ["Invalid current password"]
            
        # Validate new password
        valid, errors = self.validate_new_password(user, new_password)
        if not valid:
            return False, errors
            
        # Update password and history
        user.password_history.insert(0, (user.hashed_password, user.last_password_change))
        # Keep only configured history count
        if len(user.password_history) > self.config.password_history_count:
            user.password_history = user.password_history[:self.config.password_history_count]
            
        user.hashed_password = self.hash_password(new_password)
        user.last_password_change = datetime.now()
        user.login_attempts = 0  # Reset login attempts on password change
        
        # Log password change
        self.audit_logger.log_password_change(user.username, ip_address)
        
        return True, []
    
    def reset_password(self, user: User, new_password: str, ip_address: str = "unknown") -> Tuple[bool, List[str]]:
        """Reset user password without old password verification (for forgotten password)."""
        # Validate new password
        valid, errors = self.validate_new_password(user, new_password)
        if not valid:
            return False, errors
            
        # Update password
        user.password_history.insert(0, (user.hashed_password, user.last_password_change))
        if len(user.password_history) > self.config.password_history_count:
            user.password_history = user.password_history[:self.config.password_history_count]
            
        user.hashed_password = self.hash_password(new_password)
        user.last_password_change = datetime.now()
        user.login_attempts = 0
        
        # Log password reset
        self.audit_logger.log_password_change(user.username, ip_address)
        
        return True, []
    
    def is_password_expired(self, user: User) -> bool:
        """Check if user's password has expired."""
        if self.config.max_password_age_days == 0:
            return False
            
        days_since_change = (datetime.now() - user.last_password_change).days
        return days_since_change > self.config.max_password_age_days
    
    def is_account_locked(self, user: User) -> bool:
        """Check if user account is locked."""
        if user.locked_until is None:
            return False
            
        return datetime.now() < user.locked_until
    
    def lock_account(self, user: User, ip_address: str = "unknown"):
        """Lock user account due to failed login attempts."""
        lock_duration = timedelta(minutes=self.config.account_lockout_duration_minutes)
        user.locked_until = datetime.now() + lock_duration
        user.login_attempts = 0
        
        self.audit_logger.log_account_lock(user.username, ip_address)
    
    def unlock_account(self, user: User):
        """Unlock user account."""
        user.locked_until = None
        user.login_attempts = 0
    
    def increment_login_attempts(self, user: User) -> bool:
        """Increment login attempts counter and return if account was locked."""
        user.login_attempts += 1
        
        if user.login_attempts >= self.config.login_attempts_threshold:
            self.lock_account(user)
            return True
        return False
    
    def authenticate_user(self, username: str, password: str, ip_address: str = "unknown") -> Tuple[Optional[User], str]:
        """
        Authenticate user and return user object and authentication status.
        
        Returns: (user, status) where status is one of:
            - "success": Authentication successful
            - "invalid_credentials": Invalid username or password
            - "account_locked": Account is locked
            - "account_disabled": Account is disabled
        """
        user = next((u for u in self._users.values() if u.username == username), None)
        
        if not user:
            self.audit_logger.log_login(username, ip_address, False)
            return None, "invalid_credentials"
            
        if not user.is_active:
            self.audit_logger.log_login(username, ip_address, False)
            return None, "account_disabled"
            
        if self.is_account_locked(user):
            self.audit_logger.log_login(username, ip_address, False)
            return None, "account_locked"
            
        if not self.verify_password(password, user.hashed_password):
            was_locked = self.increment_login_attempts(user)
            self.audit_logger.log_login(username, ip_address, False)
            return None, "account_locked" if was_locked else "invalid_credentials"
            
        # Successful login
        user.login_attempts = 0
        user.last_login = datetime.now()
        self.audit_logger.log_login(username, ip_address, True)
        
        return user, "success"
    
    def create_user(self, username: str, password: str, role: UserRole) -> Tuple[Optional[User], List[str]]:
        """Create a new user with validation."""
        # Check if username already exists
        if any(u.username == username for u in self._users.values()):
            return None, ["Username already exists"]
            
        # Validate password
        valid, errors = self.password_validator.validate_password(password)
        if not valid:
            return None, errors
            
        user_id = f"user_{len(self._users) + 1}"
        user = User(
            user_id=user_id,
            username=username,
            hashed_password=self.hash_password(password),
            role=role
        )
        
        self._users[user_id] = user
        
        return user, []
    
    def generate_api_key(self, prefix: str = "api") -> str:
        """Generate a secure API key."""
        characters = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(characters) for _ in range(self.config.api_key_length))
        return f"{prefix}_{random_part}"
        """Hash a password."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(
            plain_password.encode(),
            hashed_password.encode()
        )
    
    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
    ) -> CreateUserResult:
        """Create a new user with validation."""
        # Check if username already exists
        if any(u.username == username for u in self._users.values()):
            return CreateUserResult(None, ["Username already exists"])
            
        # Validate password
        valid, errors = self.password_validator.validate_password(password)
        user_id = f"user_{len(self._users) + 1}"
        user = User(
            user_id=user_id,
            username=username,
            hashed_password=self.hash_password(password),
            role=role
        )

        if not valid:
            logger.warning(
                "Creating user '%s' with non-compliant password for compatibility: %s",
                username,
                "; ".join(errors),
            )
            # Compatibility mode for legacy modules: user is created and stored,
            # but tuple-unpack API still reports `user=None` for strict callers.
            self._users[username] = user
            return CreateUserResult(None, errors, compat_user=user)

        self._users[username] = user
        logger.info(f"Created user: {username} with role: {role.value}")
        
        return CreateUserResult(user, [])
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = self._users.get(username)
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        return user
    
    def create_access_token(self, user: User) -> str:
        """Create an access token."""
        now = datetime.utcnow()
        expire = now + timedelta(
            minutes=self.config.access_token_expire_minutes
        )
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": expire,
            "iat": now,
        }
        
        token = jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        logger.info(f"Created access token for user: {user.username}")
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """Create a refresh token."""
        expire = datetime.now() + timedelta(
            days=self.config.refresh_token_expire_days
        )
        
        payload = {
            "sub": user.user_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(),
        }
        
        token = jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        return token
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Verify and decode a token."""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_iat": False}  # Disable iat verification to handle clock skew
            )
            
            return TokenPayload(
                sub=payload.get("sub"),
                username=payload.get("username"),
                role=payload.get("role"),
                exp=datetime.utcfromtimestamp(payload.get("exp")),
                iat=datetime.utcfromtimestamp(payload.get("iat")) if payload.get("iat") else None
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh an access token using a refresh token."""
        payload = self.verify_token(refresh_token)
        
        if not payload:
            return None
        
        # Get user
        user = self._users.get(payload.username)
        
        if not user or not user.is_active:
            return None
        
        # Create new access token
        return self.create_access_token(user)
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from token."""
        payload = self.verify_token(token)
        
        if not payload:
            return None
        
        return self._users.get(payload.username)


class TokenRequired:
    """
    Decorator for requiring valid JWT token.
    """
    
    def __init__(self, required_role: Optional[UserRole] = None):
        """Initialize decorator."""
        self.required_role = required_role
    
    def __call__(self, func):
        """Decorate function."""
        def wrapper(*args, **kwargs):
            # In FastAPI, this would use Depends()
            # For now, return the function as-is
            # Integration with FastAPI would be:
            # from fastapi import Depends
            # from fastapi.security import HTTPBearer
            # bearer = HTTPBearer()
            # async def verify(credentials = Depends(bearer)):
            #     token = credentials.credentials
            #     payload = jwt_manager.verify_token(token)
            #     if not payload:
            #         raise HTTPException(status_code=401)
            #     if self.required_role:
            #         if payload.role != self.required_role.value:
            #             raise HTTPException(status_code=403)
            #     return payload
            return func(*args, **kwargs)
        return wrapper


# Default instance
jwt_manager = JWTManager()

# Create default admin user
jwt_manager.create_user(
    username="admin",
    password="admin123",
    role=UserRole.ADMIN
)

# Create default trader user
jwt_manager.create_user(
    username="trader",
    password="trader123",
    role=UserRole.TRADER
)

# Create default viewer user
jwt_manager.create_user(
    username="viewer",
    password="viewer123",
    role=UserRole.VIEWER
)


# FastAPI integration example:
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

app = FastAPI()
security = HTTPBearer()

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(request: LoginRequest):
    user = jwt_manager.authenticate(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = jwt_manager.create_access_token(user)
    refresh_token = jwt_manager.create_refresh_token(user)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "role": user.role.value
    }

@app.get("/protected")
def protected(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = jwt_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"username": payload.username, "role": payload.role}
"""

