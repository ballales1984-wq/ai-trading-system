"""
Security Module - JWT Authentication
====================================
JWT token generation and validation for API security.

Author: AI Trading System
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SubscriptionPlan(Enum):
    """Subscription plans."""
    TRIAL = "trial"
    FREE = "free"
    LIFETIME = "lifetime"  # One-time purchase


class SubscriptionStatus(Enum):
    """Subscription status."""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    EXPIRED = "expired"  # For lifetime purchases


@dataclass
class Subscription:
    """Subscription information."""
    plan: SubscriptionPlan = SubscriptionPlan.FREE
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    trial_end_date: Optional[datetime] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    cancel_at_period_end: bool = False

    def is_trial_active(self) -> bool:
        """Check if trial is active."""
        if self.trial_end_date is None:
            return False
        return datetime.now() < self.trial_end_date and self.status == SubscriptionStatus.TRIALING

    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING)

    def get_days_remaining(self) -> int:
        """Get days remaining in trial."""
        if not self.is_trial_active():
            return 0
        delta = self.trial_end_date - datetime.now()
        return max(0, delta.days)


class UserRole(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"


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
    """User entity."""
    user_id: str
    username: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    # Subscription fields
    email: str = ""
    subscription: Subscription = field(default_factory=Subscription)
    stripe_customer_id: Optional[str] = None
    stripe_price_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "subscription": {
                "plan": self.subscription.plan.value,
                "status": self.subscription.status.value,
                "trial_end_date": self.subscription.trial_end_date.isoformat() if self.subscription.trial_end_date else None,
                "is_trial_active": self.subscription.is_trial_active(),
                "days_remaining": self.subscription.get_days_remaining(),
                "current_period_end": self.subscription.current_period_end.isoformat() if self.subscription.current_period_end else None,
            },
        }


class SecurityConfig:
    """Security configuration."""
    
    def __init__(
        self,
        secret_key: str = "your-secret-key-change-in-production",
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Create from environment variables."""
        import os
        return cls(
            secret_key=os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production"),
            algorithm="HS256",
            access_token_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("JWT_REFRESH_DAYS", "7")),
        )


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
    
    def hash_password(self, password: str) -> str:
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
        email: str = "",
        trial_days: int = 0,
    ) -> User:
        """Create a new user with optional trial."""
        user_id = f"user_{len(self._users) + 1}"
        hashed_password = self.hash_password(password)
        
        # Setup subscription
        subscription = Subscription()
        if trial_days > 0:
            subscription.plan = SubscriptionPlan.TRIAL
            subscription.status = SubscriptionStatus.TRIALING
            subscription.trial_end_date = datetime.now() + timedelta(days=trial_days)
        
        user = User(
            user_id=user_id,
            username=username,
            hashed_password=hashed_password,
            role=role,
            is_active=True,
            created_at=datetime.now(),
            email=email,
            subscription=subscription,
        )
        
        self._users[username] = user
        logger.info(f"Created user: {username} with role: {role.value}, trial_days: {trial_days}")
        
        return user
    
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
    
    def check_subscription(self, user: User) -> Dict[str, Any]:
        """Check user's subscription status."""
        sub = user.subscription
        
        # Check if trial has expired
        if sub.status == SubscriptionStatus.TRIALING and sub.trial_end_date:
            if datetime.now() >= sub.trial_end_date:
                # Trial expired - convert to free or require payment
                sub.status = SubscriptionStatus.CANCELED
                sub.plan = SubscriptionPlan.FREE
                logger.warning(f"Trial expired for user: {user.username}")
        
        return {
            "plan": sub.plan.value,
            "status": sub.status.value,
            "is_active": sub.is_active(),
            "is_trial": sub.is_trial_active(),
            "trial_days_remaining": sub.get_days_remaining() if sub.is_trial_active() else 0,
            "current_period_end": sub.current_period_end.isoformat() if sub.current_period_end else None,
            "cancel_at_period_end": sub.cancel_at_period_end,
        }
    
    def activate_subscription(
        self,
        user: User,
        plan: SubscriptionPlan,
        stripe_customer_id: str = None,
        stripe_subscription_id: str = None,
        trial_days: int = 0,
    ) -> Subscription:
        """Activate or update user subscription."""
        subscription = user.subscription
        subscription.plan = plan
        subscription.stripe_customer_id = stripe_customer_id
        subscription.stripe_subscription_id = stripe_subscription_id
        
        if trial_days > 0:
            subscription.status = SubscriptionStatus.TRIALING
            subscription.trial_end_date = datetime.now() + timedelta(days=trial_days)
        else:
            subscription.status = SubscriptionStatus.ACTIVE
            subscription.current_period_start = datetime.now()
            subscription.current_period_end = datetime.now() + timedelta(days=30)
        
        logger.info(f"Activated subscription for {user.username}: plan={plan.value}, trial_days={trial_days}")
        return subscription
    
    def cancel_subscription(self, user: User, at_period_end: bool = True) -> bool:
        """Cancel user subscription."""
        subscription = user.subscription
        subscription.cancel_at_period_end = at_period_end
        
        if not at_period_end:
            subscription.status = SubscriptionStatus.CANCELED
            subscription.plan = SubscriptionPlan.FREE
        
        logger.info(f"Subscription canceled for {user.username}, at_period_end={at_period_end}")
        return True
    
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


import os

# Default instance
jwt_manager = JWTManager()

# Get users from environment variables (for production)
admin_user = os.getenv("ADMIN_USER", "admin")
admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
trader_user = os.getenv("TRADER_USER", "trader")
trader_pass = os.getenv("TRADER_PASSWORD", "trader123")

# Create default admin user
jwt_manager.create_user(
    username=admin_user,
    password=admin_pass,
    role=UserRole.ADMIN
)

# Create default trader user
jwt_manager.create_user(
    username=trader_user,
    password=trader_pass,
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

