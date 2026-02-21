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
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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
    ) -> User:
        """Create a new user."""
        user_id = f"user_{len(self._users) + 1}"
        hashed_password = self.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            hashed_password=hashed_password,
            role=role,
            is_active=True,
            created_at=datetime.now(),
        )
        
        self._users[username] = user
        logger.info(f"Created user: {username} with role: {role.value}")
        
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
        expire = datetime.now() + timedelta(
            minutes=self.config.access_token_expire_minutes
        )
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": expire,
            "iat": datetime.now(),
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
                algorithms=[self.config.algorithm]
            )
            
            return TokenPayload(
                sub=payload.get("sub"),
                username=payload.get("username"),
                role=payload.get("role"),
                exp=datetime.fromtimestamp(payload.get("exp")),
                iat=datetime.fromtimestamp(payload.get("iat")) if payload.get("iat") else None
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

