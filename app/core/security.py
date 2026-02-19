"""
Security Module
=============
JWT authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.core.config import settings


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class TokenData(BaseModel):
    """Token data."""
    user_id: str
    username: str
    exp: datetime


class User(BaseModel):
    """User model."""
    user_id: str
    username: str
    email: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False


class UserInDB(User):
    """User in database."""
    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode JWT token."""
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        exp: datetime = payload.get("exp")
        
        if user_id is None:
            return None
        
        return TokenData(user_id=user_id, username=username, exp=exp)
    
    except JWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    token_data = decode_token(token)
    
    if token_data is None:
        raise credentials_exception
    
    # In production, fetch user from database
    # For now, create a mock user
    user = User(
        user_id=token_data.user_id,
        username=token_data.username,
        email=f"{token_data.username}@example.com"
    )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user


def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require admin privileges."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    return current_user


# Demo users for testing - hashes are computed lazily
DEMO_USERS = {
    "trader1": {
        "user_id": "user_001",
        "username": "trader1",
        "email": "trader1@example.com",
        "hashed_password": None,  # Computed on first use
        "is_active": True,
        "is_superuser": False
    },
    "admin": {
        "user_id": "admin_001",
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": None,  # Computed on first use
        "is_active": True,
        "is_superuser": True
    }
}

# Passwords for demo users (used for hashing)
DEMO_PASSWORDS = {
    "trader1": "password123",
    "admin": "admin123"
}

def _get_hashed_password(username: str) -> str:
    """Get or create hashed password for demo user."""
    user = DEMO_USERS.get(username)
    if user and user["hashed_password"] is None:
        password = DEMO_PASSWORDS.get(username, "")
        user["hashed_password"] = pwd_context.hash(password[:72])
    return user["hashed_password"] if user else None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user_data = DEMO_USERS.get(username)
    
    if not user_data:
        return None
    
    # Get the hashed password (lazy computation)
    hashed_password = _get_hashed_password(username)
    
    if not verify_password(password[:72], hashed_password):
        return None
    
    return User(
        user_id=user_data["user_id"],
        username=user_data["username"],
        email=user_data["email"],
        is_active=user_data["is_active"],
        is_superuser=user_data["is_superuser"]
    )

