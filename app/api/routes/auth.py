"""
Authentication Routes
====================
Email/password login and JWT token generation.

This module provides:
- User login with email/password
- JWT access token generation
- Token refresh functionality
- Protected route testing
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

from app.core.security import jwt_manager, User, UserRole

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer(auto_error=False)


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request with email and password."""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response with tokens."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds
    user: dict


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str
    username: Optional[str] = None


class RegisterResponse(BaseModel):
    """Registration response."""
    message: str
    user_id: str
    username: str


class TokenPayloadResponse(BaseModel):
    """Token payload information."""
    username: str
    role: str
    exp: int


# Dependency for getting current user
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    This dependency can be used to protect routes.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = jwt_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = jwt_manager.get_user_from_token(token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return user


# Login endpoint
@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login with email and password.
    
    Returns JWT access token for authenticated requests.
    """
    # Use email as username for authentication
    # The security module uses username, so we'll use the email part before @
    username = request.email.split('@')[0]
    
    # Authenticate user
    user = jwt_manager.authenticate(username, request.password)
    
    if not user:
        logger.warning(f"Failed login attempt for: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = jwt_manager.create_access_token(user)
    refresh_token = jwt_manager.create_refresh_token(user)
    
    logger.info(f"User logged in: {user.username} ({user.role.value})")
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=jwt_manager.config.access_token_expire_minutes * 60,
        user={
            "user_id": user.user_id,
            "username": user.username,
            "email": request.email,
            "role": user.role.value,
        }
    )


# Register endpoint (for new users)
@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    """
    Register a new user.
    
    Creates a new user with VIEWER role by default.
    """
    # Use provided username or derive from email
    username = request.username or request.email.split('@')[0]
    
    # Check if user already exists
    existing_user = jwt_manager._users.get(username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Create new user with TRADER role (can be changed to VIEWER)
    user = jwt_manager.create_user(
        username=username,
        password=request.password,
        role=UserRole.TRADER
    )
    
    logger.info(f"New user registered: {username}")
    
    return RegisterResponse(
        message="User registered successfully",
        user_id=user.user_id,
        username=user.username
    )


# Refresh token endpoint
@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    """
    # Verify refresh token
    payload = jwt_manager.verify_token(request.refresh_token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Get user
    user = jwt_manager.get_user_from_token(request.refresh_token)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token = jwt_manager.create_access_token(user)
    refresh_token = jwt_manager.create_refresh_token(user)
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=jwt_manager.config.access_token_expire_minutes * 60,
        user={
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
        }
    )


# Protected test endpoint
@router.get("/me", response_model=dict)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    
    Requires valid JWT token.
    """
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "role": current_user.role.value,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
    }


# Verify token endpoint
@router.get("/verify")
async def verify_token(current_user: User = Depends(get_current_user)):
    """
    Verify if current token is valid.
    
    Returns token payload information.
    """
    return {
        "valid": True,
        "username": current_user.username,
        "role": current_user.role.value,
    }


# Logout endpoint (client-side token removal, server-side can add to blacklist)
@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout current user.
    
    In a production system, you would add the token to a blacklist.
    """
    logger.info(f"User logged out: {current_user.username}")
    
    return {
        "message": "Successfully logged out",
        "username": current_user.username
    }


# Health check for auth service
@router.get("/health")
async def auth_health():
    """Health check for authentication service."""
    return {
        "status": "healthy",
        "service": "authentication",
        "users_count": len(jwt_manager._users)
    }
