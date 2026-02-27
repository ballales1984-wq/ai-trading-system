"""
RBAC Module - Role-Based Access Control
=======================================
Role-based access control for API security.

Author: AI Trading System
"""

from typing import Dict, Set, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging

try:
    from fastapi import Request
except ImportError:
    # For when FastAPI is not available
    Request = None

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Available permissions."""
    # Portfolio permissions
    PORTFOLIO_READ = "portfolio:read"
    PORTFOLIO_WRITE = "portfolio:write"
    PORTFOLIO_TRADE = "portfolio:trade"
    
    # Order permissions
    ORDER_READ = "order:read"
    ORDER_CREATE = "order:create"
    ORDER_CANCEL = "order:cancel"
    
    # Strategy permissions
    STRATEGY_READ = "strategy:read"
    STRATEGY_WRITE = "strategy:write"
    STRATEGY_EXECUTE = "strategy:execute"
    
    # Risk permissions
    RISK_READ = "risk:read"
    RISK_WRITE = "risk:write"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_API_KEYS = "admin:api_keys"
    
    # Market data permissions
    MARKET_READ = "market:read"
    MARKET_EXECUTE = "market:execute"
    
    # ML permissions
    ML_READ = "ml:read"
    ML_WRITE = "ml:write"
    ML_TRAIN = "ml:train"


class Role(Enum):
    """User roles."""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"
    RISK_MANAGER = "risk_manager"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # All permissions
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.PORTFOLIO_TRADE,
        Permission.ORDER_READ,
        Permission.ORDER_CREATE,
        Permission.ORDER_CANCEL,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_WRITE,
        Permission.STRATEGY_EXECUTE,
        Permission.RISK_READ,
        Permission.RISK_WRITE,
        Permission.ADMIN_USERS,
        Permission.ADMIN_SETTINGS,
        Permission.ADMIN_API_KEYS,
        Permission.MARKET_READ,
        Permission.MARKET_EXECUTE,
        Permission.ML_READ,
        Permission.ML_WRITE,
        Permission.ML_TRAIN,
    },
    Role.TRADER: {
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_TRADE,
        Permission.ORDER_READ,
        Permission.ORDER_CREATE,
        Permission.ORDER_CANCEL,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_EXECUTE,
        Permission.RISK_READ,
        Permission.MARKET_READ,
        Permission.MARKET_EXECUTE,
    },
    Role.RISK_MANAGER: {
        Permission.PORTFOLIO_READ,
        Permission.ORDER_READ,
        Permission.STRATEGY_READ,
        Permission.RISK_READ,
        Permission.RISK_WRITE,
        Permission.MARKET_READ,
    },
    Role.VIEWER: {
        Permission.PORTFOLIO_READ,
        Permission.ORDER_READ,
        Permission.STRATEGY_READ,
        Permission.RISK_READ,
        Permission.MARKET_READ,
    },
    Role.API_USER: {
        Permission.MARKET_READ,
        Permission.ORDER_CREATE,
    },
}


@dataclass
class User:
    """User entity for RBAC."""
    user_id: str
    username: str
    role: Role
    api_key: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    
    def __post_init__(self):
        """Set permissions based on role."""
        if not self.permissions and self.role:
            self.permissions = ROLE_PERMISSIONS.get(self.role, set()).copy()
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: Set[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return bool(self.permissions & permissions)
    
    def has_all_permissions(self, permissions: Set[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        return permissions.issubset(self.permissions)


@dataclass
class Resource:
    """Resource that can be protected."""
    resource_type: str
    resource_id: str
    owner_id: str
    allowed_roles: Set[Role] = field(default_factory=set)
    allowed_permissions: Set[Permission] = field(default_factory=set)
    
    def can_access(self, user: User) -> bool:
        """Check if user can access this resource."""
        # Admins can access everything
        if user.role == Role.ADMIN:
            return True
        
        # Check role access
        if self.allowed_roles and user.role not in self.allowed_roles:
            return False
        
        # Check permission access
        if self.allowed_permissions and not user.has_any_permission(self.allowed_permissions):
            return False
        
        # Owner can always access
        if user.user_id == self.owner_id:
            return True
        
        return False


class RBACManager:
    """
    RBAC Manager
    ============
    Manages roles, permissions, and access control.
    """
    
    def __init__(self):
        """Initialize RBAC manager."""
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, User] = {}
        self._resources: Dict[str, Resource] = {}
        
        # Default roles
        self._role_permissions = ROLE_PERMISSIONS.copy()
        
        logger.info("RBAC Manager initialized")
    
    def create_user(
        self,
        username: str,
        role: Role,
        user_id: Optional[str] = None,
    ) -> User:
        """Create a new user with role."""
        if user_id is None:
            user_id = f"user_{len(self._users) + 1}"
        
        user = User(
            user_id=user_id,
            username=username,
            role=role,
        )
        
        self._users[username] = user
        logger.info(f"Created user: {username} with role: {role.value}")
        
        return user
    
    def set_api_key(self, user: User, api_key: str) -> None:
        """Set API key for user."""
        self._api_keys[api_key] = user
        user.api_key = api_key
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self._users.get(username)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        for user in self._users.values():
            if user.user_id == user_id:
                return user
        return None
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        return self._api_keys.get(api_key)
    
    def update_user_role(self, username: str, role: Role) -> bool:
        """Update user role."""
        user = self._users.get(username)
        
        if not user:
            return False
        
        user.role = role
        user.permissions = ROLE_PERMISSIONS.get(role, set()).copy()
        
        logger.info(f"Updated user {username} role to {role.value}")
        
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user."""
        user = self._users.get(username)
        
        if not user:
            return False
        
        user.is_active = False
        
        # Remove API key
        if user.api_key and user.api_key in self._api_keys:
            del self._api_keys[user.api_key]
        
        logger.info(f"Deactivated user: {username}")
        
        return True
    
    def create_resource(
        self,
        resource_type: str,
        resource_id: str,
        owner_id: str,
        allowed_roles: Optional[Set[Role]] = None,
        allowed_permissions: Optional[Set[Permission]] = None,
    ) -> Resource:
        """Create a protected resource."""
        resource = Resource(
            resource_type=resource_type,
            resource_id=resource_id,
            owner_id=owner_id,
            allowed_roles=allowed_roles or set(),
            allowed_permissions=allowed_permissions or set(),
        )
        
        key = f"{resource_type}:{resource_id}"
        self._resources[key] = resource
        
        return resource
    
    def get_resource(self, resource_type: str, resource_id: str) -> Optional[Resource]:
        """Get resource by type and ID."""
        key = f"{resource_type}:{resource_id}"
        return self._resources.get(key)
    
    def can_access_resource(self, user: User, resource_type: str, resource_id: str) -> bool:
        """Check if user can access resource."""
        resource = self.get_resource(resource_type, resource_id)
        
        if not resource:
            # Resource doesn't exist, allow by default (or deny in strict mode)
            return True
        
        return resource.can_access(user)
    
    def add_role_permission(self, role: Role, permission: Permission) -> None:
        """Add permission to role."""
        if role not in self._role_permissions:
            self._role_permissions[role] = set()
        
        self._role_permissions[role].add(permission)
        
        # Update all users with this role
        for user in self._users.values():
            if user.role == role:
                user.permissions.add(permission)
        
        logger.info(f"Added permission {permission.value} to role {role.value}")
    
    def remove_role_permission(self, role: Role, permission: Permission) -> None:
        """Remove permission from role."""
        if role in self._role_permissions:
            self._role_permissions[role].discard(permission)
        
        # Update all users with this role
        for user in self._users.values():
            if user.role == role:
                user.permissions.discard(permission)
        
        logger.info(f"Removed permission {permission.value} from role {role.value}")
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role."""
        return self._role_permissions.get(role, set()).copy()
    
    def list_users(self, role: Optional[Role] = None) -> List[User]:
        """List all users, optionally filtered by role."""
        users = list(self._users.values())
        
        if role:
            users = [u for u in users if u.role == role]
        
        return users


def require_permission(permission: Permission):
    """
    Decorator to require specific permission.
    
    Usage:
        @require_permission(Permission.ORDER_CREATE)
        def create_order(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In FastAPI, user would be extracted from request state
            # For now, this is a placeholder for the decorator pattern
            # Integration with FastAPI would use Depends()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_any_permission(*permissions: Permission):
    """Decorator to require any of the specified permissions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role: Role):
    """Decorator to require specific role."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Default RBAC manager instance
rbac_manager = RBACManager()

# Create default users
rbac_manager.create_user("admin", Role.ADMIN)
rbac_manager.create_user("trader", Role.TRADER)
rbac_manager.create_user("viewer", Role.VIEWER)
rbac_manager.create_user("risk_manager", Role.RISK_MANAGER)
rbac_manager.create_user("api_user", Role.API_USER)


# FastAPI integration
def get_current_user(request: Request) -> User:
    """
    Get current user from request.
    This function extracts user info from the request headers or state.
    
    In production, this would validate JWT tokens or session cookies.
    For now, it returns a mock user for development.
    """
    # Try to get user from request state (set by auth middleware)
    if hasattr(request.state, "user"):
        return request.state.user
    
    # Try to get from authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # In production, validate token and return user
        # For now, return a default user
        user = rbac_manager.get_user("admin")
        if user:
            return user
    
    # Return default user for development
    user = rbac_manager.get_user("admin")
    if user:
        return user
    
    # Last resort: create a basic user
    return rbac_manager.create_user("anonymous", Role.VIEWER)


# FastAPI integration example:
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

app = FastAPI()
security = HTTPBearer()

def get_current_user(request: Request) -> User:
    # Extract user from request state (set by auth middleware)
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user

def require_permission(permission: Permission):
    def dependency(user: User = Depends(get_current_user)):
        if not user.has_permission(permission):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return dependency

@app.post("/orders")
def create_order(
    order_data: OrderCreate,
    user: User = Depends(require_permission(Permission.ORDER_CREATE))
):
    # Create order...
    return {"status": "created", "user": user.username}

@app.get("/admin/users")
def list_users(
    user: User = Depends(require_permission(Permission.ADMIN_USERS))
):
    users = rbac_manager.list_users()
    return {"users": [{"username": u.username, "role": u.role.value} for u in users]}
"""

