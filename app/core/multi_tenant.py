"""
Multi-Tenant Account Management Module
=====================================
Handles multi-user account management with role-based access control.

Author: AI Trading System
Version: 1.0.0
"""

import uuid
import hashlib
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum


class UserRole(Enum):
    """User roles in the system"""
    ADMIN = "admin"
    MANAGER = "manager"
    TRADER = "trader"
    VIEWER = "viewer"


class AccountStatus(Enum):
    """Account status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    CLOSED = "closed"


@dataclass
class User:
    """User account"""
    user_id: str
    username: str
    email: str
    role: UserRole
    status: AccountStatus
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    api_keys: List[str] = field(default_factory=list)
    sub_accounts: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "sub_accounts": self.sub_accounts,
            "metadata": self.metadata
        }


@dataclass
class SubAccount:
    """Sub-account for separate trading accounts"""
    sub_account_id: str
    parent_user_id: str
    name: str
    initial_balance: float
    current_balance: float
    status: AccountStatus
    created_at: datetime = field(default_factory=datetime.now)


class MultiTenantManager:
    """
    Multi-tenant account management system.
    Supports:
    - User registration and authentication
    - Role-based access control (RBAC)
    - Sub-account management
    - API key management
    """
    
    def __init__(self):
        """Initialize multi-tenant manager"""
        self._users: Dict[str, User] = {}
        self._sub_accounts: Dict[str, SubAccount] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> user_id
        self._username_index: Dict[str, str] = {}  # username -> user_id
        self._email_index: Dict[str, str] = {}  # email -> user_id
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER
    ) -> User:
        """
        Create a new user account.
        
        Args:
            username: Unique username
            email: User email
            password: User password (will be hashed)
            role: User role
            
        Returns:
            Created User object
        """
        # Check uniqueness
        if username in self._username_index:
            raise ValueError(f"Username {username} already exists")
        if email in self._email_index:
            raise ValueError(f"Email {email} already exists")
        
        # Generate user ID
        user_id = str(uuid.uuid4())
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            status=AccountStatus.PENDING,
            metadata={"password_hash": password_hash}
        )
        
        # Store user
        self._users[user_id] = user
        self._username_index[username] = user_id
        self._email_index[email] = user_id
        
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user.
        
        Args:
            username: Username or email
            password: Plain text password
            
        Returns:
            User if authenticated, None otherwise
        """
        # Find user by username or email
        user_id = self._username_index.get(username) or self._email_index.get(username)
        
        if not user_id:
            return None
        
        user = self._users.get(user_id)
        
        if not user:
            return None
        
        # Check status
        if user.status == AccountStatus.SUSPENDED:
            return None
        if user.status == AccountStatus.CLOSED:
            return None
        
        # Verify password
        password_hash = self._hash_password(password)
        if user.metadata.get("password_hash") != password_hash:
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        # Activate pending users
        if user.status == AccountStatus.PENDING:
            user.status = AccountStatus.ACTIVE
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_id = self._username_index.get(username)
        return self._users.get(user_id) if user_id else None
    
    def update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        role: Optional[UserRole] = None,
        status: Optional[AccountStatus] = None
    ) -> Optional[User]:
        """
        Update user information.
        
        Args:
            user_id: User ID
            email: New email (optional)
            role: New role (optional)
            status: New status (optional)
            
        Returns:
            Updated User or None if not found
        """
        user = self._users.get(user_id)
        
        if not user:
            return None
        
        if email and email != user.email:
            if email in self._email_index:
                raise ValueError(f"Email {email} already in use")
            del self._email_index[user.email]
            user.email = email
            self._email_index[email] = user_id
        
        if role:
            user.role = role
        
        if status:
            user.status = status
        
        user.updated_at = datetime.now()
        
        return user
    
    def suspend_user(self, user_id: str) -> bool:
        """Suspend a user account"""
        user = self._users.get(user_id)
        if not user:
            return False
        user.status = AccountStatus.SUSPENDED
        user.updated_at = datetime.now()
        return True
    
    def activate_user(self, user_id: str) -> bool:
        """Activate a user account"""
        user = self._users.get(user_id)
        if not user:
            return False
        user.status = AccountStatus.ACTIVE
        user.updated_at = datetime.now()
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user account"""
        user = self._users.get(user_id)
        if not user:
            return False
        
        # Remove from indexes
        del self._username_index[user.username]
        del self._email_index[user.email]
        
        # Remove API keys
        for api_key in user.api_keys:
            del self._api_keys[api_key]
        
        # Remove user
        del self._users[user_id]
        
        return True
    
    def create_api_key(self, user_id: str, name: str = "default") -> str:
        """
        Create API key for user.
        
        Args:
            user_id: User ID
            name: Name/description for the API key
            
        Returns:
            Generated API key
        """
        user = self._users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Generate API key
        api_key = f"ats_{uuid.uuid4().hex}_{uuid.uuid4().hex[:8]}"
        
        # Store mapping
        self._api_keys[api_key] = user_id
        
        # Add to user
        user.api_keys.append(api_key)
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """
        Verify API key and return associated user.
        
        Args:
            api_key: API key to verify
            
        Returns:
            User if valid, None otherwise
        """
        user_id = self._api_keys.get(api_key)
        
        if not user_id:
            return None
        
        user = self._users.get(user_id)
        
        if not user or user.status != AccountStatus.ACTIVE:
            return None
        
        return user
    
    def revoke_api_key(self, user_id: str, api_key: str) -> bool:
        """Revoke an API key"""
        user = self._users.get(user_id)
        
        if not user or api_key not in user.api_keys:
            return False
        
        user.api_keys.remove(api_key)
        del self._api_keys[api_key]
        
        return True
    
    def create_sub_account(
        self,
        parent_user_id: str,
        name: str,
        initial_balance: float = 0.0
    ) -> Optional[SubAccount]:
        """
        Create a sub-account for a user.
        
        Args:
            parent_user_id: Parent user ID
            name: Sub-account name
            initial_balance: Initial balance
            
        Returns:
            Created SubAccount or None if parent not found
        """
        parent = self._users.get(parent_user_id)
        
        if not parent:
            return None
        
        # Generate sub-account ID
        sub_account_id = f"sub_{uuid.uuid4().hex[:12]}"
        
        # Create sub-account
        sub_account = SubAccount(
            sub_account_id=sub_account_id,
            parent_user_id=parent_user_id,
            name=name,
            initial_balance=initial_balance,
            current_balance=initial_balance,
            status=AccountStatus.ACTIVE
        )
        
        # Store
        self._sub_accounts[sub_account_id] = sub_account
        
        # Add to parent
        parent.sub_accounts.append(sub_account_id)
        
        return sub_account
    
    def get_sub_account(self, sub_account_id: str) -> Optional[SubAccount]:
        """Get sub-account by ID"""
        return self._sub_accounts.get(sub_account_id)
    
    def list_users(
        self,
        role: Optional[UserRole] = None,
        status: Optional[AccountStatus] = None
    ) -> List[User]:
        """
        List users with optional filters.
        
        Args:
            role: Filter by role
            status: Filter by status
            
        Returns:
            List of users
        """
        users = list(self._users.values())
        
        if role:
            users = [u for u in users if u.role == role]
        
        if status:
            users = [u for u in users if u.status == status]
        
        return users
    
    def check_permission(
        self,
        user: User,
        required_role: UserRole
    ) -> bool:
        """
        Check if user has required role.
        
        Args:
            user: User to check
            required_role: Required role
            
        Returns:
            True if user has permission
        """
        # Role hierarchy: ADMIN > MANAGER > TRADER > VIEWER
        role_hierarchy = {
            UserRole.ADMIN: 4,
            UserRole.MANAGER: 3,
            UserRole.TRADER: 2,
            UserRole.VIEWER: 1
        }
        
        return role_hierarchy.get(user.role, 0) >= role_hierarchy.get(required_role, 0)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()


# Demo
def run_demo():
    """Demo function showing multi-tenant management."""
    manager = MultiTenantManager()
    
    # Create admin
    admin = manager.create_user(
        username="admin",
        email="admin@aitrading.com",
        password="admin123",
        role=UserRole.ADMIN
    )
    print(f"Created admin: {admin.username} (ID: {admin.user_id})")
    
    # Create trader
    trader = manager.create_user(
        username="trader1",
        email="trader1@aitrading.com",
        password="trader123",
        role=UserRole.TRADER
    )
    print(f"Created trader: {trader.username} (ID: {trader.user_id})")
    
    # Authenticate
    authenticated = manager.authenticate("trader1", "trader123")
    print(f"Authentication: {'Success' if authenticated else 'Failed'}")
    
    # Create API key
    api_key = manager.create_api_key(trader.user_id, "Trading Bot")
    print(f"Created API key: {api_key[:20]}...")
    
    # Verify API key
    user_from_key = manager.verify_api_key(api_key)
    print(f"API key verification: {user_from_key.username if user_from_key else 'Failed'}")
    
    # Create sub-account
    sub = manager.create_sub_account(trader.user_id, "Main Trading Account", 10000.0)
    print(f"Created sub-account: {sub.name} (ID: {sub.sub_account_id})")
    
    # Check permissions
    can_manage = manager.check_permission(trader, UserRole.MANAGER)
    print(f"Trader can manage: {can_manage}")


if __name__ == "__main__":
    run_demo()
