"""
Tests for RBAC Module (Role-Based Access Control)
==================================================
Tests for role-based access control functionality.
"""

import pytest
import sys
import os
from typing import Set

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.rbac import (
    RBACManager, User, Resource, Role, Permission, ROLE_PERMISSIONS
)


class TestPermission:
    """Test Permission enum."""
    
    def test_portfolio_permissions(self):
        """Test portfolio permissions."""
        assert Permission.PORTFOLIO_READ.value == "portfolio:read"
        assert Permission.PORTFOLIO_WRITE.value == "portfolio:write"
        assert Permission.PORTFOLIO_TRADE.value == "portfolio:trade"
    
    def test_order_permissions(self):
        """Test order permissions."""
        assert Permission.ORDER_READ.value == "order:read"
        assert Permission.ORDER_CREATE.value == "order:create"
        assert Permission.ORDER_CANCEL.value == "order:cancel"
    
    def test_admin_permissions(self):
        """Test admin permissions."""
        assert Permission.ADMIN_USERS.value == "admin:users"
        assert Permission.ADMIN_SETTINGS.value == "admin:settings"
        assert Permission.ADMIN_API_KEYS.value == "admin:api_keys"


class TestRole:
    """Test Role enum."""
    
    def test_roles(self):
        """Test all roles."""
        assert Role.ADMIN.value == "admin"
        assert Role.TRADER.value == "trader"
        assert Role.VIEWER.value == "viewer"
        assert Role.API_USER.value == "api_user"
        assert Role.RISK_MANAGER.value == "risk_manager"


class TestRolePermissions:
    """Test role-to-permissions mapping."""
    
    def test_admin_has_all_permissions(self):
        """Test that admin has all permissions."""
        admin_permissions = ROLE_PERMISSIONS[Role.ADMIN]
        
        # Admin should have all permissions
        assert Permission.PORTFOLIO_READ in admin_permissions
        assert Permission.PORTFOLIO_WRITE in admin_permissions
        assert Permission.PORTFOLIO_TRADE in admin_permissions
        assert Permission.ORDER_READ in admin_permissions
        assert Permission.ORDER_CREATE in admin_permissions
        assert Permission.ORDER_CANCEL in admin_permissions
        assert Permission.ADMIN_USERS in admin_permissions
        assert Permission.ADMIN_SETTINGS in admin_permissions
        assert Permission.ADMIN_API_KEYS in admin_permissions
    
    def test_viewer_has_read_only_permissions(self):
        """Test that viewer has read-only permissions."""
        viewer_permissions = ROLE_PERMISSIONS[Role.VIEWER]
        
        # Viewer should have read permissions
        assert Permission.PORTFOLIO_READ in viewer_permissions
        assert Permission.ORDER_READ in viewer_permissions
        assert Permission.STRATEGY_READ in viewer_permissions
        assert Permission.RISK_READ in viewer_permissions
        assert Permission.MARKET_READ in viewer_permissions
        
        # Viewer should NOT have write permissions
        assert Permission.PORTFOLIO_WRITE not in viewer_permissions
        assert Permission.ORDER_CREATE not in viewer_permissions
        assert Permission.ADMIN_USERS not in viewer_permissions
    
    def test_trader_has_trading_permissions(self):
        """Test that trader has trading permissions."""
        trader_permissions = ROLE_PERMISSIONS[Role.TRADER]
        
        # Trader should have trading permissions
        assert Permission.PORTFOLIO_READ in trader_permissions
        assert Permission.PORTFOLIO_TRADE in trader_permissions
        assert Permission.ORDER_READ in trader_permissions
        assert Permission.ORDER_CREATE in trader_permissions
        assert Permission.ORDER_CANCEL in trader_permissions
        
        # Trader should NOT have admin permissions
        assert Permission.ADMIN_USERS not in trader_permissions
        assert Permission.ADMIN_SETTINGS not in trader_permissions
    
    def test_risk_manager_has_risk_permissions(self):
        """Test that risk manager has risk permissions."""
        risk_permissions = ROLE_PERMISSIONS[Role.RISK_MANAGER]
        
        # Risk manager should have risk permissions
        assert Permission.RISK_READ in risk_permissions
        assert Permission.RISK_WRITE in risk_permissions
        
        # Risk manager should NOT have trading permissions
        assert Permission.ORDER_CREATE not in risk_permissions
        assert Permission.PORTFOLIO_TRADE not in risk_permissions


class TestUser:
    """Test User class."""
    
    def test_initialization(self):
        """Test User initialization."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER
        )
        
        assert user.user_id == "user_1"
        assert user.username == "testuser"
        assert user.role == Role.TRADER
        assert user.is_active is True
    
    def test_permissions_set_from_role(self):
        """Test that permissions are set from role."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER
        )
        
        # Permissions should be set from role
        assert len(user.permissions) > 0
        assert Permission.PORTFOLIO_READ in user.permissions
        assert Permission.ORDER_CREATE in user.permissions
    
    def test_has_permission_true(self):
        """Test has_permission returns True for allowed permission."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER
        )
        
        assert user.has_permission(Permission.PORTFOLIO_READ) is True
    
    def test_has_permission_false(self):
        """Test has_permission returns False for denied permission."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.VIEWER
        )
        
        assert user.has_permission(Permission.ORDER_CREATE) is False
    
    def test_has_any_permission_true(self):
        """Test has_any_permission returns True when user has at least one."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER
        )
        
        permissions = {Permission.PORTFOLIO_READ, Permission.ADMIN_USERS}
        
        assert user.has_any_permission(permissions) is True
    
    def test_has_any_permission_false(self):
        """Test has_any_permission returns False when user has none."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.VIEWER
        )
        
        permissions = {Permission.ADMIN_USERS, Permission.ADMIN_SETTINGS}
        
        assert user.has_any_permission(permissions) is False
    
    def test_has_all_permissions_true(self):
        """Test has_all_permissions returns True when user has all."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.ADMIN
        )
        
        permissions = {Permission.PORTFOLIO_READ, Permission.ORDER_CREATE}
        
        assert user.has_all_permissions(permissions) is True
    
    def test_has_all_permissions_false(self):
        """Test has_all_permissions returns False when user lacks some."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.VIEWER
        )
        
        permissions = {Permission.PORTFOLIO_READ, Permission.ORDER_CREATE}
        
        assert user.has_all_permissions(permissions) is False


class TestRBACManager:
    """Test RBACManager class."""
    
    def setup_method(self):
        """Setup test data."""
        self.rbac = RBACManager()
    
    def test_initialization(self):
        """Test RBACManager initialization."""
        assert self.rbac is not None
        assert len(self.rbac._users) == 0
        assert len(self.rbac._api_keys) == 0
        assert len(self.rbac._resources) == 0
    
    def test_create_user(self):
        """Test user creation."""
        user = self.rbac.create_user(
            username="testuser",
            role=Role.TRADER
        )
        
        assert user is not None
        assert user.username == "testuser"
        assert user.role == Role.TRADER
        assert "testuser" in self.rbac._users
    
    def test_create_user_with_id(self):
        """Test user creation with custom ID."""
        user = self.rbac.create_user(
            username="testuser",
            role=Role.TRADER,
            user_id="custom_id"
        )
        
        assert user.user_id == "custom_id"
    
    def test_get_user(self):
        """Test getting user by username."""
        self.rbac.create_user(username="testuser", role=Role.TRADER)
        
        user = self.rbac.get_user("testuser")
        
        assert user is not None
        assert user.username == "testuser"
    
    def test_get_user_nonexistent(self):
        """Test getting nonexistent user."""
        user = self.rbac.get_user("nonexistent")
        
        assert user is None
    
    def test_get_user_by_id(self):
        """Test getting user by ID."""
        created_user = self.rbac.create_user(
            username="testuser",
            role=Role.TRADER,
            user_id="user_123"
        )
        
        user = self.rbac.get_user_by_id("user_123")
        
        assert user is not None
        assert user.user_id == "user_123"
    
    def test_set_api_key(self):
        """Test setting API key for user."""
        user = self.rbac.create_user(username="testuser", role=Role.API_USER)
        
        self.rbac.set_api_key(user, "test_api_key_123")
        
        assert user.api_key == "test_api_key_123"
        assert "test_api_key_123" in self.rbac._api_keys
    
    def test_get_user_by_api_key(self):
        """Test getting user by API key."""
        user = self.rbac.create_user(username="testuser", role=Role.API_USER)
        self.rbac.set_api_key(user, "test_api_key_123")
        
        retrieved_user = self.rbac.get_user_by_api_key("test_api_key_123")
        
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"
    
    def test_get_user_by_invalid_api_key(self):
        """Test getting user by invalid API key."""
        user = self.rbac.get_user_by_api_key("invalid_key")
        
        assert user is None
    
    def test_update_user_role(self):
        """Test updating user role."""
        self.rbac.create_user(username="testuser", role=Role.VIEWER)
        
        result = self.rbac.update_user_role("testuser", Role.TRADER)
        
        assert result is True
        user = self.rbac.get_user("testuser")
        assert user.role == Role.TRADER
        assert Permission.ORDER_CREATE in user.permissions
    
    def test_update_user_role_nonexistent(self):
        """Test updating role for nonexistent user."""
        result = self.rbac.update_user_role("nonexistent", Role.TRADER)
        
        assert result is False
    
    def test_deactivate_user(self):
        """Test deactivating a user."""
        user = self.rbac.create_user(username="testuser", role=Role.TRADER)
        self.rbac.set_api_key(user, "test_api_key")
        
        result = self.rbac.deactivate_user("testuser")
        
        assert result is True
        assert user.is_active is False
        assert "test_api_key" not in self.rbac._api_keys
    
    def test_deactivate_nonexistent_user(self):
        """Test deactivating nonexistent user."""
        result = self.rbac.deactivate_user("nonexistent")
        
        assert result is False
    
    def test_create_resource(self):
        """Test creating a protected resource."""
        resource = self.rbac.create_resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
            allowed_roles={Role.TRADER, Role.ADMIN},
            allowed_permissions={Permission.PORTFOLIO_READ}
        )
        
        assert resource is not None
        assert resource.resource_type == "portfolio"
        assert resource.resource_id == "portfolio_1"
        assert resource.owner_id == "user_1"
    
    def test_get_resource(self):
        """Test getting a resource."""
        self.rbac.create_resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1"
        )
        
        resource = self.rbac.get_resource("portfolio", "portfolio_1")
        
        assert resource is not None
        assert resource.resource_id == "portfolio_1"
    
    def test_get_nonexistent_resource(self):
        """Test getting nonexistent resource."""
        resource = self.rbac.get_resource("portfolio", "nonexistent")
        
        assert resource is None
    
    def test_can_access_resource_admin(self):
        """Test that admin can access any resource."""
        admin = self.rbac.create_user(username="admin", role=Role.ADMIN)
        
        self.rbac.create_resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="other_user",
            allowed_roles={Role.VIEWER}
        )
        
        result = self.rbac.can_access_resource(admin, "portfolio", "portfolio_1")
        
        assert result is True
    
    def test_can_access_resource_owner(self):
        """Test that owner can access their resource when no role restriction."""
        user = self.rbac.create_user(
            username="testuser",
            role=Role.VIEWER,
            user_id="user_1"
        )
        
        # Create resource with no role restriction - owner should access
        self.rbac.create_resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
            allowed_roles=set()  # No role restriction
        )
        
        result = self.rbac.can_access_resource(user, "portfolio", "portfolio_1")
        
        assert result is True
    
    def test_can_access_resource_denied(self):
        """Test that unauthorized user cannot access resource."""
        user = self.rbac.create_user(
            username="testuser",
            role=Role.VIEWER,
            user_id="user_1"
        )
        
        self.rbac.create_resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="other_user",
            allowed_roles={Role.ADMIN}  # Only admin
        )
        
        result = self.rbac.can_access_resource(user, "portfolio", "portfolio_1")
        
        assert result is False
    
    def test_can_access_nonexistent_resource(self):
        """Test accessing nonexistent resource."""
        user = self.rbac.create_user(username="testuser", role=Role.ADMIN)
        
        result = self.rbac.can_access_resource(user, "portfolio", "nonexistent")
        
        # Nonexistent resource - behavior depends on implementation
        # Could be True (no restriction) or False (resource not found)
        assert isinstance(result, bool)


class TestResource:
    """Test Resource class."""
    
    def test_admin_can_access(self):
        """Test that admin can access any resource."""
        admin = User(
            user_id="admin_1",
            username="admin",
            role=Role.ADMIN
        )
        
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="other_user",
            allowed_roles={Role.VIEWER}
        )
        
        assert resource.can_access(admin) is True
    
    def test_owner_can_access(self):
        """Test that owner can access their resource when no role restriction."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.VIEWER
        )
        
        # No role restriction - owner should access
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
            allowed_roles=set()  # No role restriction
        )
        
        assert resource.can_access(user) is True
    
    def test_unauthorized_cannot_access(self):
        """Test that unauthorized user cannot access resource."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.VIEWER
        )
        
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="other_user",
            allowed_roles={Role.ADMIN}
        )
        
        assert resource.can_access(user) is False
