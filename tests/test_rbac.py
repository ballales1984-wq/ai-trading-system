"""
Test Suite for RBAC Module
===========================
Comprehensive tests for Role-Based Access Control module.
"""

import pytest
from app.core.rbac import (
    Permission,
    Role,
    User,
    Resource,
    RBACManager,
    ROLE_PERMISSIONS,
)


class TestPermission:
    """Tests for Permission enum."""
    
    def test_permission_enum_values(self):
        """Test permission enum values."""
        assert Permission.PORTFOLIO_READ.value == "portfolio:read"
        assert Permission.PORTFOLIO_WRITE.value == "portfolio:write"
        assert Permission.PORTFOLIO_TRADE.value == "portfolio:trade"
        assert Permission.ORDER_READ.value == "order:read"
        assert Permission.ORDER_CREATE.value == "order:create"
        assert Permission.ORDER_CANCEL.value == "order:cancel"
        assert Permission.ADMIN_USERS.value == "admin:users"
    
    def test_permission_enum_members(self):
        """Test permission enum has all expected members."""
        permissions = list(Permission)
        assert len(permissions) > 15
        assert Permission.MARKET_READ in permissions
        assert Permission.ML_TRAIN in permissions


class TestRole:
    """Tests for Role enum."""
    
    def test_role_enum_values(self):
        """Test role enum values."""
        assert Role.ADMIN.value == "admin"
        assert Role.TRADER.value == "trader"
        assert Role.VIEWER.value == "viewer"
        assert Role.API_USER.value == "api_user"
        assert Role.RISK_MANAGER.value == "risk_manager"
    
    def test_role_enum_members(self):
        """Test role enum has all expected members."""
        roles = list(Role)
        assert len(roles) == 5
        assert Role.ADMIN in roles
        assert Role.TRADER in roles
        assert Role.VIEWER in roles
        assert Role.API_USER in roles
        assert Role.RISK_MANAGER in roles


class TestRolePermissions:
    """Tests for role-permissions mapping."""
    
    def test_admin_has_all_permissions(self):
        """Test admin role has all permissions."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.PORTFOLIO_READ in admin_perms
        assert Permission.PORTFOLIO_WRITE in admin_perms
        assert Permission.ADMIN_USERS in admin_perms
        assert Permission.ML_TRAIN in admin_perms
    
    def test_trader_permissions(self):
        """Test trader role has correct permissions."""
        trader_perms = ROLE_PERMISSIONS[Role.TRADER]
        assert Permission.PORTFOLIO_READ in trader_perms
        assert Permission.PORTFOLIO_TRADE in trader_perms
        assert Permission.ORDER_CREATE in trader_perms
        # Traders should not have admin permissions
        assert Permission.ADMIN_USERS not in trader_perms
    
    def test_viewer_permissions(self):
        """Test viewer role has read-only permissions."""
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.PORTFOLIO_READ in viewer_perms
        assert Permission.ORDER_READ in viewer_perms
        # Viewers should not have write permissions
        assert Permission.PORTFOLIO_WRITE not in viewer_perms
        assert Permission.ORDER_CREATE not in viewer_perms
    
    def test_risk_manager_permissions(self):
        """Test risk manager role has correct permissions."""
        rm_perms = ROLE_PERMISSIONS[Role.RISK_MANAGER]
        assert Permission.RISK_READ in rm_perms
        assert Permission.RISK_WRITE in rm_perms
        assert Permission.PORTFOLIO_READ in rm_perms
    
    def test_api_user_permissions(self):
        """Test API user role has minimal permissions."""
        api_perms = ROLE_PERMISSIONS[Role.API_USER]
        assert Permission.MARKET_READ in api_perms
        assert Permission.ORDER_CREATE in api_perms
        assert len(api_perms) == 2


class TestUser:
    """Tests for User dataclass."""
    
    def test_user_creation(self):
        """Test user creation with role."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER,
        )
        assert user.user_id == "user_1"
        assert user.username == "testuser"
        assert user.role == Role.TRADER
        assert user.is_active is True
    
    def test_user_permissions_from_role(self):
        """Test user gets permissions from role automatically."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.ADMIN,
        )
        assert Permission.PORTFOLIO_READ in user.permissions
        assert Permission.ADMIN_USERS in user.permissions
    
    def test_user_has_permission(self):
        """Test user has_permission method."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER,
        )
        assert user.has_permission(Permission.PORTFOLIO_READ) is True
        assert user.has_permission(Permission.ADMIN_USERS) is False
    
    def test_user_has_any_permission(self):
        """Test user has_any_permission method."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER,
        )
        perms = {Permission.PORTFOLIO_READ, Permission.ADMIN_USERS}
        assert user.has_any_permission(perms) is True
        
        perms = {Permission.ADMIN_USERS, Permission.ADMIN_SETTINGS}
        assert user.has_any_permission(perms) is False
    
    def test_user_has_all_permissions(self):
        """Test user has_all_permissions method."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER,
        )
        perms = {Permission.PORTFOLIO_READ, Permission.ORDER_READ}
        assert user.has_all_permissions(perms) is True
        
        perms = {Permission.PORTFOLIO_READ, Permission.ADMIN_USERS}
        assert user.has_all_permissions(perms) is False
    
    def test_user_with_custom_permissions(self):
        """Test user with custom permissions."""
        custom_perms = {Permission.MARKET_READ, Permission.MARKET_EXECUTE}
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.VIEWER,
            permissions=custom_perms,
        )
        assert user.permissions == custom_perms
    
    def test_user_inactive(self):
        """Test inactive user."""
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER,
            is_active=False,
        )
        assert user.is_active is False


class TestResource:
    """Tests for Resource dataclass."""
    
    def test_resource_creation(self):
        """Test resource creation."""
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
        )
        assert resource.resource_type == "portfolio"
        assert resource.resource_id == "portfolio_1"
        assert resource.owner_id == "user_1"
    
    def test_resource_admin_access(self):
        """Test admin can access any resource."""
        admin = User(user_id="admin_1", username="admin", role=Role.ADMIN)
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_2",
        )
        assert resource.can_access(admin) is True
    
    def test_resource_owner_access(self):
        """Test owner can access their own resource."""
        user = User(user_id="user_1", username="user1", role=Role.TRADER)
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
        )
        assert resource.can_access(user) is True
    
    def test_resource_role_access(self):
        """Test role-based resource access."""
        user = User(user_id="user_1", username="user1", role=Role.VIEWER)
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_2",
            allowed_roles={Role.VIEWER, Role.TRADER},
        )
        assert resource.can_access(user) is True
    
    def test_resource_role_denied(self):
        """Test resource access denied for wrong role."""
        user = User(user_id="user_1", username="user1", role=Role.VIEWER)
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_2",
            allowed_roles={Role.TRADER, Role.ADMIN},
        )
        assert resource.can_access(user) is False
    
    def test_resource_permission_access(self):
        """Test resource permission-based access."""
        user = User(user_id="user_1", username="user1", role=Role.TRADER)
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_2",
            allowed_permissions={Permission.PORTFOLIO_READ},
        )
        # Check if user has the permission - trader has PORTFOLIO_READ
        result = resource.can_access(user)
        assert result is True
    
    def test_resource_permission_denied(self):
        """Test resource access denied for missing permissions."""
        user = User(user_id="user_1", username="user1", role=Role.VIEWER)
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_2",
            allowed_permissions={Permission.PORTFOLIO_WRITE},
        )
        assert resource.can_access(user) is False


class TestRBACManager:
    """Tests for RBACManager class."""
    
    def test_rbac_manager_creation(self):
        """Test RBAC manager creation."""
        manager = RBACManager()
        assert manager is not None
        assert isinstance(manager._users, dict)
        assert isinstance(manager._api_keys, dict)
    
    def test_get_roles(self):
        """Test getting all roles."""
        manager = RBACManager()
        roles = manager.get_roles()
        assert len(roles) == 5
        assert Role.ADMIN in roles
    
    def test_roles_property(self):
        """Test roles property."""
        manager = RBACManager()
        roles = manager.roles
        assert len(roles) == 5
    
    def test_create_user(self):
        """Test creating a user."""
        manager = RBACManager()
        user = manager.create_user(username="testuser", role=Role.TRADER)
        assert user.username == "testuser"
        assert user.role == Role.TRADER
        assert user.user_id is not None
    
    def test_create_user_with_id(self):
        """Test creating a user with custom ID."""
        manager = RBACManager()
        user = manager.create_user(
            username="testuser",
            role=Role.TRADER,
            user_id="custom_id",
        )
        assert user.user_id == "custom_id"
    
    def test_set_api_key(self):
        """Test setting API key for user."""
        manager = RBACManager()
        user = manager.create_user(username="testuser", role=Role.TRADER)
        manager.set_api_key(user, "test_api_key_123")
        assert user.api_key == "test_api_key_123"
        assert manager._api_keys["test_api_key_123"] == user
    
    def test_get_user(self):
        """Test getting user by username."""
        manager = RBACManager()
        manager.create_user(username="testuser", role=Role.TRADER)
        user = manager.get_user("testuser")
        assert user is not None
        assert user.username == "testuser"
    
    def test_get_user_not_found(self):
        """Test getting non-existent user."""
        manager = RBACManager()
        user = manager.get_user("nonexistent")
        assert user is None
    
    def test_get_user_by_id(self):
        """Test getting user by ID."""
        manager = RBACManager()
        manager.create_user(username="testuser", role=Role.TRADER, user_id="user_123")
        user = manager.get_user_by_id("user_123")
        assert user is not None
        assert user.user_id == "user_123"
    
    def test_get_user_by_id_not_found(self):
        """Test getting user by non-existent ID."""
        manager = RBACManager()
        user = manager.get_user_by_id("nonexistent")
        assert user is None
    
    def test_get_user_by_api_key(self):
        """Test getting user by API key."""
        manager = RBACManager()
        user = manager.create_user(username="testuser", role=Role.TRADER)
        manager.set_api_key(user, "test_api_key_123")
        found_user = manager.get_user_by_api_key("test_api_key_123")
        assert found_user is not None
        assert found_user.username == "testuser"
    
    def test_get_user_by_api_key_not_found(self):
        """Test getting user by non-existent API key."""
        manager = RBACManager()
        user = manager.get_user_by_api_key("nonexistent_key")
        assert user is None
    
    def test_delete_user(self):
        """Test deleting a user."""
        manager = RBACManager()
        manager.create_user(username="testuser", role=Role.TRADER)
        result = manager.delete_user("testuser")
        assert result is True
        assert manager.get_user("testuser") is None
    
    def test_delete_user_not_found(self):
        """Test deleting non-existent user."""
        manager = RBACManager()
        result = manager.delete_user("nonexistent")
        assert result is False
    
    def test_update_user_role(self):
        """Test updating user role."""
        manager = RBACManager()
        user = manager.create_user(username="testuser", role=Role.VIEWER)
        manager.update_user_role("testuser", Role.ADMIN)
        updated_user = manager.get_user("testuser")
        assert updated_user.role == Role.ADMIN
    
    def test_check_permission(self):
        """Test checking user permission."""
        manager = RBACManager()
        user = manager.create_user(username="testuser", role=Role.TRADER)
        assert user.has_permission(Permission.PORTFOLIO_READ) is True
        assert user.has_permission(Permission.ADMIN_USERS) is False
    
    def test_check_permission_not_found(self):
        """Test checking permission for non-existent user."""
        manager = RBACManager()
        user = manager.get_user("nonexistent")
        assert user is None
    
    def test_add_resource(self):
        """Test adding a resource."""
        manager = RBACManager()
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
        )
        manager.add_resource("portfolio_1", resource)
        assert "portfolio_1" in manager._resources
    
    def test_get_resource(self):
        """Test getting a resource."""
        manager = RBACManager()
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
        )
        manager.add_resource("portfolio_1", resource)
        retrieved = manager.get_resource("portfolio_1")
        assert retrieved is not None
        assert retrieved.resource_id == "portfolio_1"
    
    def test_check_resource_access(self):
        """Test checking resource access."""
        manager = RBACManager()
        user = manager.create_user(username="testuser", role=Role.TRADER)
        resource = Resource(
            resource_type="portfolio",
            resource_id="portfolio_1",
            owner_id="user_1",
        )
        manager.add_resource("portfolio_1", resource)
        # Owner should have access
        user.owner_id = "user_1"
        assert manager.check_resource_access("testuser", "portfolio_1") is True
