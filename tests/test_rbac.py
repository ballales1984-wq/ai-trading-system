"""
Tests for RBAC Module
====================
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPermission:
    """Test Permission enum."""
    
    def test_permission_values(self):
        """Test Permission enum values."""
        from app.core.rbac import Permission
        
        assert Permission.PORTFOLIO_READ.value == "portfolio:read"
        assert Permission.PORTFOLIO_WRITE.value == "portfolio:write"
        assert Permission.ORDER_READ.value == "order:read"
        assert Permission.ORDER_CREATE.value == "order:create"
        assert Permission.ADMIN_USERS.value == "admin:users"


class TestRole:
    """Test Role enum."""
    
    def test_role_values(self):
        """Test Role enum values."""
        from app.core.rbac import Role
        
        assert Role.ADMIN.value == "admin"
        assert Role.TRADER.value == "trader"
        assert Role.VIEWER.value == "viewer"
        assert Role.API_USER.value == "api_user"
        assert Role.RISK_MANAGER.value == "risk_manager"


class TestRBACUser:
    """Test RBAC User class."""
    
    def test_user_creation(self):
        """Test creating a user."""
        from app.core.rbac import User, Role
        
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER
        )
        
        assert user.user_id == "user_1"
        assert user.username == "testuser"
        assert user.role == Role.TRADER
    
    def test_user_permissions(self):
        """Test user has permissions from role."""
        from app.core.rbac import User, Role, Permission
        
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.TRADER
        )
        
        # User should have permissions based on role
        assert user.role == Role.TRADER
        assert len(user.permissions) > 0
    
    def test_user_has_permission(self):
        """Test user has specific permission."""
        from app.core.rbac import User, Role, Permission
        
        user = User(
            user_id="user_1",
            username="testuser",
            role=Role.ADMIN
        )
        
        # Admin should have portfolio read permission
        assert user.has_permission(Permission.PORTFOLIO_READ) is True


class TestRolePermissions:
    """Test role permissions mapping."""
    
    def test_admin_has_all_permissions(self):
        """Test admin role has all permissions."""
        from app.core.rbac import ROLE_PERMISSIONS, Role, Permission
        
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        
        assert Permission.PORTFOLIO_READ in admin_perms
        assert Permission.PORTFOLIO_WRITE in admin_perms
        assert Permission.ORDER_CREATE in admin_perms
    
    def test_trader_has_trade_permissions(self):
        """Test trader role has trade permissions."""
        from app.core.rbac import ROLE_PERMISSIONS, Role, Permission
        
        trader_perms = ROLE_PERMISSIONS[Role.TRADER]
        
        assert Permission.PORTFOLIO_READ in trader_perms
        assert Permission.ORDER_CREATE in trader_perms
    
    def test_viewer_has_read_only(self):
        """Test viewer role has read only permissions."""
        from app.core.rbac import ROLE_PERMISSIONS, Role, Permission
        
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        
        assert Permission.PORTFOLIO_READ in viewer_perms
        assert Permission.ORDER_READ in viewer_perms


class TestRBACManager:
    """Test RBACManager class."""
    
    def test_rbac_manager_creation(self):
        """Test creating RBAC manager."""
        from app.core.rbac import RBACManager
        
        rbac = RBACManager()
        
        assert rbac is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
