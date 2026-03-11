"""
Tests for Multi-Tenant Account Management Module
"""

import pytest
from app.core.multi_tenant import (
    MultiTenantManager,
    UserRole,
    AccountStatus,
    User,
    SubAccount
)


class TestMultiTenantManager:
    """Test suite for MultiTenantManager."""
    
    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        return MultiTenantManager()
    
    @pytest.fixture
    def admin_user(self, manager):
        """Create admin user."""
        return manager.create_user(
            username="admin",
            email="admin@test.com",
            password="admin123",
            role=UserRole.ADMIN
        )
    
    @pytest.fixture
    def trader_user(self, manager):
        """Create trader user."""
        return manager.create_user(
            username="trader",
            email="trader@test.com",
            password="trader123",
            role=UserRole.TRADER
        )
    
    def test_create_user(self, manager):
        """Test user creation."""
        user = manager.create_user(
            username="testuser",
            email="test@test.com",
            password="password123",
            role=UserRole.VIEWER
        )
        
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@test.com"
        assert user.role == UserRole.VIEWER
        assert user.status == AccountStatus.PENDING
    
    def test_create_duplicate_username(self, manager):
        """Test duplicate username raises error."""
        manager.create_user(
            username="duplicate",
            email="first@test.com",
            password="password",
            role=UserRole.VIEWER
        )
        
        with pytest.raises(ValueError, match="already exists"):
            manager.create_user(
                username="duplicate",
                email="second@test.com",
                password="password",
                role=UserRole.VIEWER
            )
    
    def test_create_duplicate_email(self, manager):
        """Test duplicate email raises error."""
        manager.create_user(
            username="first",
            email="same@test.com",
            password="password",
            role=UserRole.VIEWER
        )
        
        with pytest.raises(ValueError, match="already exists"):
            manager.create_user(
                username="second",
                email="same@test.com",
                password="password",
                role=UserRole.VIEWER
            )
    
    def test_authenticate_success(self, manager, trader_user):
        """Test successful authentication."""
        user = manager.authenticate("trader", "trader123")
        
        assert user is not None
        assert user.username == "trader"
    
    def test_authenticate_wrong_password(self, manager, trader_user):
        """Test authentication with wrong password."""
        user = manager.authenticate("trader", "wrongpassword")
        
        assert user is None
    
    def test_authenticate_nonexistent(self, manager):
        """Test authentication with nonexistent user."""
        user = manager.authenticate("nonexistent", "password")
        
        assert user is None
    
    def test_get_user(self, manager, admin_user):
        """Test get user by ID."""
        user = manager.get_user(admin_user.user_id)
        
        assert user is not None
        assert user.username == "admin"
    
    def test_get_user_by_username(self, manager, trader_user):
        """Test get user by username."""
        user = manager.get_user_by_username("trader")
        
        assert user is not None
        assert user.email == "trader@test.com"
    
    def test_update_user(self, manager, trader_user):
        """Test user update."""
        updated = manager.update_user(
            trader_user.user_id,
            email="newemail@test.com",
            role=UserRole.MANAGER
        )
        
        assert updated is not None
        assert updated.email == "newemail@test.com"
        assert updated.role == UserRole.MANAGER
    
    def test_suspend_user(self, manager, trader_user):
        """Test suspending user."""
        result = manager.suspend_user(trader_user.user_id)
        
        assert result is True
        assert trader_user.status == AccountStatus.SUSPENDED
    
    def test_activate_user(self, manager, trader_user):
        """Test activating user."""
        manager.suspend_user(trader_user.user_id)
        
        result = manager.activate_user(trader_user.user_id)
        
        assert result is True
        assert trader_user.status == AccountStatus.ACTIVE
    
    def test_delete_user(self, manager, trader_user):
        """Test deleting user."""
        user_id = trader_user.user_id
        
        result = manager.delete_user(user_id)
        
        assert result is True
        assert manager.get_user(user_id) is None
    
    def test_create_api_key(self, manager, trader_user):
        """Test creating API key."""
        api_key = manager.create_api_key(trader_user.user_id, "Test Key")
        
        assert api_key is not None
        assert api_key.startswith("ats_")
        assert api_key in trader_user.api_keys
    
    def test_verify_api_key(self, manager, trader_user):
        """Test verifying API key."""
        # Activate user first (API keys only work for active users)
        manager.activate_user(trader_user.user_id)
        
        api_key = manager.create_api_key(trader_user.user_id)
        
        user = manager.verify_api_key(api_key)
        
        assert user is not None
        assert user.user_id == trader_user.user_id
    
    def test_verify_invalid_api_key(self, manager):
        """Test verifying invalid API key."""
        user = manager.verify_api_key("invalid_key_123")
        
        assert user is None
    
    def test_revoke_api_key(self, manager, trader_user):
        """Test revoking API key."""
        api_key = manager.create_api_key(trader_user.user_id)
        
        result = manager.revoke_api_key(trader_user.user_id, api_key)
        
        assert result is True
        assert api_key not in trader_user.api_keys
    
    def test_create_sub_account(self, manager, trader_user):
        """Test creating sub-account."""
        sub = manager.create_sub_account(
            trader_user.user_id,
            "Trading Account",
            10000.0
        )
        
        assert sub is not None
        assert sub.name == "Trading Account"
        assert sub.initial_balance == 10000.0
        assert sub.parent_user_id == trader_user.user_id
        assert sub.sub_account_id in trader_user.sub_accounts
    
    def test_list_users(self, manager):
        """Test listing users."""
        manager.create_user("user1", "user1@test.com", "pass", UserRole.TRADER)
        manager.create_user("user2", "user2@test.com", "pass", UserRole.VIEWER)
        manager.create_user("admin", "admin@test.com", "pass", UserRole.ADMIN)
        
        all_users = manager.list_users()
        assert len(all_users) >= 3
        
        traders = manager.list_users(role=UserRole.TRADER)
        assert len(traders) >= 1
        assert all(u.role == UserRole.TRADER for u in traders)
    
    def test_check_permission(self, manager, admin_user, trader_user):
        """Test permission checking."""
        # Admin can do everything
        assert manager.check_permission(admin_user, UserRole.ADMIN) is True
        assert manager.check_permission(admin_user, UserRole.TRADER) is True
        
        # Trader cannot be manager
        assert manager.check_permission(trader_user, UserRole.MANAGER) is False
        # But can trade
        assert manager.check_permission(trader_user, UserRole.TRADER) is True
        # And view
        assert manager.check_permission(trader_user, UserRole.VIEWER) is True
    
    def test_authenticate_suspended_user(self, manager, trader_user):
        """Test authenticating suspended user."""
        manager.suspend_user(trader_user.user_id)
        
        user = manager.authenticate("trader", "trader123")
        
        assert user is None
    
    def test_user_to_dict(self, manager, trader_user):
        """Test user to dict conversion."""
        user_dict = trader_user.to_dict()
        
        assert "user_id" in user_dict
        assert "username" in user_dict
        assert "email" in user_dict
        assert "role" in user_dict
        assert user_dict["role"] == "trader"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
