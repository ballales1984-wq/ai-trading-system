"""
Test Coverage for Multi-Tenant Module
====================================
"""

import pytest
from app.core.multi_tenant import (
    MultiTenantManager,
    UserRole,
    AccountStatus,
    User,
    SubAccount
)


class TestMultiTenantUserRole:
    """Test UserRole enum"""
    
    def test_user_role_values(self):
        """Test UserRole enum values"""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.MANAGER.value == "manager"
        assert UserRole.TRADER.value == "trader"
        assert UserRole.VIEWER.value == "viewer"


class TestMultiTenantAccountStatus:
    """Test AccountStatus enum"""
    
    def test_account_status_values(self):
        """Test AccountStatus enum values"""
        assert AccountStatus.ACTIVE.value == "active"
        assert AccountStatus.SUSPENDED.value == "suspended"
        assert AccountStatus.PENDING.value == "pending"
        assert AccountStatus.CLOSED.value == "closed"


class TestMultiTenantUser:
    """Test User dataclass"""
    
    def test_user_creation(self):
        """Test User creation"""
        user = User(
            user_id="test-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.TRADER,
            status=AccountStatus.ACTIVE
        )
        assert user.user_id == "test-id"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.TRADER
        assert user.status == AccountStatus.ACTIVE
    
    def test_user_to_dict(self):
        """Test User to_dict method"""
        user = User(
            user_id="test-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.TRADER,
            status=AccountStatus.ACTIVE
        )
        user_dict = user.to_dict()
        assert user_dict["user_id"] == "test-id"
        assert user_dict["username"] == "testuser"
        assert user_dict["email"] == "test@example.com"
        assert user_dict["role"] == "trader"
        assert user_dict["status"] == "active"


class TestMultiTenantSubAccount:
    """Test SubAccount dataclass"""
    
    def test_sub_account_creation(self):
        """Test SubAccount creation"""
        sub = SubAccount(
            sub_account_id="sub-123",
            parent_user_id="parent-123",
            name="Test Account",
            initial_balance=1000.0,
            current_balance=1000.0,
            status=AccountStatus.ACTIVE
        )
        assert sub.sub_account_id == "sub-123"
        assert sub.parent_user_id == "parent-123"
        assert sub.name == "Test Account"
        assert sub.initial_balance == 1000.0
        assert sub.current_balance == 1000.0
        assert sub.status == AccountStatus.ACTIVE


class TestMultiTenantManager:
    """Test MultiTenantManager class"""
    
    def test_manager_initialization(self):
        """Test MultiTenantManager initialization"""
        manager = MultiTenantManager()
        assert manager._users == {}
        assert manager._sub_accounts == {}
        assert manager._api_keys == {}
    
    def test_create_user(self):
        """Test user creation"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            role=UserRole.TRADER
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.TRADER
        assert user.status == AccountStatus.PENDING
    
    def test_create_user_duplicate_username(self):
        """Test duplicate username raises error"""
        manager = MultiTenantManager()
        manager.create_user(
            username="testuser",
            email="test1@example.com",
            password="password123"
        )
        with pytest.raises(ValueError, match="already exists"):
            manager.create_user(
                username="testuser",
                email="test2@example.com",
                password="password123"
            )
    
    def test_create_user_duplicate_email(self):
        """Test duplicate email raises error"""
        manager = MultiTenantManager()
        manager.create_user(
            username="user1",
            email="test@example.com",
            password="password123"
        )
        with pytest.raises(ValueError, match="already exists"):
            manager.create_user(
                username="user2",
                email="test@example.com",
                password="password123"
            )
    
    def test_authenticate_success(self):
        """Test successful authentication"""
        manager = MultiTenantManager()
        manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        user = manager.authenticate("testuser", "password123")
        assert user is not None
        assert user.username == "testuser"
        assert user.status == AccountStatus.ACTIVE  # Pending users are activated
    
    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password"""
        manager = MultiTenantManager()
        manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        user = manager.authenticate("testuser", "wrongpassword")
        assert user is None
    
    def test_authenticate_nonexistent_user(self):
        """Test authentication with nonexistent user"""
        manager = MultiTenantManager()
        user = manager.authenticate("nonexistent", "password123")
        assert user is None
    
    def test_authenticate_suspended_user(self):
        """Test authentication with suspended user"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        manager.suspend_user(user.user_id)
        result = manager.authenticate("testuser", "password123")
        assert result is None
    
    def test_authenticate_by_email(self):
        """Test authentication by email"""
        manager = MultiTenantManager()
        manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        user = manager.authenticate("test@example.com", "password123")
        assert user is not None
        assert user.username == "testuser"
    
    def test_get_user(self):
        """Test get user by ID"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        retrieved = manager.get_user(user.user_id)
        assert retrieved is not None
        assert retrieved.user_id == user.user_id
    
    def test_get_user_nonexistent(self):
        """Test get nonexistent user"""
        manager = MultiTenantManager()
        user = manager.get_user("nonexistent-id")
        assert user is None
    
    def test_get_user_by_username(self):
        """Test get user by username"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        retrieved = manager.get_user_by_username("testuser")
        assert retrieved is not None
        assert retrieved.username == "testuser"
    
    def test_update_user_email(self):
        """Test update user email"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        updated = manager.update_user(user.user_id, email="new@example.com")
        assert updated is not None
        assert updated.email == "new@example.com"
    
    def test_update_user_role(self):
        """Test update user role"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        updated = manager.update_user(user.user_id, role=UserRole.ADMIN)
        assert updated is not None
        assert updated.role == UserRole.ADMIN
    
    def test_update_user_status(self):
        """Test update user status"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        updated = manager.update_user(user.user_id, status=AccountStatus.SUSPENDED)
        assert updated is not None
        assert updated.status == AccountStatus.SUSPENDED
    
    def test_suspend_user(self):
        """Test suspend user"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        result = manager.suspend_user(user.user_id)
        assert result is True
        assert user.status == AccountStatus.SUSPENDED
    
    def test_suspend_nonexistent_user(self):
        """Test suspend nonexistent user"""
        manager = MultiTenantManager()
        result = manager.suspend_user("nonexistent-id")
        assert result is False
    
    def test_activate_user(self):
        """Test activate user"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        manager.suspend_user(user.user_id)
        result = manager.activate_user(user.user_id)
        assert result is True
        assert user.status == AccountStatus.ACTIVE
    
    def test_delete_user(self):
        """Test delete user"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        result = manager.delete_user(user.user_id)
        assert result is True
        assert manager.get_user(user.user_id) is None
    
    def test_create_api_key(self):
        """Test create API key"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        api_key = manager.create_api_key(user.user_id, "Test Key")
        assert api_key is not None
        assert api_key.startswith("ats_")
        assert api_key in user.api_keys
    
    def test_create_api_key_nonexistent_user(self):
        """Test create API key for nonexistent user"""
        manager = MultiTenantManager()
        with pytest.raises(ValueError, match="not found"):
            manager.create_api_key("nonexistent-id")
    
    def test_verify_api_key(self):
        """Test verify API key"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        # Activate user to be able to verify API key
        manager.activate_user(user.user_id)
        api_key = manager.create_api_key(user.user_id)
        verified_user = manager.verify_api_key(api_key)
        assert verified_user is not None
        assert verified_user.user_id == user.user_id
    
    def test_verify_invalid_api_key(self):
        """Test verify invalid API key"""
        manager = MultiTenantManager()
        user = manager.verify_api_key("invalid_key")
        assert user is None
    
    def test_revoke_api_key(self):
        """Test revoke API key"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        api_key = manager.create_api_key(user.user_id)
        result = manager.revoke_api_key(user.user_id, api_key)
        assert result is True
        assert api_key not in user.api_keys
    
    def test_create_sub_account(self):
        """Test create sub-account"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        sub = manager.create_sub_account(user.user_id, "Trading Account", 10000.0)
        assert sub is not None
        assert sub.name == "Trading Account"
        assert sub.initial_balance == 10000.0
        assert sub.parent_user_id == user.user_id
        assert sub.sub_account_id in user.sub_accounts
    
    def test_create_sub_account_nonexistent_user(self):
        """Test create sub-account for nonexistent user"""
        manager = MultiTenantManager()
        sub = manager.create_sub_account("nonexistent-id", "Test Account")
        assert sub is None
    
    def test_get_sub_account(self):
        """Test get sub-account"""
        manager = MultiTenantManager()
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        sub = manager.create_sub_account(user.user_id, "Trading Account")
        retrieved = manager.get_sub_account(sub.sub_account_id)
        assert retrieved is not None
        assert retrieved.sub_account_id == sub.sub_account_id
    
    def test_list_users_no_filter(self):
        """Test list users without filter"""
        manager = MultiTenantManager()
        manager.create_user("user1", "user1@example.com", "pass", UserRole.TRADER)
        manager.create_user("user2", "user2@example.com", "pass", UserRole.VIEWER)
        users = manager.list_users()
        assert len(users) == 2
    
    def test_list_users_by_role(self):
        """Test list users by role"""
        manager = MultiTenantManager()
        manager.create_user("user1", "user1@example.com", "pass", UserRole.TRADER)
        manager.create_user("user2", "user2@example.com", "pass", UserRole.VIEWER)
        traders = manager.list_users(role=UserRole.TRADER)
        assert len(traders) == 1
        assert traders[0].role == UserRole.TRADER
    
    def test_list_users_by_status(self):
        """Test list users by status"""
        manager = MultiTenantManager()
        user = manager.create_user("user1", "user1@example.com", "pass", UserRole.TRADER)
        manager.suspend_user(user.user_id)
        active_users = manager.list_users(status=AccountStatus.ACTIVE)
        suspended_users = manager.list_users(status=AccountStatus.SUSPENDED)
        assert len(active_users) == 0
        assert len(suspended_users) == 1
    
    def test_check_permission_admin(self):
        """Test check permission for admin"""
        manager = MultiTenantManager()
        admin = manager.create_user("admin", "admin@example.com", "pass", UserRole.ADMIN)
        assert manager.check_permission(admin, UserRole.ADMIN) is True
        assert manager.check_permission(admin, UserRole.MANAGER) is True
        assert manager.check_permission(admin, UserRole.TRADER) is True
        assert manager.check_permission(admin, UserRole.VIEWER) is True
    
    def test_check_permission_trader(self):
        """Test check permission for trader"""
        manager = MultiTenantManager()
        trader = manager.create_user("trader", "trader@example.com", "pass", UserRole.TRADER)
        assert manager.check_permission(trader, UserRole.ADMIN) is False
        assert manager.check_permission(trader, UserRole.MANAGER) is False
        assert manager.check_permission(trader, UserRole.TRADER) is True
        assert manager.check_permission(trader, UserRole.VIEWER) is True
    
    def test_check_permission_viewer(self):
        """Test check permission for viewer"""
        manager = MultiTenantManager()
        viewer = manager.create_user("viewer", "viewer@example.com", "pass", UserRole.VIEWER)
        assert manager.check_permission(viewer, UserRole.ADMIN) is False
        assert manager.check_permission(viewer, UserRole.MANAGER) is False
        assert manager.check_permission(viewer, UserRole.TRADER) is False
        assert manager.check_permission(viewer, UserRole.VIEWER) is True
