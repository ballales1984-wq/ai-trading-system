"""
Test Suite for New Modules
==========================
Tests for recently added modules:
- Feature Store
- Alpha Lab
- Best Execution (TWAP/VWAP)
- Security (JWT)
- Rate Limiter
- RBAC

Author: AI Trading System
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

# =============================================================================
# FEATURE STORE TESTS
# =============================================================================

class TestFeatureStore:
    """Tests for Feature Store module."""
    
    @pytest.fixture
    def feature_store(self):
        """Create feature store instance."""
        from src.research.feature_store import FeatureStore, FeatureDefinition, FeatureType
        return FeatureStore()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100),
        }, index=dates)
        return data
    
    def test_feature_store_initialization(self, feature_store):
        """Test feature store initializes correctly."""
        assert feature_store is not None
        assert hasattr(feature_store, 'register_feature')
        assert hasattr(feature_store, 'compute_features')
        assert hasattr(feature_store, 'get_online_features')
    
    def test_register_feature(self, feature_store):
        """Test feature registration."""
        from src.research.feature_store import FeatureDefinition, FeatureType
        
        feature_def = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.PRICE,
            description="Test feature",
            source="close",
            parameters={"window": 10}
        )
        
        feature_store.register_feature(feature_def)
        
        assert "test_feature" in feature_store._feature_definitions
    
    def test_compute_returns(self, feature_store, sample_data):
        """Test returns computation."""
        from src.research.feature_store import FeatureDefinition, FeatureType
        
        feature_def = FeatureDefinition(
            name="returns",
            feature_type=FeatureType.PRICE,
            description="Simple returns",
            source="close",
            parameters={}
        )
        
        feature_store.register_feature(feature_def)
        result = feature_store.compute_features(sample_data, ["returns"])
        
        assert "returns" in result.columns
        assert not result["returns"].isna().all()
    
    def test_compute_rsi(self, feature_store, sample_data):
        """Test RSI computation."""
        from src.research.feature_store import FeatureDefinition, FeatureType
        
        feature_def = FeatureDefinition(
            name="rsi",
            feature_type=FeatureType.TECHNICAL,
            description="RSI indicator",
            source="close",
            parameters={"window": 14}
        )
        
        feature_store.register_feature(feature_def)
        result = feature_store.compute_features(sample_data, ["rsi"])
        
        assert "rsi" in result.columns
        # RSI should be between 0 and 100
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_store_and_retrieve_features(self, feature_store):
        """Test storing and retrieving feature vectors."""
        from src.research.feature_store import FeatureVector
        
        fv = FeatureVector(
            feature_names=["f1", "f2", "f3"],
            values=np.array([1.0, 2.0, 3.0]),
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            source="test"
        )
        
        feature_store.store_features("BTCUSDT", fv)
        
        online_features = feature_store.get_online_features("BTCUSDT")
        
        assert "f1" in online_features
        assert online_features["f1"] == 1.0


# =============================================================================
# ALPHA LAB TESTS
# =============================================================================

class TestAlphaLab:
    """Tests for Alpha Lab module."""
    
    @pytest.fixture
    def alpha_lab(self):
        """Create alpha lab instance."""
        from src.research.alpha_lab import AlphaLab
        return AlphaLab()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 200)
        prices = 100 * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
            'high': prices * (1 + np.random.uniform(0, 0.02, 200)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, 200)),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 200),
        }, index=dates)
        return data
    
    def test_alpha_lab_initialization(self, alpha_lab):
        """Test alpha lab initializes correctly."""
        assert alpha_lab is not None
        assert hasattr(alpha_lab, 'researcher')
        assert hasattr(alpha_lab, 'run_experiment')
    
    def test_compute_momentum_alpha(self, alpha_lab, sample_data):
        """Test momentum alpha computation."""
        signal = alpha_lab.researcher.compute_alpha("momentum", sample_data, lookback=20)
        
        assert signal is not None
        assert len(signal) == len(sample_data)
    
    def test_compute_mean_reversion_alpha(self, alpha_lab, sample_data):
        """Test mean reversion alpha computation."""
        signal = alpha_lab.researcher.compute_alpha("mean_reversion", sample_data, lookback=20)
        
        assert signal is not None
        assert len(signal) == len(sample_data)
    
    def test_compute_rsi_alpha(self, alpha_lab, sample_data):
        """Test RSI alpha computation."""
        signal = alpha_lab.researcher.compute_alpha("rsi_alpha", sample_data, window=14)
        
        assert signal is not None
        # RSI alpha should be normalized around 0
        valid_signals = signal.dropna()
        assert valid_signals.min() >= -0.5
        assert valid_signals.max() <= 0.5
    
    def test_backtest_alpha(self, alpha_lab, sample_data):
        """Test alpha backtesting."""
        signal = alpha_lab.researcher.compute_alpha("momentum", sample_data)
        prices = sample_data["close"]
        
        perf = alpha_lab.researcher.backtest_alpha(signal, prices)
        
        assert perf is not None
        assert hasattr(perf, 'sharpe_ratio')
        assert hasattr(perf, 'total_return')
        assert hasattr(perf, 'max_drawdown')
    
    def test_rank_alphas(self, alpha_lab, sample_data):
        """Test alpha ranking."""
        prices = sample_data["close"]
        rankings = alpha_lab.researcher.rank_alphas(sample_data, prices)
        
        assert len(rankings) > 0
        # Should be sorted by Sharpe ratio
        sharpe_values = [r.sharpe_ratio for r in rankings]
        assert sharpe_values == sorted(sharpe_values, reverse=True)
    
    def test_run_experiment(self, alpha_lab, sample_data):
        """Test running a complete experiment."""
        prices = sample_data["close"]
        
        results = alpha_lab.run_experiment(
            name="test_experiment",
            df=sample_data,
            prices=prices
        )
        
        assert "experiment_id" in results
        assert "alphas" in results
        assert len(results["alphas"]) > 0


# =============================================================================
# BEST EXECUTION TESTS
# =============================================================================

class TestBestExecution:
    """Tests for Best Execution module."""
    
    @pytest.fixture
    def execution_engine(self):
        """Create execution engine instance."""
        from src.core.execution.best_execution import (
            TWAPAlgorithm, VWAPAlgorithm, POVAlgorithm, ExecutionEngine
        )
        return ExecutionEngine()
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        from src.core.execution.best_execution import MarketDataSnapshot
        
        snapshots = []
        base_price = 100.0
        base_time = datetime.now()
        
        for i in range(100):
            snapshot = MarketDataSnapshot(
                timestamp=base_time + timedelta(minutes=i),
                symbol="BTCUSDT",
                last_price=base_price + np.random.uniform(-1, 1),
                bid=base_price - 0.1,
                ask=base_price + 0.1,
                mid=base_price,
                volume=np.random.uniform(100, 1000)
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def test_twap_algorithm_initialization(self):
        """Test TWAP algorithm initialization."""
        from src.core.execution.best_execution import TWAPAlgorithm, TWAPConfig
        
        config = TWAPConfig(
            total_quantity=1000,
            duration_minutes=60,
            slice_interval_minutes=5
        )
        
        algo = TWAPAlgorithm(config)
        
        assert algo is not None
        assert algo.config.total_quantity == 1000
    
    def test_twap_generate_slices(self):
        """Test TWAP slice generation."""
        from src.core.execution.best_execution import TWAPAlgorithm, TWAPConfig
        
        config = TWAPConfig(
            total_quantity=1000,
            duration_minutes=60,
            slice_interval_minutes=5
        )
        
        algo = TWAPAlgorithm(config)
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=60)
        
        plan = algo.generate_plan(
            symbol="BTCUSDT",
            side="buy",
            start_time=start_time,
            end_time=end_time
        )
        
        assert plan is not None
        assert len(plan.slices) == 12  # 60 / 5 = 12 slices
        assert plan.total_quantity == 1000
    
    def test_vwap_algorithm_initialization(self):
        """Test VWAP algorithm initialization."""
        from src.core.execution.best_execution import VWAPAlgorithm, VWAPConfig
        
        config = VWAPConfig(
            total_quantity=1000,
            duration_minutes=60
        )
        
        algo = VWAPAlgorithm(config)
        
        assert algo is not None
    
    def test_pov_algorithm_initialization(self):
        """Test POV algorithm initialization."""
        from src.core.execution.best_execution import POVAlgorithm, POVConfig
        
        config = POVConfig(
            total_quantity=1000,
            target_participation=0.1,  # 10% of volume
            duration_minutes=60
        )
        
        algo = POVAlgorithm(config)
        
        assert algo is not None
        assert algo.config.target_participation == 0.1
    
    def test_execution_engine_create_plan(self, execution_engine):
        """Test execution engine plan creation."""
        from src.core.execution.best_execution import ExecutionStrategy
        
        plan = execution_engine.create_plan(
            strategy=ExecutionStrategy.TWAP,
            symbol="BTCUSDT",
            side="buy",
            quantity=1000,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )
        
        assert plan is not None
        assert plan.symbol == "BTCUSDT"
        assert plan.total_quantity == 1000


# =============================================================================
# SECURITY TESTS
# =============================================================================

class TestSecurity:
    """Tests for Security module."""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        from app.core.security import JWTManager, SecurityConfig, UserRole
        
        config = SecurityConfig(
            secret_key="test_secret_key_for_testing_only_12345",
            algorithm="HS256",
            access_token_expire_minutes=30
        )
        
        manager = JWTManager(config)
        # Create a test user
        manager.create_user("testuser", "testpass123", UserRole.TRADER)
        return manager
    
    def test_jwt_manager_initialization(self, jwt_manager):
        """Test JWT manager initializes correctly."""
        assert jwt_manager is not None
    
    def test_create_user(self, jwt_manager):
        """Test user creation."""
        from app.core.security import UserRole
        
        user = jwt_manager.create_user("newuser", "password123", UserRole.VIEWER)
        
        assert user is not None
        assert user.username == "newuser"
        assert user.role == UserRole.VIEWER
    
    def test_authenticate_user(self, jwt_manager):
        """Test user authentication."""
        user = jwt_manager.authenticate("testuser", "testpass123")
        
        assert user is not None
        assert user.username == "testuser"
    
    def test_authenticate_wrong_password(self, jwt_manager):
        """Test authentication with wrong password."""
        user = jwt_manager.authenticate("testuser", "wrongpassword")
        
        assert user is None
    
    def test_create_access_token(self, jwt_manager):
        """Test JWT token creation."""
        from app.core.security import UserRole
        
        user = jwt_manager.create_user("tokenuser", "pass123", UserRole.TRADER)
        token = jwt_manager.create_access_token(user)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self, jwt_manager):
        """Test JWT token verification."""
        from app.core.security import UserRole
        
        user = jwt_manager.create_user("verifyuser", "pass123", UserRole.TRADER)
        token = jwt_manager.create_access_token(user)
        
        payload = jwt_manager.verify_token(token)
        
        assert payload is not None
        assert payload.sub == user.user_id
        assert payload.username == "verifyuser"
        assert payload.role == "trader"
    
    def test_invalid_token_verification(self, jwt_manager):
        """Test invalid token verification."""
        payload = jwt_manager.verify_token("invalid_token_here")
        
        assert payload is None
    
    def test_password_hashing(self, jwt_manager):
        """Test password hashing."""
        password = "test_password_123"
        hashed = jwt_manager.hash_password(password)
        
        assert hashed is not None
        assert hashed != password
        assert jwt_manager.verify_password(password, hashed)
    
    def test_password_verification_failure(self, jwt_manager):
        """Test password verification failure."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = jwt_manager.hash_password(password)
        
        assert not jwt_manager.verify_password(wrong_password, hashed)


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Tests for Rate Limiter module."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        from app.core.rate_limiter import RateLimiter, RateLimitConfig, RateLimitStrategy
        
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            burst_size=5,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        
        return RateLimiter(config)
    
    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initializes correctly."""
        assert rate_limiter is not None
    
    def test_allow_request(self, rate_limiter):
        """Test request allowance."""
        client_id = "test_client"
        
        # First request should be allowed
        assert rate_limiter.is_allowed(client_id)
    
    def test_rate_limit_enforcement(self, rate_limiter):
        """Test rate limit enforcement."""
        client_id = "test_client_limit"
        
        # Make requests up to the limit
        for _ in range(10):
            rate_limiter.is_allowed(client_id)
        
        # Next request should be blocked
        assert not rate_limiter.is_allowed(client_id)
    
    def test_burst_handling(self, rate_limiter):
        """Test burst request handling."""
        client_id = "test_client_burst"
        
        # Burst of requests
        results = [rate_limiter.is_allowed(client_id) for _ in range(15)]
        
        # Some should be allowed, some blocked
        allowed_count = sum(results)
        assert allowed_count <= 10  # requests_per_minute limit
    
    def test_rate_limit_reset(self, rate_limiter):
        """Test rate limit reset."""
        client_id = "test_client_reset"
        
        # Use up the limit
        for _ in range(10):
            rate_limiter.is_allowed(client_id)
        
        # Reset the limit
        rate_limiter.reset(client_id)
        
        # Should be allowed again
        assert rate_limiter.is_allowed(client_id)


# =============================================================================
# RBAC TESTS
# =============================================================================

class TestRBAC:
    """Tests for RBAC module."""
    
    @pytest.fixture
    def rbac_manager(self):
        """Create RBAC manager instance."""
        from app.core.rbac import RBACManager
        
        return RBACManager()
    
    def test_rbac_initialization(self, rbac_manager):
        """Test RBAC initializes correctly."""
        assert rbac_manager is not None
    
    def test_role_permissions(self, rbac_manager):
        """Test role has correct permissions."""
        from app.core.rbac import Permission
        
        admin_permissions = rbac_manager.get_role_permissions("admin")
        
        # Admin should have all permissions
        assert Permission.ADMIN_USERS in admin_permissions
        assert Permission.PORTFOLIO_READ in admin_permissions
    
    def test_trader_permissions(self, rbac_manager):
        """Test trader has correct permissions."""
        from app.core.rbac import Permission
        
        trader_permissions = rbac_manager.get_role_permissions("trader")
        
        # Trader should have trading permissions
        assert Permission.PORTFOLIO_READ in trader_permissions
        assert Permission.ORDER_CREATE in trader_permissions
        # But not admin permissions
        assert Permission.ADMIN_USERS not in trader_permissions
    
    def test_viewer_permissions(self, rbac_manager):
        """Test viewer has correct permissions."""
        from app.core.rbac import Permission
        
        viewer_permissions = rbac_manager.get_role_permissions("viewer")
        
        # Viewer should only have read permissions
        assert Permission.PORTFOLIO_READ in viewer_permissions
        assert Permission.ORDER_CREATE not in viewer_permissions
    
    def test_check_permission(self, rbac_manager):
        """Test permission checking."""
        from app.core.rbac import Permission
        
        # Admin can do everything
        assert rbac_manager.check_permission("admin", Permission.ADMIN_USERS)
        
        # Trader cannot access admin functions
        assert not rbac_manager.check_permission("trader", Permission.ADMIN_USERS)
        
        # Trader can create orders
        assert rbac_manager.check_permission("trader", Permission.ORDER_CREATE)
    
    def test_permission_decorator(self, rbac_manager):
        """Test permission decorator."""
        from app.core.rbac import Permission, require_permission
        
        @require_permission(Permission.ORDER_CREATE)
        def create_order(user_role):
            return "order_created"
        
        # Should work for trader
        result = create_order("trader")
        assert result == "order_created"
        
        # Should fail for viewer
        with pytest.raises(Exception):
            create_order("viewer")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for new modules."""
    
    def test_feature_store_to_alpha_lab_integration(self):
        """Test integration between Feature Store and Alpha Lab."""
        from src.research.feature_store import FeatureStore, get_technical_features
        from src.research.alpha_lab import AlphaLab
        
        # Create feature store
        fs = FeatureStore()
        fs.register_features(get_technical_features())
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100),
        }, index=dates)
        
        # Compute features
        features = fs.compute_features(data)
        
        # Use in Alpha Lab
        alpha_lab = AlphaLab()
        signal = alpha_lab.researcher.compute_alpha("momentum", features)
        
        assert signal is not None
    
    def test_security_rate_limiter_integration(self):
        """Test integration between Security and Rate Limiter."""
        from app.core.security import SecurityManager, SecurityConfig
        from app.core.rate_limiter import RateLimiter, RateLimitConfig
        
        # Create security manager
        security = SecurityManager(SecurityConfig(
            secret_key="test_key_integration_12345678901234567890",
            algorithm="HS256"
        ))
        
        # Create rate limiter
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=10))
        
        # Create token
        token = security.create_access_token("user1", "testuser", "trader")
        
        # Verify token
        payload = security.verify_token(token)
        user_id = payload["sub"]
        
        # Check rate limit
        is_allowed = limiter.is_allowed(user_id)
        
        assert is_allowed


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
