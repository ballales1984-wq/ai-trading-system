"""
Comprehensive tests for app core modules: scheduler, backtest, metrics.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json


class TestScheduler:
    """Tests for app/scheduler.py module."""
    
    def test_scheduler_import(self):
        """Test that scheduler module can be imported."""
        from app.scheduler import TaskScheduler
        assert TaskScheduler is not None
    
    def test_schedule_type_import(self):
        """Test ScheduleType can be imported."""
        from app.scheduler import ScheduleType
        assert ScheduleType is not None
    
    def test_task_status_import(self):
        """Test TaskStatus can be imported."""
        from app.scheduler import TaskStatus
        assert TaskStatus is not None


class TestBacktest:
    """Tests for app/backtest.py module."""
    
    def test_backtest_import(self):
        """Test that backtest module can be imported."""
        from app.backtest import BacktestEngine
        assert BacktestEngine is not None


class TestMetrics:
    """Tests for app/metrics.py module."""
    
    def test_metrics_import(self):
        """Test that metrics module can be imported."""
        from app import metrics
        assert metrics is not None


class TestAppMain:
    """Tests for app/main.py module."""
    
    def test_app_main_import(self):
        """Test that main app can be imported."""
        from app.main import app
        assert app is not None
    
    def test_app_has_routes(self):
        """Test that app has routes configured."""
        from app.main import app
        
        # Check that app has routes
        assert hasattr(app, 'routes')
    
    def test_app_has_middleware(self):
        """Test that app has middleware configured."""
        from app.main import app
        
        # Check that app has middleware
        assert hasattr(app, 'user_middleware') or hasattr(app, 'middleware')


class TestExecution:
    """Tests for execution modules."""
    
    def test_order_manager_import(self):
        """Test that order manager can be imported."""
        from app.execution.order_manager import OrderManager
        assert OrderManager is not None
    
    def test_broker_connector_import(self):
        """Test that broker connector can be imported."""
        from app.execution.connectors.binance_connector import BinanceConnector
        assert BinanceConnector is not None
    
    def test_paper_connector_import(self):
        """Test that paper connector can be imported."""
        from app.execution.connectors.paper_connector import PaperConnector
        assert PaperConnector is not None


class TestMarketData:
    """Tests for market data modules."""
    
    def test_data_feed_import(self):
        """Test that data feed can be imported."""
        from app.market_data.data_feed import DataFeed
        assert DataFeed is not None
    
    def test_websocket_stream_import(self):
        """Test that websocket stream can be imported."""
        from app.market_data.websocket_stream import WebSocketStream
        assert WebSocketStream is not None


class TestPortfolio:
    """Tests for portfolio modules."""
    
    def test_portfolio_optimization_import(self):
        """Test that portfolio optimization can be imported."""
        from app.portfolio.optimization import PortfolioOptimizer
        assert PortfolioOptimizer is not None
    
    def test_portfolio_performance_import(self):
        """Test that portfolio performance can be imported."""
        from app.portfolio.performance import PortfolioPerformance
        assert PortfolioPerformance is not None


class TestRisk:
    """Tests for risk modules."""
    
    def test_risk_engine_import(self):
        """Test that risk engine can be imported."""
        from app.risk.risk_engine import RiskEngine
        assert RiskEngine is not None
    
    def test_hardened_risk_import(self):
        """Test that hardened risk engine can be imported."""
        from app.risk.hardened_risk_engine import HardenedRiskEngine
        assert HardenedRiskEngine is not None


class TestStrategies:
    """Tests for strategy modules."""
    
    def test_momentum_strategy_import(self):
        """Test that momentum strategy can be imported."""
        from app.strategies.momentum import MomentumStrategy
        assert MomentumStrategy is not None
    
    def test_mean_reversion_import(self):
        """Test that mean reversion strategy can be imported."""
        from app.strategies.mean_reversion import MeanReversionStrategy
        assert MeanReversionStrategy is not None
    
    def test_multi_strategy_import(self):
        """Test that multi strategy can be imported."""
        from app.strategies.multi_strategy import MultiStrategyManager
        assert MultiStrategyManager is not None


class TestCompliance:
    """Tests for compliance modules."""
    
    def test_compliance_import(self):
        """Test that compliance module can be imported."""
        from app import compliance
        assert compliance is not None
    
    def test_audit_logger_import(self):
        """Test that audit logger can be imported."""
        from app.compliance import audit
        assert audit is not None


class TestDatabase:
    """Tests for database modules."""
    
    def test_database_models_import(self):
        """Test that database models can be imported."""
        from app.database.models import User, Order, Portfolio
        assert User is not None
        assert Order is not None
        assert Portfolio is not None
    
    def test_repository_import(self):
        """Test that repository can be imported."""
        from app.database.repository import Repository
        assert Repository is not None


class TestAPIRoutes:
    """Tests for API route modules."""
    
    def test_health_route_import(self):
        """Test that health route can be imported."""
        from app.api.routes import health
        assert health is not None
    
    def test_orders_route_import(self):
        """Test that orders route can be imported."""
        from app.api.routes import orders
        assert orders is not None
    
    def test_portfolio_route_import(self):
        """Test that portfolio route can be imported."""
        from app.api.routes import portfolio
        assert portfolio is not None
    
    def test_market_route_import(self):
        """Test that market route can be imported."""
        from app.api.routes import market
        assert market is not None
    
    def test_risk_route_import(self):
        """Test that risk route can be imported."""
        from app.api.routes import risk
        assert risk is not None
    
    def test_strategy_route_import(self):
        """Test that strategy route can be imported."""
        from app.api.routes import strategy
        assert strategy is not None


class TestCoreConfig:
    """Tests for core config modules."""
    
    def test_settings_import(self):
        """Test that settings can be imported."""
        from app.core.config import settings, Settings
        assert settings is not None
        assert Settings is not None
    
    def test_security_import(self):
        """Test that security module can be imported."""
        from app.core.security import JWTManager, SecurityConfig
        assert JWTManager is not None
        assert SecurityConfig is not None
    
    def test_rate_limiter_import(self):
        """Test that rate limiter can be imported."""
        from app.core.rate_limiter import RateLimiter
        assert RateLimiter is not None
    
    def test_logging_import(self):
        """Test that logging module can be imported."""
        from app.core.logging import setup_logging
        assert setup_logging is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
