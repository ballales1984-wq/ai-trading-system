"""
Tests for Execution Engine module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import asdict

from app.execution.execution_engine import (
    ExecutionEngine,
    ExecutionState,
    RetryConfig,
    ExecutionResult
)
from app.execution.broker_connector import (
    BrokerConnector,
    Order,
    OrderSide,
    OrderType,
    OrderStatus
)


class TestExecutionState:
    """Test ExecutionState enum."""

    def test_execution_state_values(self):
        """Test ExecutionState enum values."""
        assert ExecutionState.IDLE.value == "idle"
        assert ExecutionState.EXECUTING.value == "executing"
        assert ExecutionState.RETRYING.value == "retrying"
        assert ExecutionState.PAUSED.value == "paused"


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_retry_config_defaults(self):
        """Test default retry config values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_retry_config_custom(self):
        """Test custom retry config values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=False
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_retry_config_to_dict(self):
        """Test converting retry config to dict."""
        config = RetryConfig(max_retries=5)
        config_dict = asdict(config)
        assert config_dict["max_retries"] == 5


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_success(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            order_id="order123",
            message="Order filled",
            filled_quantity=10.0,
            avg_price=100.0,
            commission=0.1
        )
        assert result.success is True
        assert result.order_id == "order123"
        assert result.filled_quantity == 10.0
        assert result.avg_price == 100.0

    def test_execution_result_failure(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            message="Order failed",
            error="Insufficient funds"
        )
        assert result.success is False
        assert result.error == "Insufficient funds"

    def test_execution_result_defaults(self):
        """Test execution result default values."""
        result = ExecutionResult(success=False)
        assert result.order_id is None
        assert result.message == ""
        assert result.filled_quantity == 0.0
        assert result.avg_price == 0.0
        assert result.commission == 0.0
        assert result.attempts == 1
        assert result.error is None


class TestExecutionEngine:
    """Test ExecutionEngine class."""

    def test_execution_engine_creation(self):
        """Test creating execution engine."""
        mock_broker = Mock(spec=BrokerConnector)
        engine = ExecutionEngine(broker=mock_broker)
        
        assert engine.broker is mock_broker
        assert engine.state == ExecutionState.IDLE
        assert engine.retry_config is not None

    def test_execution_engine_with_risk_engine(self):
        """Test execution engine with risk engine."""
        mock_broker = Mock(spec=BrokerConnector)
        mock_risk = Mock()
        
        engine = ExecutionEngine(
            broker=mock_broker,
            risk_engine=mock_risk
        )
        
        assert engine.risk_engine is mock_risk

    def test_execution_engine_with_custom_retry(self):
        """Test execution engine with custom retry config."""
        mock_broker = Mock(spec=BrokerConnector)
        custom_config = RetryConfig(max_retries=10)
        
        engine = ExecutionEngine(
            broker=mock_broker,
            retry_config=custom_config
        )
        
        assert engine.retry_config.max_retries == 10

    @pytest.mark.asyncio
    async def test_execution_engine_initial_state(self):
        """Test execution engine initial state."""
        mock_broker = Mock(spec=BrokerConnector)
        engine = ExecutionEngine(broker=mock_broker)
        
        assert engine.state == ExecutionState.IDLE

    def test_execution_engine_order_tracking(self):
        """Test execution engine has order tracking."""
        mock_broker = Mock(spec=BrokerConnector)
        engine = ExecutionEngine(broker=mock_broker)
        
        assert hasattr(engine, 'pending_orders')
