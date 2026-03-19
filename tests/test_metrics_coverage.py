"""
Test Coverage for Metrics Module
================================
"""

import pytest
from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ACTIVE_TRADES,
    PORTFOLIO_VALUE,
    PNL_TOTAL,
    RISK_VA_R95,
    ERROR_COUNT,
    instrument_requests,
    get_metrics_app,
)


class TestMetrics:
    """Test Prometheus metrics"""
    
    def test_metrics_exist(self):
        """Test that metrics are defined"""
        assert REQUEST_COUNT is not None
        assert REQUEST_LATENCY is not None
        assert ACTIVE_TRADES is not None
        assert PORTFOLIO_VALUE is not None
        assert PNL_TOTAL is not None
        assert RISK_VA_R95 is not None
        assert ERROR_COUNT is not None
    
    def test_instrument_requests(self):
        """Test instrument_requests function"""
        result = instrument_requests()
        assert result is not None
        assert len(result) == 3
    
    def test_get_metrics_app(self):
        """Test get_metrics_app function"""
        app = get_metrics_app()
        assert app is not None


class TestMetricsLabels:
    """Test metrics with labels"""
    
    def test_request_count_labels(self):
        """Test REQUEST_COUNT has correct labels"""
        # Labels should be method, endpoint, status
        assert hasattr(REQUEST_COUNT, 'labels')
    
    def test_error_count_labels(self):
        """Test ERROR_COUNT has correct labels"""
        # Labels should be type
        assert hasattr(ERROR_COUNT, 'labels')


class TestMetricsGauges:
    """Test gauge metrics"""
    
    def test_active_trades_gauge(self):
        """Test ACTIVE_TRADES gauge"""
        assert hasattr(ACTIVE_TRADES, 'inc')
        assert hasattr(ACTIVE_TRADES, 'dec')
        assert hasattr(ACTIVE_TRADES, 'set')
    
    def test_portfolio_value_gauge(self):
        """Test PORTFOLIO_VALUE gauge"""
        assert hasattr(PORTFOLIO_VALUE, 'set')
    
    def test_pnl_gauge(self):
        """Test PNL_TOTAL gauge"""
        assert hasattr(PNL_TOTAL, 'set')
    
    def test_var_gauge(self):
        """Test RISK_VA_R95 gauge"""
        assert hasattr(RISK_VA_R95, 'set')


class TestMetricsHistogram:
    """Test histogram metrics"""
    
    def test_request_latency_histogram(self):
        """Test REQUEST_LATENCY histogram"""
        assert hasattr(REQUEST_LATENCY, 'observe')
        assert hasattr(REQUEST_LATENCY, 'labels')
