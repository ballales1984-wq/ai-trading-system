"""
Tests for TimescaleDB Continuous Aggregates
===========================================
Tests for hypertables, continuous aggregates, and retention policies.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys


class TestTimescaleModels:
    """Test TimescaleDB model definitions."""
    
    def test_ohlcv_bar_model(self):
        """Test OHLCVBar model has correct columns."""
        from app.database.timescale_models import OHLCVBar
        
        # Check required columns exist
        assert hasattr(OHLCVBar, 'time')
        assert hasattr(OHLCVBar, 'symbol')
        assert hasattr(OHLCVBar, 'interval')
        assert hasattr(OHLCVBar, 'open')
        assert hasattr(OHLCVBar, 'high')
        assert hasattr(OHLCVBar, 'low')
        assert hasattr(OHLCVBar, 'close')
        assert hasattr(OHLCVBar, 'volume')
    
    def test_trade_tick_model(self):
        """Test TradeTick model has correct columns."""
        from app.database.timescale_models import TradeTick
        
        assert hasattr(TradeTick, 'time')
        assert hasattr(TradeTick, 'symbol')
        assert hasattr(TradeTick, 'trade_id')
        assert hasattr(TradeTick, 'price')
        assert hasattr(TradeTick, 'quantity')
        assert hasattr(TradeTick, 'is_buyer_maker')
    
    def test_portfolio_history_model(self):
        """Test PortfolioHistory model has correct columns."""
        from app.database.timescale_models import PortfolioHistory
        
        assert hasattr(PortfolioHistory, 'time')
        assert hasattr(PortfolioHistory, 'portfolio_id')
        assert hasattr(PortfolioHistory, 'total_value')
        assert hasattr(PortfolioHistory, 'cash')
        assert hasattr(PortfolioHistory, 'equity')
        assert hasattr(PortfolioHistory, 'unrealized_pnl')
        assert hasattr(PortfolioHistory, 'realized_pnl')
        assert hasattr(PortfolioHistory, 'daily_pnl')
        assert hasattr(PortfolioHistory, 'drawdown')
        assert hasattr(PortfolioHistory, 'sharpe')
        assert hasattr(PortfolioHistory, 'win_rate')
    
    def test_risk_metrics_history_model(self):
        """Test RiskMetricsHistory model has correct columns."""
        from app.database.timescale_models import RiskMetricsHistory
        
        assert hasattr(RiskMetricsHistory, 'time')
        assert hasattr(RiskMetricsHistory, 'portfolio_id')
        assert hasattr(RiskMetricsHistory, 'var_1d_95')
        assert hasattr(RiskMetricsHistory, 'var_1d_99')
        assert hasattr(RiskMetricsHistory, 'volatility_daily')
        assert hasattr(RiskMetricsHistory, 'current_drawdown')
        assert hasattr(RiskMetricsHistory, 'max_drawdown')
    
    def test_signal_timeseries_model(self):
        """Test SignalTimeseries model has correct columns."""
        from app.database.timescale_models import SignalTimeseries
        
        assert hasattr(SignalTimeseries, 'timestamp')
        assert hasattr(SignalTimeseries, 'symbol')
        assert hasattr(SignalTimeseries, 'action')
        assert hasattr(SignalTimeseries, 'confidence')
        assert hasattr(SignalTimeseries, 'executed')
        assert hasattr(SignalTimeseries, 'result_pnl')
        assert hasattr(SignalTimeseries, 'technical_score')
        assert hasattr(SignalTimeseries, 'sentiment_score')
        assert hasattr(SignalTimeseries, 'ml_score')


class TestContinuousAggregates:
    """Test continuous aggregate definitions."""
    
    def test_aggregates_list_not_empty(self):
        """Test that aggregates list is defined and not empty."""
        from app.database.timescale_models import create_continuous_aggregates
        
        # Get the function source to verify it contains aggregate definitions
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        # Check for expected aggregate views
        assert 'ohlcv_5m' in source
        assert 'ohlcv_1h' in source
        assert 'ohlcv_1d' in source
        assert 'trade_volume_1h' in source
        assert 'daily_signals' in source
        assert 'weekly_performance' in source
        assert 'hourly_risk_metrics' in source
    
    def test_aggregates_use_time_bucket(self):
        """Test that aggregates use TimescaleDB time_bucket function."""
        from app.database.timescale_models import create_continuous_aggregates
        
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        assert 'time_bucket' in source
        assert 'WITH (timescaledb.continuous)' in source
    
    def test_daily_signals_aggregate_structure(self):
        """Test daily_signals aggregate has correct structure."""
        from app.database.timescale_models import create_continuous_aggregates
        
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        # Check for expected columns in daily_signals
        assert 'signal_count' in source
        assert 'buy_ratio' in source
        assert 'avg_confidence' in source
        assert 'execution_rate' in source
        assert 'total_result_pnl' in source
        assert 'signals_ts' in source
    
    def test_weekly_performance_aggregate_structure(self):
        """Test weekly_performance aggregate has correct structure."""
        from app.database.timescale_models import create_continuous_aggregates
        
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        # Check for expected columns in weekly_performance
        assert 'start_value' in source
        assert 'end_value' in source
        assert 'value_change' in source
        assert 'return_pct' in source
        assert 'max_drawdown' in source
        assert 'avg_sharpe' in source
        assert 'portfolio_history' in source


class TestRefreshPolicies:
    """Test refresh policy definitions."""
    
    def test_refresh_policies_defined(self):
        """Test that refresh policies are defined."""
        from app.database.timescale_models import create_continuous_aggregates
        
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        assert 'refresh_policies' in source
        assert 'add_continuous_aggregate_policy' in source
    
    def test_refresh_intervals_appropriate(self):
        """Test that refresh intervals are appropriate for each aggregate."""
        from app.database.timescale_models import create_continuous_aggregates
        
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        # 5m aggregate should refresh every 5 minutes
        assert "INTERVAL '5 minutes'" in source
        
        # 1h aggregate should refresh every hour
        assert "INTERVAL '1 hour'" in source
        
        # 1d aggregate should refresh every day
        assert "INTERVAL '1 day'" in source


class TestRetentionPolicies:
    """Test retention policy definitions."""
    
    def test_retention_policies_defined(self):
        """Test that retention policies are defined."""
        from app.database.timescale_models import create_continuous_aggregates
        
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        assert 'retention_policies' in source
        assert 'add_retention_policy' in source
    
    def test_retention_periods_appropriate(self):
        """Test that retention periods are appropriate for each data type."""
        from app.database.timescale_models import create_continuous_aggregates
        
        import inspect
        source = inspect.getsource(create_continuous_aggregates)
        
        # OHLCV bars: 90 days
        assert "INTERVAL '90 days'" in source
        
        # Trade ticks: 30 days (high volume)
        assert "INTERVAL '30 days'" in source
        
        # Orderbook snapshots: 7 days (very high volume)
        assert "INTERVAL '7 days'" in source
        
        # Portfolio history: 5 years
        assert "INTERVAL '5 years'" in source
        
        # Risk metrics: 2 years
        assert "INTERVAL '2 years'" in source
        
        # Signals: 1 year
        assert "INTERVAL '1 year'" in source


class TestHypertableCreation:
    """Test hypertable creation methods."""
    
    def test_ohlcv_create_hypertable_method(self):
        """Test OHLCVBar has create_hypertable classmethod."""
        from app.database.timescale_models import OHLCVBar
        
        assert hasattr(OHLCVBar, 'create_hypertable')
        assert callable(getattr(OHLCVBar, 'create_hypertable'))
    
    def test_trade_tick_create_hypertable_method(self):
        """Test TradeTick has create_hypertable classmethod."""
        from app.database.timescale_models import TradeTick
        
        assert hasattr(TradeTick, 'create_hypertable')
        assert callable(getattr(TradeTick, 'create_hypertable'))
    
    def test_portfolio_history_create_hypertable_method(self):
        """Test PortfolioHistory has create_hypertable classmethod."""
        from app.database.timescale_models import PortfolioHistory
        
        assert hasattr(PortfolioHistory, 'create_hypertable')
        assert callable(getattr(PortfolioHistory, 'create_hypertable'))
    
    def test_signal_timeseries_create_hypertable_method(self):
        """Test SignalTimeseries has create_hypertable classmethod."""
        from app.database.timescale_models import SignalTimeseries
        
        assert hasattr(SignalTimeseries, 'create_hypertable')
        assert callable(getattr(SignalTimeseries, 'create_hypertable'))
    
    @patch('app.database.timescale_models.create_engine')
    @patch('app.database.timescale_models.text')
    def test_hypertable_creation_handles_exception(self, mock_text, mock_engine):
        """Test that hypertable creation handles exceptions gracefully."""
        from app.database.timescale_models import OHLCVBar
        
        # Mock engine that raises exception
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("TimescaleDB not available")
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        
        # Should not raise exception
        try:
            OHLCVBar.create_hypertable(mock_engine.return_value)
            exception_raised = False
        except Exception:
            exception_raised = True
        
        assert not exception_raised, "create_hypertable should handle exceptions gracefully"


class TestInitTimescaleDB:
    """Test TimescaleDB initialization function."""
    
    def test_init_function_exists(self):
        """Test that init_timescaledb function exists."""
        from app.database.timescale_models import init_timescaledb
        
        assert callable(init_timescaledb)
    
    @patch('app.database.timescale_models.create_engine')
    @patch('app.database.timescale_models.text')
    @patch('app.database.timescale_models.Base')
    @patch('app.database.timescale_models.OHLCVBar')
    @patch('app.database.timescale_models.TradeTick')
    @patch('app.database.timescale_models.OrderBookSnapshot')
    @patch('app.database.timescale_models.FundingRate')
    @patch('app.database.timescale_models.LiquidationEvent')
    @patch('app.database.timescale_models.PortfolioHistory')
    @patch('app.database.timescale_models.RiskMetricsHistory')
    @patch('app.database.timescale_models.SignalTimeseries')
    @patch('app.database.timescale_models.create_continuous_aggregates')
    def test_init_creates_all_hypertables(
        self, mock_aggregates, mock_signal, mock_risk, mock_portfolio,
        mock_liquidation, mock_funding, mock_orderbook, mock_trade,
        mock_ohlcv, mock_base, mock_text, mock_engine
    ):
        """Test that init_timescaledb creates all hypertables."""
        from app.database.timescale_models import init_timescaledb
        
        # Mock engine
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        
        # Call init
        result = init_timescaledb("postgresql://test")
        
        # Verify all hypertables were created
        mock_ohlcv.create_hypertable.assert_called_once()
        mock_trade.create_hypertable.assert_called_once()
        mock_orderbook.create_hypertable.assert_called_once()
        mock_funding.create_hypertable.assert_called_once()
        mock_liquidation.create_hypertable.assert_called_once()
        mock_portfolio.create_hypertable.assert_called_once()
        mock_risk.create_hypertable.assert_called_once()
        mock_signal.create_hypertable.assert_called_once()
        
        # Verify continuous aggregates were created
        mock_aggregates.assert_called_once()


class TestTimeSeriesQueries:
    """Test time-series query helpers."""
    
    def test_query_class_exists(self):
        """Test that TimeSeriesQueries class exists."""
        from app.database.timescale_models import TimeSeriesQueries
        
        assert TimeSeriesQueries is not None
    
    def test_get_ohlcv_range_method(self):
        """Test get_ohlcv_range method exists."""
        from app.database.timescale_models import TimeSeriesQueries
        
        assert hasattr(TimeSeriesQueries, 'get_ohlcv_range')
        assert callable(getattr(TimeSeriesQueries, 'get_ohlcv_range'))
    
    def test_get_recent_trades_method(self):
        """Test get_recent_trades method exists."""
        from app.database.timescale_models import TimeSeriesQueries
        
        assert hasattr(TimeSeriesQueries, 'get_recent_trades')
        assert callable(getattr(TimeSeriesQueries, 'get_recent_trades'))
    
    def test_get_portfolio_history_method(self):
        """Test get_portfolio_history method exists."""
        from app.database.timescale_models import TimeSeriesQueries
        
        assert hasattr(TimeSeriesQueries, 'get_portfolio_history')
        assert callable(getattr(TimeSeriesQueries, 'get_portfolio_history'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])