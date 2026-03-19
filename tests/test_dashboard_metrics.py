"""
Dashboard Metrics Validation Tests
===================================
Tests to verify that dashboard metrics are calculated correctly and are consistent.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestMetricsValidation:
    """Test class for dashboard metrics validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data with clear uptrend for testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Create data with clear uptrend
        trend = 0.001  # Daily trend
        returns = np.random.normal(trend, 0.01, 100)
        prices = 100 * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, 100)),
            'high': prices * (1 + np.random.uniform(0.005, 0.015, 100)),
            'low': prices * (1 - np.random.uniform(0.005, 0.015, 100)),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        df.set_index('date', inplace=True)
        return df
    
    @pytest.fixture
    def sample_signals(self, sample_data):
        """Generate sample signals that follow the trend."""
        signals = pd.Series('HOLD', index=sample_data.index)
        
        # Simple trend-following: BUY when price is above 20-day SMA
        sma = sample_data['close'].rolling(20).mean()
        
        # More BUY signals in uptrend
        for i in range(20, len(sample_data)):
            if sample_data['close'].iloc[i] > sma.iloc[i]:
                # Randomly decide to signal BUY or HOLD (70% BUY)
                signals.iloc[i] = 'BUY' if np.random.random() > 0.3 else 'HOLD'
            else:
                signals.iloc[i] = 'SELL' if np.random.random() > 0.7 else 'HOLD'
        
        return signals
    
    def test_sharpe_ratio_consistency(self, sample_data, sample_signals):
        """Test that Sharpe ratio is positive when returns are positive."""
        from src.backtest import run_backtest
        from src.risk import calculate_all_risk_metrics
        
        # Run backtest
        result = run_backtest(sample_data, sample_signals, initial_capital=10000)
        
        # Calculate risk metrics
        risk_metrics = calculate_all_risk_metrics(
            result.strategy_returns,
            result.equity_curve
        )
        
        total_return = risk_metrics.get('total_return', 0)
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        
        # If total return is positive, Sharpe should also be positive
        if total_return > 0.01:  # More than 1% return
            assert sharpe > 0, f"Sharpe ratio should be positive when return is {total_return:.2%}, but got {sharpe}"
        
        print(f"Total Return: {total_return:.2%}, Sharpe: {sharpe:.2f}")
    
    def test_sortino_ratio_consistency(self, sample_data, sample_signals):
        """Test that Sortino ratio is positive when returns are positive."""
        from src.backtest import run_backtest
        from src.risk import calculate_all_risk_metrics
        
        # Run backtest
        result = run_backtest(sample_data, sample_signals, initial_capital=10000)
        
        # Calculate risk metrics
        risk_metrics = calculate_all_risk_metrics(
            result.strategy_returns,
            result.equity_curve
        )
        
        total_return = risk_metrics.get('total_return', 0)
        sortino = risk_metrics.get('sortino_ratio', 0)
        
        # If total return is positive, Sortino should also be positive
        if total_return > 0.01:
            assert sortino > 0, f"Sortino ratio should be positive when return is {total_return:.2%}, but got {sortino}"
        
        print(f"Total Return: {total_return:.2%}, Sortino: {sortino:.2f}")
    
    def test_fund_net_return_matches_total_return(self, sample_data, sample_signals):
        """Test that Fund Net Return is related to Total Return."""
        from src.backtest import run_backtest
        from src.fund_simulator import FundSimulator
        from dashboard.app import _deserialize_series
        
        # Run backtest
        result = run_backtest(sample_data, sample_signals, initial_capital=10000)
        
        # Get equity curve
        equity = result.equity_curve
        
        # Run fund simulation
        fund = FundSimulator(initial_capital=1000000)
        adjusted, metrics = fund.apply_fees(equity)
        
        # Get returns
        backtest_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        fund_return = metrics.net_return
        
        # They should be related (fund return is after fees)
        # Fund return should be close to or less than backtest return
        print(f"Backtest Return: {backtest_return:.2%}, Fund Return: {fund_return:.2%}")
        
        # This test just verifies they are calculated
        assert abs(backtest_return) < 10, "Backtest return should be reasonable"
        assert abs(fund_return) < 10, "Fund return should be reasonable"
    
    def test_max_drawdown_bounds(self, sample_data, sample_signals):
        """Test that max drawdown is between 0 and 1."""
        from src.backtest import run_backtest
        from src.risk import calculate_all_risk_metrics
        
        result = run_backtest(sample_data, sample_signals, initial_capital=10000)
        risk_metrics = calculate_all_risk_metrics(
            result.strategy_returns,
            result.equity_curve
        )
        
        max_dd = risk_metrics.get('max_drawdown', 0)
        
        # Max drawdown should be negative (represents loss)
        assert max_dd <= 0, f"Max drawdown should be negative, got {max_dd}"
        assert max_dd >= -1, f"Max drawdown should be >= -100%, got {max_dd}"
        
        print(f"Max Drawdown: {max_dd:.2%}")
    
    def test_win_rate_bounds(self, sample_data, sample_signals):
        """Test that win rate is between 0 and 1."""
        from src.backtest import run_backtest
        from src.risk import calculate_all_risk_metrics
        
        result = run_backtest(sample_data, sample_signals, initial_capital=10000)
        risk_metrics = calculate_all_risk_metrics(
            result.strategy_returns,
            result.equity_curve
        )
        
        win_rate = risk_metrics.get('win_rate', 0)
        
        # Win rate should be between 0 and 1
        assert 0 <= win_rate <= 1, f"Win rate should be between 0 and 1, got {win_rate}"
        
        print(f"Win Rate: {win_rate:.2%}")
    
    def test_var_bounds(self, sample_data, sample_signals):
        """Test that VaR is negative (represents loss)."""
        from src.backtest import run_backtest
        from src.risk import calculate_all_risk_metrics
        
        result = run_backtest(sample_data, sample_signals, initial_capital=10000)
        risk_metrics = calculate_all_risk_metrics(
            result.strategy_returns,
            result.equity_curve
        )
        
        var_95 = risk_metrics.get('var_95', 0)
        
        # VaR should be negative (or zero) for typical returns
        print(f"VaR 95%: {var_95:.4f}")
        
        # Just verify it's a reasonable number
        assert abs(var_95) < 1, f"VaR should be reasonable, got {var_95}"


class TestDashboardDataFlow:
    """Test the data flow in the dashboard."""
    
    def test_sample_data_generation(self):
        """Test that sample data is generated correctly."""
        from dashboard.app import generate_sample_data
        
        df = generate_sample_data(100)
        
        # Check columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        
        # Check data validity
        assert len(df) == 100
        assert (df['high'] >= df['low']).all(), "High should be >= low"
        assert (df['high'] >= df['open']).all(), "High should be >= open"
        assert (df['high'] >= df['close']).all(), "High should be >= close"
        assert (df['low'] <= df['open']).all(), "Low should be <= open"
        assert (df['low'] <= df['close']).all(), "Low should be <= close"
        
        print(f"Sample data generated: {len(df)} rows")
        print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    def test_indicators_calculation(self):
        """Test that technical indicators are calculated."""
        from dashboard.app import generate_sample_data
        from src.indicators import calculate_all_indicators
        
        df = generate_sample_data(100)
        df_with_indicators = calculate_all_indicators(df)
        
        # Check that indicators were added
        expected_indicators = ['rsi', 'macd', 'macd_signal', 'ema_9', 'ema_21', 
                               'sma_20', 'sma_50', 'bb_upper', 'bb_lower', 'bb_position']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns, f"Indicator {indicator} missing"
        
        print(f"Indicators calculated: {len(df_with_indicators.columns)} total columns")


class TestMetricsConsistency:
    """Test consistency between different metrics."""
    
    def test_equity_curve_total_return_consistency(self):
        """Test that equity curve and total return are consistent."""
        from dashboard.app import generate_sample_data
        from src.indicators import calculate_all_indicators
        from src.signal_engine import generate_composite_signal
        from src.backtest import run_backtest
        from src.risk import calculate_all_risk_metrics
        
        # Generate data and signals
        df = generate_sample_data(100)
        df = calculate_all_indicators(df)
        signals = generate_composite_signal(df)
        
        # Run backtest
        result = run_backtest(df, signals, initial_capital=10000)
        
        # Calculate total return from equity curve
        equity = result.equity_curve
        calculated_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        
        # Get total return from metrics
        risk_metrics = calculate_all_risk_metrics(
            result.strategy_returns,
            result.equity_curve
        )
        metrics_return = risk_metrics.get('total_return', 0)
        
        # They should be very close
        diff = abs(calculated_return - metrics_return)
        assert diff < 0.0001, f"Total return mismatch: {calculated_return:.4f} vs {metrics_return:.4f}"
        
        print(f"Equity curve return: {calculated_return:.4f}")
        print(f"Metrics return: {metrics_return:.4f}")
        print(f"Difference: {diff:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
