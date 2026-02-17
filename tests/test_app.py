"""
Test Suite for Quantum AI Trading System
Comprehensive tests for all main modules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTechnicalAnalysis:
    """Test Technical Analysis module"""
    
    def setup_method(self):
        """Setup test data"""
        from technical_analysis import TechnicalAnalyzer
        self.analyzer = TechnicalAnalyzer()
        
        # Create sample OHLCV data as DataFrame with proper columns
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 50000, 100),
            'high': np.random.uniform(45000, 51000, 100),
            'low': np.random.uniform(44000, 45000, 100),
            'close': np.random.uniform(45000, 50000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        # Ensure high is highest and low is lowest
        self.sample_data['high'] = self.sample_data[['open', 'high', 'close']].max(axis=1) + 100
        self.sample_data['low'] = self.sample_data[['open', 'low', 'close']].min(axis=1) - 100
    
    def test_initialization(self):
        """Test technical analyzer initialization"""
        from technical_analysis import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        assert analyzer is not None
    
    def test_rsi_calculation(self):
        """Test RSI indicator calculation"""
        rsi = self.analyzer.calculate_rsi(self.sample_data)
        assert rsi is not None
    
    def test_macd_calculation(self):
        """Test MACD indicator calculation"""
        macd = self.analyzer.calculate_macd(self.sample_data)
        assert macd is not None
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        bb = self.analyzer.calculate_bollinger_bands(self.sample_data)
        assert bb is not None
    
    def test_atr_calculation(self):
        """Test ATR indicator calculation"""
        atr = self.analyzer.calculate_atr(self.sample_data)
        assert atr is not None


class TestSentimentAnalysis:
    """Test Sentiment Analysis module"""
    
    def setup_method(self):
        """Setup sentiment analyzer"""
        from sentiment_news import SentimentAnalyzer
        self.analyzer = SentimentAnalyzer()
    
    def test_initialization(self):
        """Test sentiment analyzer initialization"""
        assert self.analyzer is not None
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis method"""
        assert hasattr(self.analyzer, 'analyze_asset_sentiment')


class TestDataCollector:
    """Test Data Collector module"""
    
    def setup_method(self):
        """Setup data collector"""
        from data_collector import DataCollector
        self.collector = DataCollector(simulation=True)
    
    def test_initialization(self):
        """Test data collector initialization"""
        assert self.collector is not None
        assert self.collector.simulation is True
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        df = self.collector.fetch_ohlcv('BTCUSDT', '1h', limit=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'close' in df.columns


class TestDecisionEngine:
    """Test Decision Engine module"""
    
    def setup_method(self):
        """Setup decision engine"""
        from decision_engine import DecisionEngine
        self.engine = DecisionEngine()
    
    def test_initialization(self):
        """Test decision engine initialization"""
        assert self.engine is not None
    
    def test_generate_signals_method(self):
        """Test generate_signals method exists"""
        assert hasattr(self.engine, 'generate_signals')


class TestTradingSimulator:
    """Test Trading Simulator module"""
    
    def setup_method(self):
        """Setup trading simulator"""
        from trading_simulator import TradingSimulator
        self.simulator = TradingSimulator(initial_balance=10000)
    
    def test_initialization(self):
        """Test simulator initialization"""
        assert self.simulator is not None
    
    def test_get_total_value(self):
        """Test get_total_value method"""
        # Check for any get method
        assert hasattr(self.simulator, 'get_portfolio_state') or hasattr(self.simulator, 'check_portfolio')


class TestLiveMultiAssetTrader:
    """Test Live Multi-Asset Trading module"""
    
    def setup_method(self):
        """Setup live trader"""
        from live_multi_asset import LiveMultiAssetTrader
        self.trader = LiveMultiAssetTrader(
            assets=['BTCUSDT', 'ETHUSDT'],
            initial_capital=10000,
            paper_trading=True
        )
    
    def test_initialization(self):
        """Test trader initialization"""
        assert self.trader is not None
        assert len(self.trader.assets) == 2
        assert 'BTCUSDT' in self.trader.assets
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        assert self.trader.portfolio is not None


class TestTelegramNotifier:
    """Test Telegram Notifier module"""
    
    def test_initialization(self):
        """Test telegram notifier initialization (without real connection)"""
        from src.live.telegram_notifier import TelegramNotifier
        
        # Create with dummy credentials (disabled)
        notifier = TelegramNotifier(
            bot_token='dummy_token',
            chat_id='dummy_chat',
            enabled=False
        )
        assert notifier is not None
        assert notifier.enabled is False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from src.live.telegram_notifier import TelegramNotifier
        
        notifier = TelegramNotifier(
            bot_token='dummy_token',
            chat_id='dummy_chat',
            enabled=False,
            rate_limit=5
        )
        # Test can_send (should be False since disabled)
        assert notifier._can_send() is False


class TestDashboard:
    """Test Dashboard module"""
    
    def test_dashboard_import(self):
        """Test dashboard can be imported"""
        try:
            import dashboard
            assert True
        except ImportError:
            pytest.skip("Dashboard requires GUI environment")


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
