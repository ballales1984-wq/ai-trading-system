"""
Unit tests for technical analysis module
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestTechnicalAnalysis(unittest.TestCase):
    """Test cases for TechnicalAnalyzer"""
    
    def setUp(self):
        """Set up test data"""
        from technical_analysis import TechnicalAnalyzer
        self.analyzer = TechnicalAnalyzer()
        
        # Generate sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
        prices = 45000 + np.cumsum(np.random.randn(100) * 500)
        
        self.df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 100,
            'high': prices + abs(np.random.randn(100) * 200),
            'low': prices - abs(np.random.randn(100) * 200),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        analysis = self.analyzer.analyze(self.df, 'BTC/USDT')
        
        self.assertIsInstance(analysis.rsi, float)
        self.assertGreaterEqual(analysis.rsi, 0)
        self.assertLessEqual(analysis.rsi, 100)
    
    def test_ema_calculation(self):
        """Test EMA calculation"""
        analysis = self.analyzer.analyze(self.df, 'BTC/USDT')
        
        self.assertIsInstance(analysis.ema_short, float)
        self.assertIsInstance(analysis.ema_medium, float)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        analysis = self.analyzer.analyze(self.df, 'BTC/USDT')
        
        self.assertIsInstance(analysis.bb_upper, float)
        self.assertIsInstance(analysis.bb_middle, float)
        self.assertIsInstance(analysis.bb_lower, float)
        
        # Upper should be greater than lower
        self.assertGreater(analysis.bb_upper, analysis.bb_lower)
    
    def test_macd(self):
        """Test MACD calculation"""
        analysis = self.analyzer.analyze(self.df, 'BTC/USDT')
        
        self.assertIsInstance(analysis.macd, float)
        self.assertIsInstance(analysis.macd_signal, float)
        self.assertIsInstance(analysis.macd_histogram, float)
    
    def test_analysis_output(self):
        """Test that analysis returns all expected fields"""
        analysis = self.analyzer.analyze(self.df, 'BTC/USDT')
        
        # Check all attributes exist
        self.assertTrue(hasattr(analysis, 'rsi'))
        self.assertTrue(hasattr(analysis, 'ema_short'))
        self.assertTrue(hasattr(analysis, 'ema_medium'))
        self.assertTrue(hasattr(analysis, 'bb_upper'))
        self.assertTrue(hasattr(analysis, 'macd'))
        self.assertTrue(hasattr(analysis, 'technical_score'))
        self.assertTrue(hasattr(analysis, 'trend'))


class TestDataCollector(unittest.TestCase):
    """Test cases for DataCollector"""
    
    def test_initialization(self):
        """Test DataCollector initialization"""
        from data_collector import DataCollector
        
        collector = DataCollector(simulation=True)
        
        self.assertIsNotNone(collector)
        self.assertTrue(collector.simulation)
    
    def test_supported_symbols(self):
        """Test getting supported symbols"""
        from data_collector import DataCollector
        
        collector = DataCollector(simulation=True)
        symbols = collector.get_supported_symbols()
        
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)
        self.assertIn('BTC/USDT', symbols)
    
    def test_price_fetch(self):
        """Test fetching current price"""
        from data_collector import DataCollector
        
        collector = DataCollector(simulation=True)
        price = collector.fetch_current_price('BTC/USDT')
        
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)


class TestDecisionEngine(unittest.TestCase):
    """Test cases for DecisionEngine"""
    
    def setUp(self):
        """Set up test data"""
        from data_collector import DataCollector
        from decision_engine import DecisionEngine
        
        self.collector = DataCollector(simulation=True)
        self.engine = DecisionEngine(self.collector)
    
    def test_signal_generation(self):
        """Test signal generation"""
        signals = self.engine.generate_signals(['BTC/USDT'])
        
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)
    
    def test_signal_attributes(self):
        """Test that signals have required attributes"""
        signals = self.engine.generate_signals(['BTC/USDT'])
        
        if signals:
            signal = signals[0]
            self.assertTrue(hasattr(signal, 'symbol'))
            self.assertTrue(hasattr(signal, 'action'))
            self.assertTrue(hasattr(signal, 'confidence'))
            self.assertTrue(hasattr(signal, 'current_price'))


class TestTradingSimulator(unittest.TestCase):
    """Test cases for TradingSimulator"""
    
    def test_initialization(self):
        """Test simulator initialization"""
        from trading_simulator import TradingSimulator
        
        simulator = TradingSimulator(initial_balance=10000)
        
        self.assertEqual(simulator.portfolio.balance, 10000)
        self.assertEqual(simulator.portfolio.initial_balance, 10000)
    
    def test_portfolio_state(self):
        """Test getting portfolio state"""
        from trading_simulator import TradingSimulator
        
        simulator = TradingSimulator(initial_balance=10000)
        state = simulator.get_portfolio_state()
        
        self.assertIsInstance(state, dict)
        self.assertIn('balance', state)
        self.assertIn('total_value', state)


if __name__ == '__main__':
    unittest.main()
