"""
Unit tests for Decision Engine with ML integration
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestDecisionEngine(unittest.TestCase):
    """Test cases for DecisionEngine with ML blackbox integration"""
    
    def setUp(self):
        """Set up test data and mocks"""
        # Mock data collector
        self.mock_data_collector = Mock()
        self.mock_data_collector.get_supported_symbols.return_value = ['BTC/USDT', 'ETH/USDT']
        
        # Mock market data
        self.mock_market_data = Mock()
        self.mock_market_data.current_price = 45000.0
        
        self.mock_data_collector.fetch_market_data.return_value = self.mock_market_data
        
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
        
        self.mock_data_collector.fetch_ohlcv.return_value = self.df
        
        # Mock sentiment analyzer
        self.mock_sentiment = {
            'combined_score': 0.2,
            'confidence': 0.6,
            'sentiment': 'positive'
        }
        
    def test_decision_engine_initialization(self):
        """Test DecisionEngine initialization with ML"""
        from decision_engine import DecisionEngine
        
        with patch('decision_engine.DataCollector', return_value=self.mock_data_collector):
            with patch('decision_engine.SentimentAnalyzer') as mock_sent:
                mock_sent.return_value.get_combined_sentiment.return_value = self.mock_sentiment
                
                engine = DecisionEngine(
                    data_collector=self.mock_data_collector,
                    sentiment_analyzer=mock_sent.return_value
                )
                
                # Verify ML predictor is initialized
                self.assertIsNotNone(engine.ml_predictor)
                self.assertTrue(engine.ml_enabled)
    
    def test_ml_enabled_property(self):
        """Test ML enable/disable functionality"""
        from decision_engine import DecisionEngine
        
        with patch('decision_engine.DataCollector', return_value=self.mock_data_collector):
            with patch('decision_engine.SentimentAnalyzer') as mock_sent:
                with patch('decision_engine.get_ml_predictor') as mock_ml:
                    mock_ml.return_value = Mock()
                    
                    engine = DecisionEngine(
                        data_collector=self.mock_data_collector,
                        sentiment_analyzer=mock_sent.return_value
                    )
                    
                    # Test enable/disable
                    engine.enable_ml(False)
                    self.assertFalse(engine.ml_enabled)
                    
                    engine.enable_ml(True)
                    self.assertTrue(engine.ml_enabled)
    
    def test_is_ml_ready(self):
        """Test ML readiness check"""
        from decision_engine import DecisionEngine
        
        with patch('decision_engine.DataCollector', return_value=self.mock_data_collector):
            with patch('decision_engine.SentimentAnalyzer') as mock_sent:
                with patch('decision_engine.get_ml_predictor') as mock_ml:
                    mock_predictor = Mock()
                    mock_ml.return_value = mock_predictor
                    
                    engine = DecisionEngine(
                        data_collector=self.mock_data_collector,
                        sentiment_analyzer=mock_sent.return_value
                    )
                    
                    # Test when not trained
                    mock_predictor.is_trained = False
                    self.assertFalse(engine.is_ml_ready())
                    
                    # Test when trained but disabled
                    mock_predictor.is_trained = True
                    engine.ml_enabled = False
                    self.assertFalse(engine.is_ml_ready())
                    
                    # Test when trained and enabled
                    engine.ml_enabled = True
                    self.assertTrue(engine.is_ml_ready())
    
    def test_ml_prediction_in_signal(self):
        """Test that ML prediction is included in signal"""
        from decision_engine import DecisionEngine, TradingSignal
        
        with patch('decision_engine.DataCollector', return_value=self.mock_data_collector):
            with patch('decision_engine.SentimentAnalyzer') as mock_sent:
                mock_sent.return_value.get_combined_sentiment.return_value = self.mock_sentiment
                
                with patch('decision_engine.get_ml_predictor') as mock_ml:
                    mock_predictor = Mock()
                    mock_predictor.is_trained = True
                    mock_predictor.predict.return_value = {
                        'prediction': 1,  # Bullish
                        'confidence': 0.7
                    }
                    mock_ml.return_value = mock_predictor
                    
                    engine = DecisionEngine(
                        data_collector=self.mock_data_collector,
                        sentiment_analyzer=mock_sent.return_value
                    )
                    
                    # Generate signal with complete mocks
                    mock_tech_analysis = Mock()
                    mock_tech_analysis.technical_score = 0.6
                    mock_tech_analysis.momentum_score = 0.5
                    mock_tech_analysis.atr = 100
                    mock_tech_analysis.macd_histogram = 0.5
                    mock_tech_analysis.rsi = 55
                    mock_tech_analysis.ema_short = 45000
                    mock_tech_analysis.ema_medium = 44500
                    
                    with patch.object(engine.technical_analyzer, 'analyze', return_value=mock_tech_analysis):
                        with patch.object(engine, '_analyze_correlations', return_value={'avg_correlation': 0.3, 'correlations': {}}):
                            with patch.object(engine, '_calculate_volatility_score', return_value=0.5):
                                with patch.object(engine, '_generate_reason', return_value='Test signal'):
                                    signal = engine._generate_signal('BTC/USDT')
                                    
                                    # Verify ml_score is in the signal
                                    self.assertIsNotNone(signal.ml_score)
                                    self.assertEqual(signal.ml_score, 1.0)  # (1+1)/2 = 1.0
    
    def test_trading_signal_ml_score_field(self):
        """Test TradingSignal has ml_score field"""
        from decision_engine import TradingSignal
        
        signal = TradingSignal(
            symbol='BTC/USDT',
            asset_type='crypto',
            action='BUY',
            confidence=0.7,
            strength='STRONG',
            current_price=45000.0,
            ml_score=0.8
        )
        
        self.assertEqual(signal.ml_score, 0.8)
    
    def test_combine_factors_with_ml(self):
        """Test _combine_factors includes ML score"""
        from decision_engine import DecisionEngine
        from technical_analysis import TechnicalAnalysis
        
        with patch('decision_engine.DataCollector', return_value=self.mock_data_collector):
            with patch('decision_engine.SentimentAnalyzer') as mock_sent:
                engine = DecisionEngine(
                    data_collector=self.mock_data_collector,
                    sentiment_analyzer=mock_sent.return_value
                )
                
                # Test without ML
                mock_tech = Mock(spec=TechnicalAnalysis)
                mock_tech.technical_score = 0.6
                mock_tech.momentum_score = 0.5
                
                score = engine._combine_factors(
                    mock_tech,
                    {'combined_score': 0.0},
                    {'avg_correlation': 0.3},
                    0.5
                )
                self.assertIsNotNone(score)
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)
                
                # Test with ML score
                score_with_ml = engine._combine_factors(
                    mock_tech,
                    {'combined_score': 0.0},
                    {'avg_correlation': 0.3},
                    0.5,
                    ml_score=1.0  # Bullish
                )
                # Score should be higher with bullish ML
                self.assertGreater(score_with_ml, score)
    
    def test_calculate_confidence_with_ml(self):
        """Test _calculate_confidence includes ML confidence"""
        from decision_engine import DecisionEngine
        from technical_analysis import TechnicalAnalysis
        
        with patch('decision_engine.DataCollector', return_value=self.mock_data_collector):
            with patch('decision_engine.SentimentAnalyzer') as mock_sent:
                engine = DecisionEngine(
                    data_collector=self.mock_data_collector,
                    sentiment_analyzer=mock_sent.return_value
                )
                
                mock_tech = Mock(spec=TechnicalAnalysis)
                mock_tech.technical_score = 0.6
                
                # Test without ML
                confidence = engine._calculate_confidence(
                    mock_tech,
                    {'confidence': 0.5},
                    {},
                    0.5
                )
                self.assertGreaterEqual(confidence, 0)
                self.assertLessEqual(confidence, 1)
                
                # Test with ML confidence
                confidence_with_ml = engine._calculate_confidence(
                    mock_tech,
                    {'confidence': 0.5},
                    {},
                    0.5,
                    ml_confidence=0.9
                )
                # Confidence should be higher with high ML confidence
                self.assertGreater(confidence_with_ml, confidence)


class TestMLModelPersistence(unittest.TestCase):
    """Test cases for ML model save/load functionality"""
    
    def test_ml_model_save_load_exists(self):
        """Check if ML model has save/load methods"""
        from ml_predictor import PricePredictor
        
        predictor = PricePredictor()
        
        # Check if save/load methods exist
        self.assertTrue(hasattr(predictor, 'save_model') or 
                       hasattr(predictor, 'save') or
                       hasattr(predictor, 'to_dict'))
    
    def test_ml_model_persistence_methods(self):
        """Test ML model can be serialized"""
        from ml_predictor import PricePredictor
        
        predictor = PricePredictor()
        
        # Check if model state attributes exist
        has_rf = hasattr(predictor, 'rf_model')
        has_gb = hasattr(predictor, 'gb_model')
        has_scaler = hasattr(predictor, 'scaler')
        has_save = hasattr(predictor, 'save_model')
        has_load = hasattr(predictor, 'load_model')
        has_to_dict = hasattr(predictor, 'to_dict')
        
        # At least some of these should be true
        self.assertTrue(has_rf or has_gb or has_scaler or has_save or has_to_dict)


if __name__ == '__main__':
    unittest.main()
