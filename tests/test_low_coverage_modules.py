"""
Test coverage for low-coverage modules.

This file provides tests for modules that have low test coverage:
- sentiment_concept_bridge (new integration)
- Additional edge cases
- Error handling scenarios
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


# ==================== SENTIMENT CONCEPT BRIDGE ====================

class TestSentimentConceptBridge:
    """Tests for sentiment_concept_bridge.py"""
    
    def test_bridge_import(self):
        """Test sentiment concept bridge import"""
        try:
            from sentiment_concept_bridge import SentimentConceptBridge
            assert SentimentConceptBridge is not None
        except ImportError:
            pytest.skip("sentiment_concept_bridge not available")
    
    @patch('sentiment_concept_bridge.SENTIMENT_AVAILABLE', False)
    def test_bridge_no_sentiment(self):
        """Test bridge initialization without sentiment"""
        try:
            from sentiment_concept_bridge import SentimentConceptBridge
            # Should handle missing sentiment gracefully
            bridge = SentimentConceptBridge(enable_semantic=False)
            assert bridge is not None
        except ImportError:
            pytest.skip("sentiment_concept_bridge not available")
    
    def test_bridge_concept_detection(self):
        """Test concept detection in text"""
        try:
            from sentiment_concept_bridge import SentimentConceptBridge
            bridge = SentimentConceptBridge(enable_semantic=False)
            
            # Test basic concept detection
            text = "Bitcoin shows bullish momentum with high volatility"
            concepts = bridge.detect_concepts_in_text(text)
            
            assert isinstance(concepts, list)
        except ImportError:
            pytest.skip("sentiment_concept_bridge not available")
    
    def test_bridge_empty_text(self):
        """Test concept detection with empty text"""
        try:
            from sentiment_concept_bridge import SentimentConceptBridge
            bridge = SentimentConceptBridge(enable_semantic=False)
            
            concepts = bridge.detect_concepts_in_text("")
            
            assert concepts == []
        except ImportError:
            pytest.skip("sentiment_concept_bridge not available")


# ==================== BROKER CONNECTORS ====================

class TestBrokerConnectorsNew:
    """Tests for new broker connectors"""
    
    def test_coinbase_connector_import(self):
        """Test Coinbase connector import"""
        try:
            from app.execution.connectors import CoinbaseConnector, COINBASE_AVAILABLE
            if COINBASE_AVAILABLE:
                assert CoinbaseConnector is not None
            else:
                pytest.skip("Coinbase connector not available")
        except ImportError:
            pytest.skip("Coinbase connector not available")
    
    def test_bybit_connector_import(self):
        """Test Bybit connector import"""
        try:
            from app.execution.connectors import BybitConnector, BYBIT_AVAILABLE
            if BYBIT_AVAILABLE:
                assert BybitConnector is not None
            else:
                pytest.skip("Bybit connector not available")
        except ImportError:
            pytest.skip("Bybit connector not available")
    
    def test_coinbase_config(self):
        """Test Coinbase connector configuration"""
        try:
            from app.execution.connectors import CoinbaseConnector, COINBASE_AVAILABLE
            if not COINBASE_AVAILABLE:
                pytest.skip("Coinbase connector not available")
            
            config = {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "passphrase": "test_pass",
                "sandbox": True
            }
            
            connector = CoinbaseConnector(config)
            assert connector.api_key == "test_key"
            assert connector.sandbox == True
        except ImportError:
            pytest.skip("Coinbase connector not available")
    
    def test_bybit_config(self):
        """Test Bybit connector configuration"""
        try:
            from app.execution.connectors import BybitConnector, BYBIT_AVAILABLE
            if not BYBIT_AVAILABLE:
                pytest.skip("Bybit connector not available")
            
            config = {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "testnet": True
            }
            
            connector = BybitConnector(config)
            assert connector.api_key == "test_key"
            assert connector.testnet == True
        except ImportError:
            pytest.skip("Bybit connector not available")


# ==================== EDGE CASES ====================

class TestEdgeCases:
    """Edge case tests for various modules"""
    
    def test_concept_engine_empty_concepts(self):
        """Test concept engine with empty concept database"""
        try:
            from concept_engine import ConceptEngine
            with patch('concept_engine.FINANCIAL_CONCEPTS', {}):
                engine = ConceptEngine()
                assert engine is not None
        except ImportError:
            pytest.skip("concept_engine not available")
    
    def test_sentiment_analyzer_empty_assets(self):
        """Test sentiment analyzer with empty assets"""
        try:
            from sentiment_news import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            
            # Should handle empty list gracefully
            news = analyzer.fetch_news(assets=[])
            assert isinstance(news, list)
        except ImportError:
            pytest.skip("sentiment_news not available")
    
    def test_ml_predictor_no_data(self):
        """Test ML predictor without training data"""
        try:
            from ml_predictor import ImprovedPricePredictor
            predictor = ImprovedPricePredictor()
            
            assert predictor.is_trained == False
        except ImportError:
            pytest.skip("ml_predictor not available")


# ==================== ERROR HANDLING ====================

class TestErrorHandling:
    """Error handling tests"""
    
    @patch('requests.get')
    def test_sentiment_api_error(self, mock_get):
        """Test sentiment analyzer handles API errors"""
        try:
            from sentiment_news import SentimentAnalyzer
            mock_get.side_effect = Exception("Network error")
            
            analyzer = SentimentAnalyzer()
            # Should fall back to simulated data
            sentiment = analyzer.analyze_asset_sentiment("BTC")
            
            assert sentiment is not None
        except ImportError:
            pytest.skip("sentiment_news not available")
    
    @patch('requests.get')
    def test_fear_greed_api_error(self, mock_get):
        """Test fear greed index handles API errors"""
        try:
            from sentiment_news import SentimentAnalyzer
            mock_get.side_effect = Exception("Network error")
            
            analyzer = SentimentAnalyzer()
            result = analyzer.fetch_fear_greed_index()
            
            # Should return fallback data
            assert 'value' in result
        except ImportError:
            pytest.skip("sentiment_news not available")
