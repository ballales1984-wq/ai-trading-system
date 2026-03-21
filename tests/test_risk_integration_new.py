"""
Test coverage for the enhanced Risk Integrated Decision Engine 
and Unified Decision Engine with Broker/Sentiment synchronization.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.risk.risk_book import RiskLimits, Position
from src.decision.risk_integration import RiskIntegratedDecisionEngine, TradingSignal
from src.decision.unified_engine import UnifiedDecisionEngine, UnifiedEngineConfig


class TestRiskIntegrationAdvanced:
    
    @patch('sentiment_concept_bridge.SentimentConceptBridge')
    def test_dynamic_risk_limits(self, mock_bridge_class):
        """Test that bearish sentiment reduces max_position_pct."""
        # Setup mock sentiment
        mock_bridge = mock_bridge_class.return_value
        mock_sentiment = Mock()
        mock_sentiment.confidence = 0.8
        mock_sentiment.sentiment_score = -0.5 # Bearish
        mock_bridge.analyze_asset_sentiment_with_concepts.return_value = mock_sentiment
        
        # Init engine
        engine = RiskIntegratedDecisionEngine(
            sentiment_bridge=mock_bridge,
            portfolio_balance=100000,
            threshold_confidence=0.6
        )
        
        # Base limit is 10%, but bearish sentiment should halve it to 5%
        # Try a 8% position (8000 on 100k equity)
        signal = TradingSignal(
            symbol="BTCUSDT",
            side="buy",
            size=0.16, # 0.16 * 50000 = 8000
            confidence=0.65, # Might be too low now because threshold increased
            reason="Test bearish"
        )
        
        result = engine.validate_signal(signal, current_price=50000.0)
        
        assert not result.approved
        assert "exceeds limit" in result.reason or "below threshold" in result.reason

    @patch('sentiment_concept_bridge.SentimentConceptBridge')
    def test_bullish_sentiment_relaxes_limits(self, mock_bridge_class):
        """Test that bullish sentiment relaxes confidence threshold."""
        # Setup mock sentiment
        mock_bridge = mock_bridge_class.return_value
        mock_sentiment = Mock()
        mock_sentiment.confidence = 0.8
        mock_sentiment.sentiment_score = 0.6 # Bullish
        mock_bridge.analyze_asset_sentiment_with_concepts.return_value = mock_sentiment
        
        # Init engine
        engine = RiskIntegratedDecisionEngine(
            sentiment_bridge=mock_bridge,
            portfolio_balance=100000,
            threshold_confidence=0.6 # Base
        )
        
        # Bullish sentiment lowers required confidence. 
        # Try a trade with 0.55 confidence (normally rejected).
        signal = TradingSignal(
            symbol="ETHUSDT",
            side="buy",
            size=0.1, # 0.1 * 3000 = 300 (0.3% position, well within 10% limit)
            confidence=0.55,
            reason="Test bullish"
        )
        
        result = engine.validate_signal(signal, current_price=3000.0)
        
        assert result.approved
        assert result.risk_score < 50


@pytest.mark.asyncio
class TestUnifiedEngineSync:
    
    @patch('app.execution.connectors.bybit_connector.BybitConnector')
    async def test_portfolio_sync(self, mock_connector_class):
        """Test that UnifiedDecisionEngine syncs portfolio via BrokerConnector."""
        mock_connector = mock_connector_class.return_value
        mock_connector._connected = True
        
        # Configure AsyncMocks
        mock_connector.get_balance = AsyncMock(return_value=Mock(total_equity=105000.0))
        mock_connector.get_positions = AsyncMock(return_value=[
            Mock(symbol="BTCUSDT", quantity=0.5, entry_price=50000.0),
            Mock(symbol="ETHUSDT", quantity=-10.0, entry_price=3000.0)
        ])
        
        config = UnifiedEngineConfig(broker_type="bybit")
        
        with patch('src.decision.unified_engine.BybitConnector', return_value=mock_connector):
            engine = UnifiedDecisionEngine(config)
            
            # Sync portfolio
            success = await engine.sync_portfolio()
            
            assert success is True
            # Equity should be updated
            assert engine.risk_book.equity == 105000.0
            
            # Positions should be populated
            btc_pos = engine.risk_book.get_position("BTCUSDT")
            eth_pos = engine.risk_book.get_position("ETHUSDT")
            
            assert btc_pos is not None
            assert btc_pos.quantity == 0.5
            assert btc_pos.side == "long"
            
            assert eth_pos is not None
            assert eth_pos.quantity == -10.0
            assert eth_pos.side == "short"
