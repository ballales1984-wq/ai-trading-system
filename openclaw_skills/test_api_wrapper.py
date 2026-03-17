"""
Unit tests for OpenClaw API Wrapper
====================================
Tests for the QuantTradingAPI class without requiring backend
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestQuantTradingAPI:
    """Test suite for QuantTradingAPI class"""
    
    @pytest.fixture
    def api(self):
        """Create API instance with mock URL"""
        from openclaw_skills.api_wrapper import QuantTradingAPI
        return QuantTradingAPI(base_url="http://test.local/api/v1")
    
    def test_api_initialization(self, api):
        """Test API initializes with correct defaults"""
        assert api.base_url == "http://test.local/api/v1"
        assert api.timeout == 30
        assert api._token is None
    
    def test_get_headers_without_token(self, api):
        """Test headers without authentication"""
        headers = api._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers
    
    def test_get_headers_with_token(self, api):
        """Test headers with authentication"""
        api._token = "test_token_123"
        headers = api._get_headers()
        assert headers["Authorization"] == "Bearer test_token_123"
    
    @patch('requests.post')
    def test_login_success(self, mock_post, api):
        """Test successful login"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "jwt_token_abc",
            "token_type": "bearer"
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = api.login("admin", "password123")
        
        assert result["access_token"] == "jwt_token_abc"
        assert api._token == "jwt_token_abc"
        mock_post.assert_called_once()
    
    @patch('requests.get')
    def test_get_risk_metrics(self, mock_get, api):
        """Test getting risk metrics"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = {
            "var_1d": 12500.0,
            "var_5d": 28000.0,
            "cvar_1d": 18750.0,
            "volatility": 0.25,
            "sharpe_ratio": 1.85
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = api.get_risk_metrics()
        
        assert result["var_1d"] == 12500.0
        assert result["sharpe_ratio"] == 1.85
    
    @patch('requests.post')
    def test_check_order_risk_approved(self, mock_post, api):
        """Test order risk check - approved"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = {
            "order_id": "ord_123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.5,
            "price": 65000.0,
            "risk_score": 45.0,
            "approved": True,
            "rejection_reasons": [],
            "warnings": []
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = api.check_order_risk("BTCUSDT", "BUY", 0.5, 65000)
        
        assert result["approved"] is True
        assert result["risk_score"] == 45.0
    
    @patch('requests.post')
    def test_check_order_risk_rejected(self, mock_post, api):
        """Test order risk check - rejected"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = {
            "order_id": "ord_456",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 5.0,
            "price": 65000.0,
            "risk_score": 85.0,
            "approved": False,
            "rejection_reasons": ["Risk score exceeds threshold"],
            "warnings": ["Large order"]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = api.check_order_risk("BTCUSDT", "BUY", 5.0, 65000)
        
        assert result["approved"] is False
        assert "Risk score exceeds threshold" in result["rejection_reasons"]
    
    @patch('requests.get')
    def test_get_portfolio_summary(self, mock_get, api):
        """Test portfolio summary"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = {
            "real": {
                "total_value": 100000.0,
                "daily_pnl": 1200.5,
                "unrealized_pnl": 500.0,
                "cash_balance": 25000.0
            },
            "simulated": {
                "total_value": 98432.5,
                "daily_pnl": 432.1,
                "unrealized_pnl": 812.3,
                "num_positions": 3
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = api.get_portfolio_summary()
        
        assert result["real"]["total_value"] == 100000.0
        assert result["simulated"]["num_positions"] == 3
    
    @patch('requests.get')
    def test_run_monte_carlo(self, mock_get, api):
        """Test Monte Carlo simulation"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = {
            "simulations": 10000,
            "confidence": 0.95,
            "var_1d": 12500.0,
            "cvar_1d": 18750.0,
            "worst_case": -28000.0,
            "best_case": 32000.0,
            "mean_outcome": 1500.0
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = api.run_monte_carlo(10000, 0.95)
        
        assert result["simulations"] == 10000
        assert result["var_1d"] == 12500.0
    
    @patch('requests.get')
    def test_get_risk_limits(self, mock_get, api):
        """Test risk limits"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "limit_id": "var_limit",
                "limit_type": "var",
                "limit_value": 20000.0,
                "current_value": 12500.0,
                "limit_percentage": 62.5,
                "is_breached": False,
                "severity": "green"
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = api.get_risk_limits()
        
        assert len(result) == 1
        assert result[0]["limit_id"] == "var_limit"
        assert result[0]["is_breached"] is False
    
    @patch('requests.get')
    def test_get_positions(self, mock_get, api):
        """Test getting positions"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.5,
                "entry_price": 64000.0,
                "current_price": 65000.0,
                "market_value": 32500.0,
                "unrealized_pnl": 500.0
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = api.get_positions()
        
        assert len(result) == 1
        assert result[0]["symbol"] == "BTCUSDT"
    
    @patch('requests.get')
    def test_get_positions_filtered(self, mock_get, api):
        """Test getting positions with symbol filter"""
        api._token = "test_token"
        mock_response = Mock()
        mock_response.json.return_value = [
            {"symbol": "BTCUSDT", "quantity": 0.5}
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = api.get_positions(symbol="BTCUSDT")
        
        # Verify the symbol parameter was passed
        call_args = mock_get.call_args
        assert call_args[1]["params"]["symbol"] == "BTCUSDT"


class TestAPIWrapperIntegration:
    """Integration-style tests with mocked responses"""
    
    def test_analyze_trade_approved(self):
        """Test complete trade analysis - approved"""
        from openclaw_skills.api_wrapper import QuantTradingAPI
        
        with patch.object(QuantTradingAPI, 'get_positions', return_value=[]), \
             patch.object(QuantTradingAPI, 'get_risk_metrics', return_value={"var_1d": 12500}), \
             patch.object(QuantTradingAPI, 'check_order_risk', return_value={
                 "approved": True,
                 "risk_score": 45,
                 "rejection_reasons": []
             }):
            
            api = QuantTradingAPI()
            api._token = "test"
            
            result = api.analyze_trade("BTCUSDT", "BUY", 0.5, 65000)
            
            assert result["recommendation"] == "✅ Trade approved - passes all risk gates"
    
    def test_analyze_trade_rejected(self):
        """Test complete trade analysis - rejected"""
        from openclaw_skills.api_wrapper import QuantTradingAPI
        
        with patch.object(QuantTradingAPI, 'get_positions', return_value=[]), \
             patch.object(QuantTradingAPI, 'get_risk_metrics', return_value={"var_1d": 12500}), \
             patch.object(QuantTradingAPI, 'check_order_risk', return_value={
                 "approved": False,
                 "risk_score": 85,
                 "rejection_reasons": ["Concentration limit exceeded"]
             }):
            
            api = QuantTradingAPI()
            api._token = "test"
            
            result = api.analyze_trade("BTCUSDT", "BUY", 5.0, 65000)
            
            assert "rejected" in result["recommendation"].lower()
            assert "Concentration limit exceeded" in result["risk_check"]["rejection_reasons"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=openclaw_skills", "--cov-report=term-missing"])
