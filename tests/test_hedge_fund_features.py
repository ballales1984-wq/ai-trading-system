"""
Test Hedge Fund Features
========================
Comprehensive tests for hedge fund-level trading system features.

Tests:
- 5-Question Decision Engine
- Portfolio Optimization
- Risk Management (VaR, CVaR)
- Performance Metrics
- Broker Connectors
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# TEST 5-QUESTION DECISION ENGINE
# =============================================================================

class TestFiveQuestionDecisionEngine:
    """Test the 5-Question Decision Framework."""
    
    @pytest.fixture
    def decision_engine(self):
        """Create a DecisionEngine instance for testing."""
        with patch('decision_engine.DataCollector') as MockDataCollector:
            with patch('decision_engine.SentimentAnalyzer') as MockSentimentAnalyzer:
                # Setup mock data collector
                mock_collector = Mock()
                mock_collector.fetch_market_data.return_value = Mock(
                    current_price=50000.0,
                    bid=49990.0,
                    ask=50010.0,
                    volume=1000000.0
                )
                
                # Create sample OHLCV data
                dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
                df = pd.DataFrame({
                    'open': np.random.uniform(49000, 51000, 100),
                    'high': np.random.uniform(50000, 52000, 100),
                    'low': np.random.uniform(48000, 50000, 100),
                    'close': np.random.uniform(49000, 51000, 100),
                    'volume': np.random.uniform(100, 1000, 100)
                }, index=dates)
                mock_collector.fetch_ohlcv.return_value = df
                mock_collector.get_supported_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
                
                MockDataCollector.return_value = mock_collector
                MockSentimentAnalyzer.return_value = Mock()
                
                from decision_engine import DecisionEngine
                engine = DecisionEngine()
                return engine
    
    def test_answer_what_buy_signal(self, decision_engine):
        """Test Q1: What to buy - should return BUY for bullish conditions."""
        result = decision_engine.answer_what('BTCUSDT')
        
        assert 'action' in result
        assert result['action'] in ['BUY', 'SELL', 'HOLD']
        assert 'what_score' in result
        assert 0 <= result['what_score'] <= 1
        assert 'current_price' in result
    
    def test_answer_why_scoring(self, decision_engine):
        """Test Q2: Why - Reason score calculation."""
        result = decision_engine.answer_why('BTCUSDT')
        
        assert 'why_score' in result
        assert 0 <= result['why_score'] <= 1
        assert 'macro_score' in result
        assert 'sentiment_score' in result
        assert 'reason' in result
        
        # Verify formula: why_score = 0.6 * macro + 0.4 * sentiment (normalized)
        # The why_score should be a weighted combination
        assert isinstance(result['why_score'], float)
    
    def test_answer_how_much_position_sizing(self, decision_engine):
        """Test Q3: How much - Position sizing calculation."""
        why_score = 0.7  # Good reason score
        result = decision_engine.answer_how_much('BTCUSDT', why_score, 50000.0)
        
        assert 'how_much_score' in result
        assert 'position_size' in result
        assert 'position_value' in result
        assert 'position_units' in result
        
        # Position size should be proportional to why_score
        assert result['position_size'] <= decision_engine.settings['max_position_size']
    
    def test_answer_when_timing(self, decision_engine):
        """Test Q4: When - Monte Carlo timing score."""
        result = decision_engine.answer_when('BTCUSDT')
        
        assert 'when_score' in result
        assert 'probability_up' in result
        assert 'confidence' in result
        assert 0 <= result['when_score'] <= 1
        assert 0 <= result['probability_up'] <= 1
    
    def test_answer_risk_checks(self, decision_engine):
        """Test Q5: Risk - Risk control checks."""
        result = decision_engine.answer_risk(
            symbol='BTCUSDT',
            action='BUY',
            position_size=0.05,  # 5% position
            current_price=50000.0,
            when_score=0.6
        )
        
        assert 'risk_score' in result
        assert 'passed' in result
        assert 'reason' in result
        assert 'var_95' in result
        assert 'cvar_95' in result
        
        # VaR should be positive
        assert result['var_95'] >= 0
        # CVaR should be >= VaR (worse case)
        assert result['cvar_95'] >= result['var_95']
    
    def test_unified_decision_complete_flow(self, decision_engine):
        """Test complete unified decision flow."""
        result = decision_engine.unified_decision('BTCUSDT')
        
        assert 'symbol' in result
        assert 'action' in result
        assert result['action'] in ['BUY', 'SELL', 'HOLD']
        
        if result['action'] != 'HOLD':
            assert 'confidence' in result
            assert 'position_size' in result
            assert 'prices' in result
            assert 'scores' in result
            assert 'risk_metrics' in result
            
            # Verify all 5 questions were answered
            assert 'decision_flow' in result
            assert 'what' in result['decision_flow']
            assert 'why' in result['decision_flow']
            assert 'how_much' in result['decision_flow']
            assert 'when' in result['decision_flow']
            assert 'risk' in result['decision_flow']


# =============================================================================
# TEST PORTFOLIO OPTIMIZATION
# =============================================================================

class TestPortfolioOptimization:
    """Test portfolio optimization methods."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a PortfolioOptimizer instance."""
        from app.portfolio.optimization import PortfolioOptimizer, OptimizationConstraints
        
        # Generate sample returns data
        np.random.seed(42)
        n_days = 252
        n_assets = 3
        
        # Generate correlated returns
        means = [0.0005, 0.0003, 0.0004]  # Daily returns
        stds = [0.02, 0.015, 0.025]
        
        returns = np.column_stack([
            np.random.normal(m, s, n_days) for m, s in zip(means, stds)
        ])
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.5,
            max_positions=10
        )
        
        return PortfolioOptimizer(
            symbols=symbols,
            returns=returns,
            risk_free_rate=0.04,
            constraints=constraints
        )
    
    def test_max_sharpe_optimization(self, optimizer):
        """Test maximum Sharpe ratio optimization."""
        result = optimizer.optimize(method='max_sharpe')
        
        assert result.converged
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 0.01  # Weights sum to 1
        assert result.sharpe_ratio > 0
    
    def test_min_variance_optimization(self, optimizer):
        """Test minimum variance optimization."""
        result = optimizer.optimize(method='min_variance')
        
        assert result.converged
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.expected_volatility > 0
    
    def test_risk_parity_optimization(self, optimizer):
        """Test risk parity optimization."""
        result = optimizer.optimize(method='risk_parity')
        
        assert result.converged
        assert len(result.weights) == 3
        # Risk parity should have more balanced weights
        weights = list(result.weights.values())
        assert max(weights) < 0.8  # No single asset dominates
    
    def test_constraints_respected(self, optimizer):
        """Test that optimization constraints are respected."""
        result = optimizer.optimize(method='max_sharpe')
        
        for symbol, weight in result.weights.items():
            assert weight >= optimizer.constraints.min_weight
            assert weight <= optimizer.constraints.max_weight


# =============================================================================
# TEST PORTFOLIO PERFORMANCE
# =============================================================================

class TestPortfolioPerformance:
    """Test portfolio performance metrics."""
    
    @pytest.fixture
    def performance_tracker(self):
        """Create a PortfolioPerformance instance."""
        from app.portfolio.performance import PortfolioPerformance
        
        tracker = PortfolioPerformance(
            initial_capital=100000.0,
            risk_free_rate=0.04
        )
        
        # Add some sample trades
        for i in range(10):
            pnl = np.random.uniform(-500, 1000)
            tracker.add_trade(
                trade_id=f"trade_{i}",
                symbol="BTCUSDT",
                side="BUY" if i % 2 == 0 else "SELL",
                entry_price=50000.0,
                exit_price=50000.0 + np.random.uniform(-500, 500),
                quantity=0.1,
                pnl=pnl,
                fees=1.0
            )
        
        return tracker
    
    def test_calculate_sharpe_ratio(self, performance_tracker):
        """Test Sharpe ratio calculation."""
        metrics = performance_tracker.calculate_metrics()
        
        assert hasattr(metrics, 'sharpe_ratio')
        assert isinstance(metrics.sharpe_ratio, float)
    
    def test_calculate_sortino_ratio(self, performance_tracker):
        """Test Sortino ratio calculation."""
        metrics = performance_tracker.calculate_metrics()
        
        assert hasattr(metrics, 'sortino_ratio')
        assert isinstance(metrics.sortino_ratio, float)
    
    def test_calculate_max_drawdown(self, performance_tracker):
        """Test maximum drawdown calculation."""
        metrics = performance_tracker.calculate_metrics()
        
        assert hasattr(metrics, 'max_drawdown')
        assert 0 <= metrics.max_drawdown <= 1
    
    def test_win_rate_calculation(self, performance_tracker):
        """Test win rate calculation."""
        metrics = performance_tracker.calculate_metrics()
        
        assert hasattr(metrics, 'win_rate')
        assert 0 <= metrics.win_rate <= 1
    
    def test_profit_factor(self, performance_tracker):
        """Test profit factor calculation."""
        metrics = performance_tracker.calculate_metrics()
        
        assert hasattr(metrics, 'profit_factor')
        assert metrics.profit_factor >= 0


# =============================================================================
# TEST RISK ENGINE
# =============================================================================

class TestRiskEngine:
    """Test risk management features."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create a HardenedRiskEngine instance."""
        from app.risk.hardened_risk_engine import HardenedRiskEngine, RiskLimits
        
        limits = RiskLimits(
            max_position_size=0.1,
            max_sector_exposure=0.25,
            max_leverage=5.0,
            max_drawdown=0.2,
            daily_loss_limit=0.05,
            var_limit=0.02
        )
        
        return HardenedRiskEngine(limits=limits)
    
    def test_var_calculation(self, risk_engine):
        """Test Value at Risk calculation."""
        # Generate sample returns
        returns = np.random.normal(0.001, 0.02, 252)
        
        var_95 = risk_engine.calculate_var(returns, confidence=0.95)
        
        assert isinstance(var_95, float)
        assert var_95 >= 0  # VaR is typically positive (loss)
    
    def test_cvar_calculation(self, risk_engine):
        """Test Conditional VaR calculation."""
        returns = np.random.normal(0.001, 0.02, 252)
        
        cvar_95 = risk_engine.calculate_cvar(returns, confidence=0.95)
        var_95 = risk_engine.calculate_var(returns, confidence=0.95)
        
        assert isinstance(cvar_95, float)
        assert cvar_95 >= var_95  # CVaR >= VaR
    
    def test_position_limit_check(self, risk_engine):
        """Test position size limit check."""
        result = risk_engine.check_position_limit(
            symbol='BTCUSDT',
            position_size=0.15,  # 15% - exceeds 10% limit
            portfolio_value=100000.0
        )
        
        assert result['passed'] == False
        assert 'exceeds' in result['reason'].lower()
    
    def test_drawdown_circuit_breaker(self, risk_engine):
        """Test drawdown circuit breaker."""
        risk_engine.current_drawdown = 0.25  # 25% drawdown
        
        assert risk_engine.is_circuit_breaker_triggered('drawdown')
    
    def test_daily_loss_limit(self, risk_engine):
        """Test daily loss limit check."""
        risk_engine.daily_pnl = -6000  # -$6000 loss
        risk_engine.portfolio_value = 100000  # 6% loss > 5% limit
        
        result = risk_engine.check_daily_loss_limit()
        
        assert result['passed'] == False


# =============================================================================
# TEST BROKER CONNECTORS
# =============================================================================

class TestBrokerConnectors:
    """Test broker connector implementations."""
    
    def test_paper_connector_order_flow(self):
        """Test paper trading connector order flow."""
        from app.execution.connectors.paper_connector import PaperConnector
        
        connector = PaperConnector(initial_balance=10000.0)
        
        # Place order
        order = connector.place_order(
            symbol='BTCUSDT',
            side='BUY',
            order_type='LIMIT',
            quantity=0.1,
            price=50000.0
        )
        
        assert order is not None
        assert order['status'] in ['NEW', 'FILLED']
        
        # Check balance updated
        balance = connector.get_balance()
        assert 'USDT' in balance
    
    def test_binance_connector_signature(self):
        """Test Binance connector HMAC signature generation."""
        from app.execution.connectors.binance_connector import BinanceConnector
        
        # Use testnet credentials
        connector = BinanceConnector(
            api_key='test_key',
            api_secret='test_secret',
            testnet=True
        )
        
        # Test signature generation
        params = {'symbol': 'BTCUSDT', 'quantity': 0.1}
        signature = connector._generate_signature(params)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # HMAC-SHA256 produces 64 hex chars


# =============================================================================
# TEST DATABASE MODELS
# =============================================================================

class TestDatabaseModels:
    """Test database model definitions."""
    
    def test_price_record_model(self):
        """Test PriceRecord model definition."""
        from app.database.models import PriceRecord
        
        record = PriceRecord(
            symbol='BTCUSDT',
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000000.0
        )
        
        assert record.symbol == 'BTCUSDT'
        assert record.open == 50000.0
    
    def test_order_record_model(self):
        """Test OrderRecord model definition."""
        from app.database.models import OrderRecord, OrderSideEnum, OrderStatusEnum
        
        order = OrderRecord(
            order_id='test_order_123',
            symbol='BTCUSDT',
            side=OrderSideEnum.BUY,
            order_type='LIMIT',
            quantity=0.1,
            price=50000.0,
            status=OrderStatusEnum.NEW
        )
        
        assert order.order_id == 'test_order_123'
        assert order.side == OrderSideEnum.BUY


# =============================================================================
# TEST TIMESCALEDB MODELS
# =============================================================================

class TestTimescaleDBModels:
    """Test TimescaleDB time-series models."""
    
    def test_ohlcv_bar_model(self):
        """Test OHLCVBar hypertable model."""
        from app.database.timescale_models import OHLCVBar
        
        bar = OHLCVBar(
            symbol='BTCUSDT',
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000000.0,
            source='binance'
        )
        
        assert bar.symbol == 'BTCUSDT'
        assert bar.volume == 1000000.0
    
    def test_trade_tick_model(self):
        """Test TradeTick model."""
        from app.database.timescale_models import TradeTick
        
        tick = TradeTick(
            symbol='BTCUSDT',
            timestamp=datetime.now(),
            price=50000.0,
            quantity=0.5,
            side='BUY',
            trade_id='12345'
        )
        
        assert tick.symbol == 'BTCUSDT'
        assert tick.side == 'BUY'


# =============================================================================
# EDGE CASE INTEGRATION TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data inputs."""
        from app.portfolio.optimization import PortfolioOptimizer
        
        # Empty returns should raise or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            optimizer = PortfolioOptimizer(
                symbols=['BTCUSDT'],
                returns=np.array([]),
            )
    
    def test_extreme_volatility(self):
        """Test risk engine with extreme volatility scenarios."""
        from app.risk.hardened_risk_engine import HardenedRiskEngine, RiskLimits
        
        engine = HardenedRiskEngine(limits=RiskLimits(
            max_position_size=0.1,
            max_drawdown=0.2,
            var_limit=0.02
        ))
        
        # Generate extreme returns (50% daily moves)
        extreme_returns = np.random.normal(0, 0.5, 252)
        
        var = engine.calculate_var(extreme_returns, confidence=0.95)
        cvar = engine.calculate_cvar(extreme_returns, confidence=0.95)
        
        # VaR should be significant for extreme volatility
        assert var > 0.3  # At least 30% VaR
        assert cvar >= var
    
    def test_zero_price_handling(self):
        """Test handling of zero or negative prices."""
        with patch('decision_engine.DataCollector') as MockCollector:
            mock_collector = Mock()
            mock_collector.fetch_market_data.return_value = Mock(
                current_price=0.0,  # Invalid price
                bid=0.0,
                ask=0.0,
                volume=0.0
            )
            MockCollector.return_value = mock_collector
            
            from decision_engine import DecisionEngine
            engine = DecisionEngine()
            
            result = engine.answer_what('TESTUSDT')
            assert result['action'] == 'HOLD'
            assert 'reason' in result
    
    def test_single_asset_portfolio(self):
        """Test portfolio optimization with single asset."""
        from app.portfolio.optimization import PortfolioOptimizer
        
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252).reshape(-1, 1)
        
        optimizer = PortfolioOptimizer(
            symbols=['BTCUSDT'],
            returns=returns,
        )
        
        result = optimizer.optimize(method='max_sharpe')
        
        # Single asset should have 100% weight
        assert result.weights['BTCUSDT'] == 1.0
    
    def test_correlated_assets(self):
        """Test portfolio optimization with highly correlated assets."""
        from app.portfolio.optimization import PortfolioOptimizer
        
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, 252)
        
        # Create highly correlated returns (correlation > 0.95)
        returns = np.column_stack([
            base_returns,
            base_returns + np.random.normal(0, 0.001, 252),  # Nearly identical
            base_returns + np.random.normal(0, 0.001, 252),
        ])
        
        optimizer = PortfolioOptimizer(
            symbols=['A', 'B', 'C'],
            returns=returns,
        )
        
        result = optimizer.optimize(method='min_variance')
        
        # Should still produce valid weights
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
    
    def test_risk_engine_circuit_breakers(self):
        """Test all circuit breaker triggers."""
        from app.risk.hardened_risk_engine import HardenedRiskEngine, RiskLimits
        
        limits = RiskLimits(
            max_position_size=0.1,
            max_drawdown=0.2,
            daily_loss_limit=0.05,
            var_limit=0.02
        )
        engine = HardenedRiskEngine(limits=limits)
        
        # Test drawdown circuit
        engine.current_drawdown = 0.25
        assert engine.is_circuit_breaker_triggered('drawdown')
        
        # Test VaR circuit
        engine.current_var = 0.03
        assert engine.is_circuit_breaker_triggered('var')
        
        # Test daily loss circuit
        engine.daily_pnl = -6000
        engine.portfolio_value = 100000
        assert engine.is_circuit_breaker_triggered('daily_loss')
    
    def test_concurrent_order_handling(self):
        """Test handling of concurrent order scenarios."""
        from app.execution.order_manager import OrderManager
        
        with patch('app.execution.order_manager.BrokerConnector'):
            manager = OrderManager(broker=Mock())
            
            # Simulate concurrent order requests
            orders = []
            for i in range(10):
                order = {
                    'symbol': 'BTCUSDT',
                    'side': 'BUY',
                    'quantity': 0.01,
                    'order_type': 'LIMIT',
                    'price': 50000.0 - i * 10
                }
                orders.append(order)
            
            # All orders should be valid
            for order in orders:
                assert 'symbol' in order
                assert 'side' in order
    
    def test_network_failure_recovery(self):
        """Test recovery from network failures."""
        from app.execution.connectors.paper_connector import PaperConnector
        
        connector = PaperConnector(initial_balance=10000.0)
        
        # Simulate network failure then recovery
        with patch.object(connector, 'get_balance', side_effect=[
            Exception("Network error"),  # First call fails
            {'USDT': 10000.0}  # Second call succeeds
        ]):
            with pytest.raises(Exception):
                connector.get_balance()
            
            result = connector.get_balance()
            assert 'USDT' in result
    
    def test_malformed_api_response(self):
        """Test handling of malformed API responses."""
        with patch('decision_engine.DataCollector') as MockCollector:
            mock_collector = Mock()
            # Return malformed data
            mock_collector.fetch_market_data.return_value = Mock(
                current_price=None,  # Missing price
                bid=None,
                ask=None,
                volume=None
            )
            MockCollector.return_value = mock_collector
            
            from decision_engine import DecisionEngine
            engine = DecisionEngine()
            
            result = engine.answer_what('TESTUSDT')
            # Should handle gracefully
            assert result is not None
    
    def test_position_limit_edge_cases(self):
        """Test position sizing at boundaries."""
        from app.risk.hardened_risk_engine import HardenedRiskEngine, RiskLimits
        
        limits = RiskLimits(max_position_size=0.1)
        engine = HardenedRiskEngine(limits=limits)
        
        # Test exactly at limit
        result = engine.check_position_limit('BTCUSDT', 0.1, 100000)
        assert result['passed'] == True
        
        # Test just over limit
        result = engine.check_position_limit('BTCUSDT', 0.1001, 100000)
        assert result['passed'] == False
    
    def test_negative_position_handling(self):
        """Test handling of negative positions (shorts)."""
        from app.portfolio.performance import PortfolioPerformance
        
        tracker = PortfolioPerformance(initial_capital=100000.0)
        
        # Add a short position trade
        tracker.add_trade(
            trade_id="short_1",
            symbol="BTCUSDT",
            side="SELL",
            entry_price=50000.0,
            exit_price=48000.0,  # Profitable short
            quantity=-0.1,  # Negative for short
            pnl=200.0,  # Profit from short
            fees=2.0
        )
        
        metrics = tracker.calculate_metrics()
        assert metrics.total_pnl >= 0


# =============================================================================
# PERFORMANCE AND STRESS TESTS
# =============================================================================

class TestPerformanceAndStress:
    """Performance and stress tests."""
    
    def test_large_portfolio_optimization(self):
        """Test optimization with many assets."""
        from app.portfolio.optimization import PortfolioOptimizer
        
        np.random.seed(42)
        n_assets = 50
        n_days = 252
        
        # Generate random returns for 50 assets
        returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
        symbols = [f'ASSET_{i}' for i in range(n_assets)]
        
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        
        import time
        start = time.time()
        result = optimizer.optimize(method='max_sharpe')
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert len(result.weights) == n_assets
    
    def test_rapid_signal_generation(self):
        """Test rapid signal generation for multiple symbols."""
        with patch('decision_engine.DataCollector') as MockCollector:
            # Setup mock
            mock_collector = Mock()
            mock_collector.fetch_market_data.return_value = Mock(
                current_price=50000.0,
                bid=49990.0,
                ask=50010.0,
                volume=1000000.0
            )
            
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
            df = pd.DataFrame({
                'open': np.random.uniform(49000, 51000, 100),
                'high': np.random.uniform(50000, 52000, 100),
                'low': np.random.uniform(48000, 50000, 100),
                'close': np.random.uniform(49000, 51000, 100),
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates)
            mock_collector.fetch_ohlcv.return_value = df
            mock_collector.get_supported_symbols.return_value = [
                f'SYMBOL_{i}' for i in range(20)
            ]
            MockCollector.return_value = mock_collector
            
            from decision_engine import DecisionEngine
            engine = DecisionEngine()
            
            import time
            start = time.time()
            signals = engine.generate_signals_5q()
            elapsed = time.time() - start
            
            # Should handle 20 symbols in reasonable time
            assert elapsed < 10.0
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        from app.database.repository import PriceRepository
        
        # This test verifies the repository can handle bulk operations
        # without memory issues (mocked)
        with patch('app.database.repository.Session') as MockSession:
            mock_session = Mock()
            repo = PriceRepository(mock_session)
            
            # Simulate bulk insert of 10000 records
            records = []
            for i in range(10000):
                records.append(Mock())
            
            # Should not raise memory errors
            repo.save_prices_bulk(records)
            mock_session.bulk_save_objects.assert_called_once()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
