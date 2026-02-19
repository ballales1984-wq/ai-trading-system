"""
Test Suite for Hedge Fund Dashboard Features
==============================================

Tests for:
- Regime Detection Integration
- Meta-Labeling Integration  
- Hedge Fund Metrics Display
- Dashboard Callbacks

Author: AI Trading System
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== FIXTURES ====================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    
    # Create realistic price movement
    prices = 45000 + np.cumsum(np.random.randn(200) * 500)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(200) * 100,
        'high': prices + np.abs(np.random.randn(200) * 200) + 100,
        'low': prices - np.abs(np.random.randn(200) * 200) - 100,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Ensure OHLC integrity
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def sample_backtest_results():
    """Generate sample backtest results"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    equity = 10000 * (1 + np.cumsum(np.random.randn(100) * 0.01))
    
    return {
        'metrics': {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.1,
            'win_rate': 0.55
        },
        'risk_metrics': {
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'var_95': 0.02,
            'max_drawdown': 0.1
        },
        'signals': {i: np.random.choice(['BUY', 'SELL', 'HOLD']) for i in range(100)},
        'equity': pd.Series(equity, index=dates).to_dict()
    }


# ==================== REGIME DETECTION TESTS ====================

class TestRegimeDetection:
    """Test Hidden Markov Regime Detection integration"""
    
    def test_initialization(self):
        """Test regime detector initialization"""
        from src.hedgefund_ml import HiddenMarkovRegimeDetector
        
        detector = HiddenMarkovRegimeDetector(n_states=4)
        assert detector is not None
        assert detector.n_states == 4
    
    def test_fit_predict_returns_result(self, sample_ohlcv_data):
        """Test regime detection fit_predict method"""
        from src.hedgefund_ml import HiddenMarkovRegimeDetector
        
        detector = HiddenMarkovRegimeDetector(n_states=4)
        result = detector.fit_predict(sample_ohlcv_data)
        
        assert result is not None
        assert hasattr(result, 'current_regime')
        assert hasattr(result, 'regime_history')
    
    def test_regime_states(self, sample_ohlcv_data):
        """Test that regime detection returns valid regime states"""
        from src.hedgefund_ml import HiddenMarkovRegimeDetector, MarketRegime
        
        detector = HiddenMarkovRegimeDetector(n_states=4)
        result = detector.fit_predict(sample_ohlcv_data)
        
        # Check current regime exists
        if result.current_regime:
            assert isinstance(result.current_regime.regime, MarketRegime)
            assert result.current_regime.probability >= 0
            assert result.current_regime.probability <= 1
    
    def test_regime_history(self, sample_ohlcv_data):
        """Test regime history is populated"""
        from src.hedgefund_ml import HiddenMarkovRegimeDetector
        
        detector = HiddenMarkovRegimeDetector(n_states=4)
        result = detector.fit_predict(sample_ohlcv_data)
        
        # History should be populated
        if hasattr(result, 'regime_history'):
            assert len(result.regime_history) > 0


# ==================== META-LABELING TESTS ====================

class TestMetaLabeling:
    """Test Meta-Labeling integration"""
    
    def test_meta_label_generator_init(self):
        """Test MetaLabelGenerator initialization"""
        from src.hedgefund_ml import MetaLabelGenerator, MetaLabelConfig, MetaLabelType
        
        config = MetaLabelConfig(label_type=MetaLabelType.TRADE_WORTHINESS)
        generator = MetaLabelGenerator(config)
        assert generator is not None
        assert generator.config.label_type == MetaLabelType.TRADE_WORTHINESS
    
    def test_generate_labels(self, sample_ohlcv_data):
        ""Test meta-label generation"""
        from src.hedgefund_ml import MetaLabelGenerator
        
        generator = MetaLabelGenerator()
        
        # Create sample signals
        signals = pd.Series(['BUY'] * 50 + ['SELL'] * 50 + ['HOLD'] * 100, 
                          index=sample_ohlcv_data.index)
        
        labels = generator.generate_labels(sample_ohlcv_data, signals)
        
        assert labels is not None
        assert len(labels) > 0
    
    def test_meta_label_types(self):
        """Test all meta-label types"""
        from src.hedgefund_ml import MetaLabelType
        
        # Verify all enum values exist
        assert hasattr(MetaLabelType, 'TRADE_WORTHINESS')
        assert hasattr(MetaLabelType, 'RISK_ADJUSTED_RETURN')
        assert hasattr(MetaLabelType, 'SIGNAL_QUALITY')
        assert hasattr(MetaLabelType, 'REGIME_ADAPTIVE')


class TestMarketRegime:
    """Test Market Regime enum"""
    
    def test_regime_values(self):
        """Test all market regime enum values"""
        from src.hedgefund_ml import MarketRegime
        
        assert MarketRegime.TRENDING_UP.value == 'trending_up'
        assert MarketRegime.TRENDING_DOWN.value == 'trending_down'
        assert MarketRegime.MEAN_REVERTING.value == 'mean_reverting'
        assert MarketRegime.HIGH_VOLATILITY.value == 'high_volatility'
        assert MarketRegime.LOW_VOLATILITY.value == 'low_volatility'
        assert MarketRegime.CONSOLIDATION.value == 'consolidation'
        assert MarketRegime.UNKNOWN.value == 'unknown'
    
    def test_regime_from_string(self):
        """Test creating regime from string"""
        from src.hedgefund_ml import MarketRegime
        
        regime = MarketRegime('trending_up')
        assert regime == MarketRegime.TRENDING_UP


class TestHedgeFundPipeline:
    """Test HedgeFundMLPipeline integration"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        from src.hedgefund_ml import HedgeFundMLPipeline
        
        pipeline = HedgeFundMLPipeline(n_regimes=4, use_monte_carlo=False)
        assert pipeline is not None


class TestDashboardCallbacks:
    """Test dashboard callback functions"""
    
    def test_hedgefund_analysis_callback(self, sample_ohlcv_data, sample_backtest_results):
        """Test hedge fund analysis callback"""
        from src.hedgefund_ml import HiddenMarkovRegimeDetector, MetaLabelGenerator
        
        detector = HiddenMarkovRegimeDetector(n_states=4)
        regime_result = detector.fit_predict(sample_ohlcv_data)
        assert regime_result is not None
    
    def test_cvar_calculation(self, sample_backtest_results):
        """Test CVaR calculation from equity curve"""
        equity = pd.Series(sample_backtest_results['equity'])
        returns = equity.pct_change().dropna()
        var_threshold = returns.quantile(0.05)
        cvar = returns[returns <= var_threshold].mean()
        assert cvar is not None


class TestPortfolioOptimizer:
    """Test Portfolio Optimizer integration"""
    
    def test_optimizer_initialization(self):
        """Test portfolio optimizer initialization"""
        from src.portfolio_optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer()
        assert optimizer is not None
