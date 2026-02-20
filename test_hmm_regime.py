"""
Test script for HMM Regime Detection Module
============================================
Tests the Hidden Markov Model regime detection functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.hmm_regime import (
    HMMRegimeDetector,
    RegimeAwareSignalGenerator,
    RegimeState,
    RegimeResult,
    get_regime_score,
    compute_returns,
    compute_volatility,
    create_regime_detector,
    HMM_AVAILABLE
)


class TestHMMRegimeDetector:
    """Test cases for HMMRegimeDetector class."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 200
        
        # Generate prices with regime changes
        returns = np.zeros(n)
        
        # Bull market (0-50)
        returns[:50] = np.random.normal(0.001, 0.01, 50)
        
        # Bear market (50-100)
        returns[50:100] = np.random.normal(-0.002, 0.02, 50)
        
        # Sideways (100-150)
        returns[100:150] = np.random.normal(0.0, 0.005, 50)
        
        # Bull market (150-200)
        returns[150:] = np.random.normal(0.0015, 0.012, 50)
        
        prices = 100 * np.exp(np.cumsum(returns))
        return prices
    
    @pytest.fixture
    def sample_returns(self, sample_prices):
        """Generate sample returns."""
        return compute_returns(sample_prices)
    
    def test_hmm_available(self):
        """Test that HMM is available."""
        assert HMM_AVAILABLE, "hmmlearn should be installed"
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = HMMRegimeDetector(n_regimes=3)
        
        assert detector.n_regimes == 3
        assert detector.covariance_type == "full"
        assert not detector.is_fitted
    
    def test_detector_fit(self, sample_returns):
        """Test fitting the detector."""
        detector = HMMRegimeDetector(n_regimes=3, n_iter=10)
        
        # Fit with returns only
        detector.fit(sample_returns)
        
        assert detector.is_fitted
        assert detector.model is not None
    
    def test_detector_fit_with_volatility(self, sample_returns):
        """Test fitting with volatility."""
        detector = HMMRegimeDetector(n_regimes=3, n_iter=10, covariance_type='diag')
        volatility = compute_volatility(sample_returns)
        
        # Use diagonal covariance to avoid positive-definite issues
        detector.fit(sample_returns, volatility)
        
        assert detector.is_fitted
    
    def test_detector_predict(self, sample_returns):
        """Test prediction."""
        detector = HMMRegimeDetector(n_regimes=3, n_iter=10)
        detector.fit(sample_returns)
        
        result = detector.predict(sample_returns)
        
        assert isinstance(result, RegimeResult)
        assert isinstance(result.current_regime, RegimeState)
        assert result.current_regime.regime in [0, 1, 2]
        assert result.current_regime.regime_name in [
            "Sideways/Low Vol", "Bull/Low Vol", "Bear/High Vol"
        ]
        assert 0 <= result.current_regime.probability <= 1
    
    def test_predict_without_fit_raises(self, sample_returns):
        """Test that prediction without fit raises error."""
        detector = HMMRegimeDetector(n_regimes=3)
        
        with pytest.raises(ValueError, match="must be fitted"):
            detector.predict(sample_returns)
    
    def test_regime_mapping(self, sample_returns):
        """Test that regimes are mapped correctly."""
        detector = HMMRegimeDetector(n_regimes=3, n_iter=10)
        detector.fit(sample_returns)
        
        # Check that regime mapping exists
        assert len(detector._regime_mapping) == 3
    
    def test_expected_duration(self, sample_returns):
        """Test expected duration calculation."""
        detector = HMMRegimeDetector(n_regimes=3, n_iter=10)
        detector.fit(sample_returns)
        
        for regime in range(3):
            duration = detector.get_expected_duration(regime)
            assert duration > 0


class TestRegimeAwareSignalGenerator:
    """Test cases for RegimeAwareSignalGenerator."""
    
    @pytest.fixture
    def detector(self):
        """Create fitted detector."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 200)
        
        detector = HMMRegimeDetector(n_regimes=3, n_iter=10)
        detector.fit(returns)
        return detector
    
    @pytest.fixture
    def generator(self, detector):
        """Create signal generator."""
        return RegimeAwareSignalGenerator(regime_detector=detector)
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample signal."""
        return {
            'confidence': 0.7,
            'position_size': 100.0,
            'action': 'BUY'
        }
    
    def test_generator_initialization(self, detector):
        """Test generator initialization."""
        generator = RegimeAwareSignalGenerator(regime_detector=detector)
        
        assert generator.regime_detector is detector
        assert generator.bull_multiplier == 1.2
        assert generator.bear_multiplier == 0.5
    
    def test_adjust_signal_bull(self, generator, sample_signal):
        """Test signal adjustment in bull market."""
        # Create bull regime result
        regime_state = RegimeState(
            regime=1,
            regime_name="Bull/Low Vol",
            probability=0.9,  # High probability
            volatility=0.01,
            expected_return=0.001,
            duration=10
        )
        regime_result = RegimeResult(
            current_regime=regime_state,
            regime_history=[1] * 10,
            probabilities=np.array([[0.1, 0.9, 0.0]]),
            transition_matrix=np.eye(3),
            means=np.array([[-0.001], [0.001], [0.0]]),
            covariances=np.array([[[0.01]], [[0.01]], [[0.01]]])
        )
        
        adjusted = generator.adjust_signal(sample_signal, regime_result)
        
        # Bull market should have regime info
        assert adjusted['regime'] == "Bull/Low Vol"
        assert adjusted['regime_probability'] == 0.9
    
    def test_adjust_signal_bear(self, generator, sample_signal):
        """Test signal adjustment in bear market."""
        # Create bear regime result
        regime_state = RegimeState(
            regime=2,
            regime_name="Bear/High Vol",
            probability=0.7,
            volatility=0.03,
            expected_return=-0.002,
            duration=5
        )
        regime_result = RegimeResult(
            current_regime=regime_state,
            regime_history=[2] * 10,
            probabilities=np.array([[0.1, 0.2, 0.7]]),
            transition_matrix=np.eye(3),
            means=np.array([[-0.002], [0.001], [0.0]]),
            covariances=np.array([[[0.03]], [[0.01]], [[0.01]]])
        )
        
        adjusted = generator.adjust_signal(sample_signal, regime_result)
        
        # Bear market should decrease confidence
        assert adjusted['confidence'] <= sample_signal['confidence']
        assert adjusted['regime'] == "Bear/High Vol"
    
    def test_should_trade_bull(self, generator):
        """Test trading decision in bull market."""
        regime_state = RegimeState(
            regime=1,
            regime_name="Bull/Low Vol",
            probability=0.8,
            volatility=0.01,
            expected_return=0.001,
            duration=10
        )
        regime_result = RegimeResult(
            current_regime=regime_state,
            regime_history=[1] * 10,
            probabilities=np.array([[0.1, 0.8, 0.1]]),
            transition_matrix=np.eye(3),
            means=np.array([[0.0], [0.001], [-0.001]]),
            covariances=np.array([[[0.01]], [[0.01]], [[0.02]]])
        )
        
        should_trade, reason = generator.should_trade(regime_result)
        
        assert should_trade is True
        assert "Bull" in reason
    
    def test_should_trade_bear_high_prob(self, generator):
        """Test trading decision in high probability bear market."""
        regime_state = RegimeState(
            regime=2,
            regime_name="Bear/High Vol",
            probability=0.8,
            volatility=0.03,
            expected_return=-0.002,
            duration=5
        )
        regime_result = RegimeResult(
            current_regime=regime_state,
            regime_history=[2] * 10,
            probabilities=np.array([[0.1, 0.1, 0.8]]),
            transition_matrix=np.eye(3),
            means=np.array([[0.0], [0.001], [-0.002]]),
            covariances=np.array([[[0.01]], [[0.01]], [[0.03]]])
        )
        
        should_trade, reason = generator.should_trade(regime_result)
        
        assert should_trade is False
        assert "bear" in reason.lower()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_returns(self):
        """Test returns computation."""
        prices = np.array([100, 101, 102, 101, 103])
        returns = compute_returns(prices)
        
        assert len(returns) == len(prices) - 1
        assert np.isclose(returns[0], np.log(101/100))
    
    def test_compute_volatility(self):
        """Test volatility computation."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        volatility = compute_volatility(returns, window=20)
        
        assert len(volatility) == len(returns)
        assert volatility[0] > 0  # Should have a value
    
    def test_get_regime_score_bull(self):
        """Test regime score for bull market."""
        regime_state = RegimeState(
            regime=1,
            regime_name="Bull/Low Vol",
            probability=0.8,
            volatility=0.01,
            expected_return=0.001,
            duration=10
        )
        regime_result = RegimeResult(
            current_regime=regime_state,
            regime_history=[1] * 10,
            probabilities=np.array([[0.1, 0.8, 0.1]]),
            transition_matrix=np.eye(3),
            means=np.array([[0.0], [0.001], [-0.001]]),
            covariances=np.array([[[0.01]], [[0.01]], [[0.02]]])
        )
        
        score = get_regime_score(regime_result)
        
        assert 0 < score <= 1  # Positive for bull
    
    def test_get_regime_score_bear(self):
        """Test regime score for bear market."""
        regime_state = RegimeState(
            regime=2,
            regime_name="Bear/High Vol",
            probability=0.7,
            volatility=0.03,
            expected_return=-0.002,
            duration=5
        )
        regime_result = RegimeResult(
            current_regime=regime_state,
            regime_history=[2] * 10,
            probabilities=np.array([[0.1, 0.2, 0.7]]),
            transition_matrix=np.eye(3),
            means=np.array([[0.0], [0.001], [-0.002]]),
            covariances=np.array([[[0.01]], [[0.01]], [[0.03]]])
        )
        
        score = get_regime_score(regime_result)
        
        assert -1 <= score < 0  # Negative for bear
    
    def test_get_regime_score_sideways(self):
        """Test regime score for sideways market."""
        regime_state = RegimeState(
            regime=0,
            regime_name="Sideways/Low Vol",
            probability=0.6,
            volatility=0.01,
            expected_return=0.0,
            duration=15
        )
        regime_result = RegimeResult(
            current_regime=regime_state,
            regime_history=[0] * 10,
            probabilities=np.array([[0.6, 0.2, 0.2]]),
            transition_matrix=np.eye(3),
            means=np.array([[0.0], [0.001], [-0.001]]),
            covariances=np.array([[[0.01]], [[0.01]], [[0.02]]])
        )
        
        score = get_regime_score(regime_result)
        
        assert score == 0.0  # Zero for sideways


class TestCreateRegimeDetector:
    """Test factory function."""
    
    def test_create_detector_basic(self):
        """Test creating detector with prices."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 200)))
        
        detector = create_regime_detector(prices, n_regimes=3)
        
        assert detector.is_fitted
        assert detector.n_regimes == 3
    
    def test_create_detector_with_volatility(self):
        """Test creating detector with volatility."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 200)))
        
        detector = create_regime_detector(prices, n_regimes=3, use_volatility=True)
        
        assert detector.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
