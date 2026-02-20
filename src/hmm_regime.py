"""
Hidden Markov Model Regime Detection Module
============================================
Market regime detection using HMM for trading signals.

Features:
- Bull/Bear/Sideways market regime detection
- Volatility regime detection
- Integration with trading signals

Author: AI Trading System
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

# Import hmmlearn with fallback
try:
    from hmmlearn.hmm import GaussianHMM, GMMHMM
    HMM_AVAILABLE = True
    logger.info("hmmlearn available for regime detection")
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")


@dataclass
class RegimeState:
    """Container for regime state information"""
    regime: int  # 0, 1, 2, etc.
    regime_name: str  # Bull, Bear, Sideways
    probability: float
    volatility: float
    expected_return: float
    duration: int  # Expected duration in periods


@dataclass
class RegimeResult:
    """Container for regime detection results"""
    current_regime: RegimeState
    regime_history: List[int]
    probabilities: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


class HMMRegimeDetector:
    """
    Hidden Markov Model for Market Regime Detection.
    
    Detects market regimes (Bull/Bear/Sideways) based on:
    - Returns
    - Volatility
    - Volume (optional)
    
    Usage:
        detector = HMMRegimeDetector(n_regimes=3)
        detector.fit(returns, volatility)
        regime = detector.predict(returns[-1], volatility[-1])
    """
    
    REGIME_NAMES = {
        0: "Sideways/Low Vol",
        1: "Bull/Low Vol",
        2: "Bear/High Vol",
        3: "Bull/High Vol",
        4: "Bear/Low Vol"
    }
    
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize HMM Regime Detector.
        
        Args:
            n_regimes: Number of hidden states (default 3: Bull, Bear, Sideways)
            covariance_type: Type of covariance parameters
            n_iter: Number of EM iterations
            random_state: Random seed for reproducibility
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model: Optional[GaussianHMM] = None
        self.is_fitted = False
        self._regime_mapping: Dict[int, int] = {}
        
        logger.info(f"HMMRegimeDetector initialized with {n_regimes} regimes")
    
    def _prepare_features(
        self,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Prepare feature matrix for HMM."""
        features = [returns.reshape(-1, 1)]
        
        if volatility is not None:
            features.append(volatility.reshape(-1, 1))
        if volume is not None:
            # Normalize volume
            vol_normalized = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
            features.append(vol_normalized.reshape(-1, 1))
        
        return np.hstack(features)
    
    def fit(
        self,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None
    ) -> 'HMMRegimeDetector':
        """
        Fit HMM model to market data.
        
        Args:
            returns: Array of returns (log returns recommended)
            volatility: Array of volatility measures (optional)
            volume: Array of volume data (optional)
            
        Returns:
            self
        """
        X = self._prepare_features(returns, volatility, volume)
        
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X)
        
        self.is_fitted = True
        
        # Map regimes to logical order (by mean return)
        self._map_regimes()
        
        logger.info(f"HMM fitted with {len(returns)} observations")
        return self
    
    def _map_regimes(self):
        """Map regime indices to logical order based on means."""
        if self.model is None:
            return
        
        means = self.model.means_.flatten()
        sorted_indices = np.argsort(means)
        
        # Map: lowest mean = Bear, highest = Bull, middle = Sideways
        self._regime_mapping = {}
        for i, idx in enumerate(sorted_indices):
            if i == 0:
                self._regime_mapping[idx] = 2  # Bear
            elif i == len(sorted_indices) - 1:
                self._regime_mapping[idx] = 1  # Bull
            else:
                self._regime_mapping[idx] = 0  # Sideways
    
    def predict(
        self,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None
    ) -> RegimeResult:
        """
        Predict current market regime.
        
        Args:
            returns: Array of returns
            volatility: Array of volatility measures (optional)
            volume: Array of volume data (optional)
            
        Returns:
            RegimeResult with regime information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._prepare_features(returns, volatility, volume)
        
        # Get hidden states
        hidden_states = self.model.predict(X)
        
        # Get state probabilities
        probabilities = self.model.predict_proba(X)
        
        # Current regime
        current_state = hidden_states[-1]
        current_prob = probabilities[-1]
        
        # Map to logical regime
        mapped_regime = self._regime_mapping.get(current_state, current_state)
        
        regime_state = RegimeState(
            regime=mapped_regime,
            regime_name=self.REGIME_NAMES.get(mapped_regime, f"Regime_{mapped_regime}"),
            probability=float(np.max(current_prob)),
            volatility=float(self.model.covars_[current_state][0, 0] ** 0.5),
            expected_return=float(self.model.means_[current_state][0]),
            duration=int(1 / (1 - self.model.transmat_[current_state, current_state]))
        )
        
        return RegimeResult(
            current_regime=regime_state,
            regime_history=hidden_states.tolist(),
            probabilities=probabilities,
            transition_matrix=self.model.transmat_,
            means=self.model.means_,
            covariances=self.model.covars_
        )
    
    def get_regime_probability(self, regime: int) -> float:
        """Get probability of being in a specific regime."""
        if not self.is_fitted:
            return 0.0
        
        # This would need the latest observation
        return 0.0
    
    def get_expected_duration(self, regime: int) -> int:
        """Get expected duration of a regime in periods."""
        if not self.is_fitted:
            return 0
        
        transition_prob = self.model.transmat_[regime, regime]
        return int(1 / (1 - transition_prob + 1e-8))
    
    def save(self, filepath: str):
        """Save model to file."""
        import joblib
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'n_regimes': self.n_regimes,
            'regime_mapping': self._regime_mapping
        }
        joblib.dump(model_data, filepath)
        logger.info(f"HMM model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        import joblib
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.n_regimes = model_data['n_regimes']
        self._regime_mapping = model_data['regime_mapping']
        self.is_fitted = True
        
        logger.info(f"HMM model loaded from {filepath}")


class RegimeAwareSignalGenerator:
    """
    Generate trading signals adjusted for market regime.
    
    Adjusts signal confidence and position sizing based on:
    - Current market regime
    - Regime probability
    - Regime duration
    """
    
    def __init__(
        self,
        regime_detector: Optional[HMMRegimeDetector] = None,
        bull_multiplier: float = 1.2,
        bear_multiplier: float = 0.5,
        sideways_multiplier: float = 0.8
    ):
        """
        Initialize regime-aware signal generator.
        
        Args:
            regime_detector: HMMRegimeDetector instance
            bull_multiplier: Signal multiplier in bull market
            bear_multiplier: Signal multiplier in bear market
            sideways_multiplier: Signal multiplier in sideways market
        """
        self.regime_detector = regime_detector
        self.bull_multiplier = bull_multiplier
        self.bear_multiplier = bear_multiplier
        self.sideways_multiplier = sideways_multiplier
    
    def adjust_signal(
        self,
        signal: Dict,
        regime_result: Optional[RegimeResult] = None
    ) -> Dict:
        """
        Adjust trading signal based on market regime.
        
        Args:
            signal: Original trading signal dict
            regime_result: Regime detection result (optional, uses detector if not provided)
            
        Returns:
            Adjusted signal dict
        """
        adjusted_signal = signal.copy()
        
        if regime_result is None and self.regime_detector is not None:
            # Would need current data to predict
            return signal
        
        if regime_result is None:
            return signal
        
        regime = regime_result.current_regime
        
        # Get multiplier based on regime
        if regime.regime == 1:  # Bull
            multiplier = self.bull_multiplier
        elif regime.regime == 2:  # Bear
            multiplier = self.bear_multiplier
        else:  # Sideways
            multiplier = self.sideways_multiplier
        
        # Adjust confidence
        adjusted_signal['confidence'] = min(
            signal.get('confidence', 0.5) * multiplier * regime.probability,
            1.0
        )
        
        # Adjust position size
        adjusted_signal['position_size'] = signal.get('position_size', 0.0) * multiplier
        
        # Add regime info
        adjusted_signal['regime'] = regime.regime_name
        adjusted_signal['regime_probability'] = regime.probability
        adjusted_signal['regime_expected_return'] = regime.expected_return
        
        return adjusted_signal
    
    def should_trade(self, regime_result: RegimeResult) -> Tuple[bool, str]:
        """
        Determine if trading is advisable in current regime.
        
        Args:
            regime_result: Regime detection result
            
        Returns:
            Tuple of (should_trade, reason)
        """
        regime = regime_result.current_regime
        
        if regime.regime == 2:  # Bear
            if regime.probability > 0.7:
                return False, "High probability bear market - avoid long positions"
            return True, "Bear market - consider short positions only"
        
        if regime.regime == 1:  # Bull
            return True, "Bull market - favorable for long positions"
        
        # Sideways
        if regime.volatility > 0.03:  # High volatility
            return False, "Sideways with high volatility - wait for direction"
        
        return True, "Sideways market - use range-bound strategies"


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from prices."""
    return np.diff(np.log(prices))


def compute_volatility(
    returns: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """Compute rolling volatility."""
    vol = np.zeros(len(returns))
    for i in range(window, len(returns)):
        vol[i] = np.std(returns[i-window:i])
    
    # Fill initial values
    vol[:window] = vol[window]
    return vol


def create_regime_detector(
    prices: np.ndarray,
    n_regimes: int = 3,
    use_volatility: bool = True
) -> HMMRegimeDetector:
    """
    Factory function to create and fit a regime detector.
    
    Args:
        prices: Array of prices
        n_regimes: Number of regimes
        use_volatility: Whether to include volatility in features
        
    Returns:
        Fitted HMMRegimeDetector
    """
    returns = compute_returns(prices)
    
    if use_volatility:
        volatility = compute_volatility(returns)
        detector = HMMRegimeDetector(n_regimes=n_regimes)
        detector.fit(returns, volatility)
    else:
        detector = HMMRegimeDetector(n_regimes=n_regimes)
        detector.fit(returns)
    
    return detector


# ==================== INTEGRATION WITH DECISION ENGINE ====================

def get_regime_score(regime_result: RegimeResult) -> float:
    """
    Convert regime result to a score for decision engine.
    
    Returns:
        Score from -1 (strong bear) to +1 (strong bull)
    """
    regime = regime_result.current_regime
    
    if regime.regime == 1:  # Bull
        return regime.probability
    elif regime.regime == 2:  # Bear
        return -regime.probability
    else:  # Sideways
        return 0.0


def integrate_with_decision_engine(
    decision_engine,
    regime_detector: HMMRegimeDetector,
    prices: np.ndarray
) -> Dict:
    """
    Integrate regime detection with decision engine.
    
    Args:
        decision_engine: DecisionEngine instance
        regime_detector: HMMRegimeDetector instance
        prices: Recent price data
        
    Returns:
        Dict with regime-adjusted signals
    """
    returns = compute_returns(prices)
    volatility = compute_volatility(returns)
    
    regime_result = regime_detector.predict(returns, volatility)
    regime_score = get_regime_score(regime_result)
    
    return {
        'regime': regime_result.current_regime.regime_name,
        'regime_score': regime_score,
        'regime_probability': regime_result.current_regime.probability,
        'expected_duration': regime_result.current_regime.duration,
        'should_trade': RegimeAwareSignalGenerator().should_trade(regime_result)
    }
