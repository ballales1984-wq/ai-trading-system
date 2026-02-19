"""
Hedge Fund Research-Grade ML Module
====================================
Advanced ML utilities for systematic trading:

1. Regime Detection Module - Hidden Markov Model (HMM) for market regime identification
2. Meta-Labeling - Using risk engine signals as labels for ML
3. Walk-Forward Validation - Monte Carlo cross-validation with rolling windows
4. Advanced Feature Engineering - Rolling beta, cross-asset correlations, microstructure

Author: AI Trading System
Version: 3.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings

# ML and Stats
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
from scipy.stats import normaltest, jarque_bera

# HMM - Hidden Markov Model
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not available. Regime detection will use fallback method.")

# For Monte Carlo
try:
    from sklearn.utils import resample
except ImportError:
    from sklearn.utils import bootstrap

import joblib
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# REGIME DETECTION - Hidden Markov Model
# =============================================================================

class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CONSOLIDATION = "consolidation"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current state of market regime"""
    regime: MarketRegime
    probability: float
    regime_id: int
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if isinstance(self.regime, str):
            self.regime = MarketRegime(self.regime)


@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""
    current_regime: RegimeState
    regime_probabilities: Dict[MarketRegime, float]
    regime_history: List[RegimeState]
    transition_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


class HiddenMarkovRegimeDetector(BaseEstimator, ClassifierMixin):
    """
    Hidden Markov Model for Market Regime Detection
    
    Identifies market regimes:
    - Trending (up/down)
    - Mean-reverting
    - High volatility
    - Low volatility
    - Consolidation
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        n_features: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        regime_names: Optional[List[str]] = None
    ):
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Default regime names
        self.regime_names = regime_names or [
            "low_vol", "trending", "high_vol", "mean_revert"
        ][:n_regimes]
        
        self.model_: Optional[GaussianHMM] = None
        self.scaler_ = RobustScaler()
        self.label_encoder_ = LabelEncoder()
        self.is_fitted_ = False
        
        # Regime history
        self.regime_history: List[RegimeState] = []
        self.transition_matrix_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'HiddenMarkovRegimeDetector':
        """Fit the HMM model"""
        
        # Scale features
        X_scaled = self.scaler_.fit_transform(X)
        
        if HMM_AVAILABLE and X_scaled.shape[0] > self.n_regimes:
            try:
                self.model_ = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state
                )
                self.model_.fit(X_scaled)
                self.is_fitted_ = True
                
                # Extract transition matrix
                if hasattr(self.model_, 'transmat_'):
                    self.transition_matrix_ = self.model_.transmat_
                    
            except Exception as e:
                logger.warning(f"HMM fitting failed, using fallback: {e}")
                self._fit_fallback(X_scaled)
        else:
            self._fit_fallback(X_scaled)
            
        return self
    
    def _fit_fallback(self, X: np.ndarray):
        """Fallback regime detection using clustering"""
        from sklearn.cluster import KMeans
        
        # Use K-Means as fallback
        kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(X)
        
        # Store cluster centers and scaler info for prediction
        self.cluster_centers_ = kmeans.cluster_centers_
        self.is_fitted_ = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime labels"""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
            
        X_scaled = self.scaler_.transform(X)
        
        if HMM_AVAILABLE and self.model_ is not None:
            try:
                return self.model_.predict(X_scaled)
            except:
                pass
                
        # Fallback: use nearest cluster
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state)
        kmeans.fit(self.cluster_centers_)
        return kmeans.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities"""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
            
        X_scaled = self.scaler_.transform(X)
        
        if HMM_AVAILABLE and self.model_ is not None:
            try:
                return self.model_.predict_proba(X_scaled)
            except:
                pass
                
        # Fallback: distance-based probabilities
        distances = np.zeros((X_scaled.shape[0], self.n_regimes))
        for i in range(self.n_regimes):
            distances[:, i] = np.linalg.norm(
                X_scaled - self.cluster_centers_[i], axis=1
            )
        
        # Convert distances to probabilities (inverse distance)
        inv_distances = 1 / (distances + 1e-10)
        return inv_distances / inv_distances.sum(axis=1, keepdims=True)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score samples (log likelihood)"""
        X_scaled = self.scaler_.transform(X)
        
        if HMM_AVAILABLE and self.model_ is not None:
            try:
                return self.model_.score_samples(X_scaled)
            except:
                pass
                
        # Fallback
        return -np.linalg.norm(X_scaled - self.cluster_centers_.mean(axis=0), axis=1)
    
    def get_current_regime(self, X: np.ndarray) -> RegimeDetectionResult:
        """Get current regime with probabilities"""
        X_scaled = self.scaler_.transform(X)
        
        if len(X_scaled) == 0:
            return RegimeDetectionResult(
                current_regime=RegimeState(
                    regime=MarketRegime.UNKNOWN,
                    probability=0.0,
                    regime_id=-1
                ),
                regime_probabilities={},
                regime_history=[],
                transition_matrix=self.transition_matrix_
            )
        
        # Get last observation
        X_last = X_scaled[-1:]
        
        # Predict probabilities
        proba = self.predict_proba(X_last)[0]
        regime_id = np.argmax(proba)
        probability = proba[regime_id]
        
        # Map to regime names
        regime_name = self.regime_names[regime_id] if regime_id < len(self.regime_names) else "unknown"
        
        # Create regime mapping
        regime_map = {
            0: MarketRegime.LOW_VOLATILITY,
            1: MarketRegime.TRENDING_UP,
            2: MarketRegime.HIGH_VOLATILITY,
            3: MarketRegime.MEAN_REVERTING
        }
        
        current_regime = RegimeState(
            regime=regime_map.get(regime_id, MarketRegime.UNKNOWN),
            probability=probability,
            regime_id=regime_id,
            timestamp=datetime.now()
        )
        
        # Build probability dictionary
        regime_probs = {}
        for i, prob in enumerate(proba):
            if i < len(self.regime_names):
                regime_map_local = {
                    "low_vol": MarketRegime.LOW_VOLATILITY,
                    "trending": MarketRegime.TRENDING_UP,
                    "high_vol": MarketRegime.HIGH_VOLATILITY,
                    "mean_revert": MarketRegime.MEAN_REVERTING,
                    "consolidation": MarketRegime.CONSOLIDATION
                }
                regime_probs[regime_map_local.get(self.regime_names[i], MarketRegime.UNKNOWN)] = prob
        
        return RegimeDetectionResult(
            current_regime=current_regime,
            regime_probabilities=regime_probs,
            regime_history=self.regime_history[-100:],  # Last 100
            transition_matrix=self.transition_matrix_
        )
    
    def detect_regimes(self, X: np.ndarray) -> List[RegimeState]:
        """Detect regimes for entire dataset"""
        regime_ids = self.predict(X)
        probas = self.predict_proba(X)
        
        states = []
        for i, (reg_id, proba) in enumerate(zip(regime_ids, probas)):
            regime_map = {
                0: MarketRegime.LOW_VOLATILITY,
                1: MarketRegime.TRENDING_UP,
                2: MarketRegime.HIGH_VOLATILITY,
                3: MarketRegime.MEAN_REVERTING
            }
            
            states.append(RegimeState(
                regime=regime_map.get(reg_id, MarketRegime.UNKNOWN),
                probability=proba[reg_id],
                regime_id=reg_id
            ))
        
        self.regime_history = states
        return states


class RegimeFeatureGenerator:
    """
    Generate features for regime detection
    """
    
    @staticmethod
    def create_regime_features(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Create features for regime detection
        
        Features:
        - Returns and volatility
        - Trend strength (ADX-like)
        - Mean reversion indicators
        - Volume features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic returns
        returns = df['close'].pct_change()
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        for w in windows:
            features[f'volatility_{w}'] = returns.rolling(w).std()
            features[f'realized_vol_{w}'] = returns.rolling(w).std() * np.sqrt(252)
        
        # Trend features
        for w in windows:
            # Price relative to SMA
            sma = df['close'].rolling(w).mean()
            features[f'price_sma_ratio_{w}'] = df['close'] / sma - 1
            
            # Trend strength (absolute return / volatility)
            trend_return = df['close'].pct_change(w)
            trend_vol = returns.rolling(w).std()
            features[f'trend_strength_{w}'] = np.abs(trend_return) / (trend_vol + 1e-10)
        
        # Mean reversion features
        for w in [10, 20]:
            # Distance from mean
            rolling_mean = df['close'].rolling(w).mean()
            rolling_std = df['close'].rolling(w).std()
            features[f'z_score_{w}'] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)
            
            # Bollinger Band position
            bb_upper = rolling_mean + 2 * rolling_std
            bb_lower = rolling_mean - 2 * rolling_std
            features[f'bb_position_{w}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # Volume features
        if 'volume' in df.columns:
            for w in [10, 20]:
                features[f'volume_ratio_{w}'] = df['volume'] / df['volume'].rolling(w).mean() - 1
        
        # Momentum
        for w in [5, 10, 20]:
            features[f'momentum_{w}'] = df['close'].pct_change(w)
            features[f'momentum_accel_{w}'] = features[f'momentum_{w}'].diff()
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features


# =============================================================================
# META-LABELING - Using Risk Engine as Labels
# =============================================================================

class MetaLabelType(Enum):
    """Types of meta-labels"""
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    SIGNAL_QUALITY = "signal_quality"
    TRADE_WORTHINESS = "trade_worthiness"
    REGIME_ADAPTIVE = "regime_adaptive"
    MULTI_FACTOR = "multi_factor"


@dataclass
class MetaLabelConfig:
    """Configuration for meta-labeling"""
    label_type: MetaLabelType = MetaLabelType.TRADE_WORTHINESS
    
    # Risk-based thresholds
    min_risk_reward: float = 1.5
    max_drawdown_threshold: float = 0.05
    min_win_rate: float = 0.45
    
    # Signal quality thresholds
    min_signal_strength: float = 0.6
    min_confidence: float = 0.5
    
    # Position sizing
    use_kelly_criterion: bool = True
    max_kelly_fraction: float = 0.25
    
    # Regime considerations
    use_regime_filtering: bool = True
    regime_confidence_threshold: float = 0.7


class MetaLabelGenerator:
    """
    Generate meta-labels using risk engine signals
    
    Instead of simple up/down labels, learns:
    - Is this trade worth taking?
    - Does the risk/reward justify the position?
    - Should we size up or down?
    """
    
    def __init__(self, config: Optional[MetaLabelConfig] = None):
        self.config = config or MetaLabelConfig()
        
    def create_risk_based_labels(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        signals: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Create labels based on risk-adjusted returns
        
        Label = 1 if:
        - Return > threshold AND risk metrics acceptable
        - Sharpe-like ratio positive
        
        Label = 0 otherwise
        """
        labels = np.zeros(len(predictions))
        
        # Calculate risk-adjusted returns
        rolling_vol = pd.Series(actual_returns).rolling(20).std()
        risk_adj_returns = actual_returns / (rolling_vol + 1e-10)
        
        # Also consider the prediction confidence
        for i in range(len(predictions)):
            ret = actual_returns[i]
            adj_ret = risk_adj_returns[i]
            
            # Check multiple conditions
            if self.config.label_type == MetaLabelType.RISK_ADJUSTED_RETURN:
                # Label based on risk-adjusted return
                labels[i] = 1 if adj_ret > 0 else 0
                
            elif self.config.label_type == MetaLabelType.TRADE_WORTHINESS:
                # More sophisticated: is trade worth taking?
                # Good if: positive return OR (low vol AND positive prediction)
                is_profitable = ret > 0
                is_confident_prediction = predictions[i] > self.config.min_signal_strength
                is_low_vol = rolling_vol.iloc[i] < rolling_vol.median() if i >= 20 else True
                
                labels[i] = 1 if (is_profitable or (is_confident_prediction and is_low_vol)) else 0
                
            elif self.config.label_type == MetaLabelType.SIGNAL_QUALITY:
                # Label based on prediction quality
                correct_direction = (predictions[i] > 0.5) == (ret > 0)
                labels[i] = 1 if correct_direction else 0
                
            elif self.config.label_type == MetaLabelType.REGIME_ADAPTIVE:
                # Consider regime - in trending regimes, follow trend
                # In mean-reverting, be more selective
                regime = signals.get('regime', [MarketRegime.UNKNOWN] * len(predictions))[i] if signals is not None else MarketRegime.UNKNOWN
                
                if regime == MarketRegime.TRENDING_UP:
                    labels[i] = 1 if ret > 0 else 0
                elif regime == MarketRegime.TRENDING_DOWN:
                    labels[i] = 1 if ret < 0 else 0
                else:
                    # Mean reversion - require stronger signal
                    labels[i] = 1 if abs(ret) > actual_returns.std() else 0
            else:
                # Default: binary
                labels[i] = 1 if ret > 0 else 0
                
        return labels
    
    def create_ternary_labels(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        actual_returns: np.ndarray
    ) -> np.ndarray:
        """
        Create ternary labels: -1 (sell), 0 (no trade), 1 (buy)
        
        This is useful for position sizing - learning WHEN to trade
        """
        labels = np.zeros(len(predictions))
        
        # Strong signals only
        strong_buy_threshold = 0.65
        strong_sell_threshold = 0.35
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(actual_returns).rolling(20).mean()
        rolling_std = pd.Series(actual_returns).rolling(20).std()
        
        for i in range(len(predictions)):
            pred = predictions[i]
            ret = actual_returns[i]
            
            # Strong buy signal with positive return
            if pred > strong_buy_threshold and ret > 0:
                labels[i] = 1
            # Strong sell signal with negative return
            elif pred < strong_sell_threshold and ret < 0:
                labels[i] = -1
            # Otherwise: no trade
            
        return labels
    
    def create_position_size_labels(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        regimes: Optional[List[MarketRegime]] = None
    ) -> np.ndarray:
        """
        Learn optimal position sizes
        
        Returns fraction (0 to 1) representing position size
        """
        # Base size on prediction confidence
        base_size = np.abs(predictions - 0.5) * 2  # 0 to 1
        
        # Adjust for regime
        if regimes is not None and self.config.use_regime_filtering:
            regime_multipliers = {
                MarketRegime.LOW_VOLATILITY: 1.0,
                MarketRegime.TRENDING_UP: 1.2,
                MarketRegime.TRENDING_DOWN: 1.2,
                MarketRegime.HIGH_VOLATILITY: 0.5,
                MarketRegime.MEAN_REVERTING: 0.7,
                MarketRegime.CONSOLIDATION: 0.3,
                MarketRegime.UNKNOWN: 0.5
            }
            
            for i, regime in enumerate(regimes):
                if i < len(base_size):
                    mult = regime_multipliers.get(regime, 0.5)
                    base_size[i] = min(base_size[i] * mult, self.config.max_kelly_fraction)
        
        # Kelly criterion adjustment
        if self.config.use_kelly_criterion:
            # Calculate win rate and avg win/loss
            win_rate = np.mean(actual_returns > 0)
            avg_win = np.mean(actual_returns[actual_returns > 0]) if np.any(actual_returns > 0) else 0
            avg_loss = np.abs(np.mean(actual_returns[actual_returns < 0])) if np.any(actual_returns < 0) else 1
            
            if avg_loss > 0:
                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly = max(0, min(kelly, self.config.max_kelly_fraction))
                
                # Apply Kelly to base size
                base_size = base_size * (kelly / self.config.max_kelly_fraction)
        
        return base_size


# =============================================================================
# WALK-FORWARD VALIDATION - Monte Carlo
# =============================================================================

@dataclass
class WalkForwardResult:
    """Result of walk-forward validation"""
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_size: int
    test_size: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    metric_name: str
    mean: float
    std: float
    median: float
    ci_95_lower: float
    ci_95_upper: float
    min_value: float
    max_value: float
    n_simulations: int
    distribution: List[float] = field(default_factory=list)


class MonteCarloWalkForward:
    """
    Monte Carlo Walk-Forward Validation
    
    Features:
    - Rolling window cross-validation
    - Bootstrap resampling for robustness
    - Multiple train/test splits
    - Statistical significance testing
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size_ratio: float = 0.7,
        n_monte_carlo: int = 100,
        purge_size: int = 5,
        embargo_size: int = 2,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.train_size_ratio = train_size_ratio
        self.n_monte_carlo = n_monte_carlo
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.random_state = random_state
        
        self.results: List[WalkForwardResult] = []
        self.monte_carlo_results: Dict[str, MonteCarloResult] = {}
        
    def run_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        feature_func: Optional[Callable] = None
    ) -> Dict[str, MonteCarloResult]:
        """
        Run Monte Carlo walk-forward validation
        
        Args:
            X: Feature matrix
            y: Target vector
            model_factory: Function that creates a fresh model
            feature_func: Optional function to transform features
            
        Returns:
            Dictionary of MonteCarloResult for each metric
        """
        np.random.seed(self.random_state)
        
        n_samples = len(y)
        test_size = int(n_samples * (1 - self.train_size_ratio) / self.n_splits)
        
        # Store all simulation results
        all_metrics: Dict[str, List[float]] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Run walk-forward splits
        for split in range(self.n_splits):
            # Calculate indices with purge
            train_end = int((split + self.train_size_ratio) * n_samples / self.n_splits) + split * test_size
            test_start = train_end + self.purge_size
            test_end = min(test_start + test_size, n_samples)
            
            if test_end - test_start < 10 or train_end < 50:
                continue
            
            # Split data
            X_train, X_test = X[:train_end], X[test_start:test_end]
            y_train, y_test = y[:train_end], y[test_start:test_end]
            
            # Apply feature function if provided
            if feature_func is not None:
                X_train = feature_func(X_train)
                X_test = feature_func(X_test)
            
            # Monte Carlo within this split
            for mc in range(self.n_monte_carlo):
                # Bootstrap resample with purge
                if len(X_train) > 50:
                    indices = np.random.choice(
                        len(X_train),
                        size=len(X_train),
                        replace=True
                    )
                    # Embargo: remove recent samples
                    indices = indices[indices < len(X_train) - self.embargo_size]
                    
                    if len(indices) < 20:
                        continue
                        
                    X_train_mc = X_train[indices]
                    y_train_mc = y_train[indices]
                else:
                    X_train_mc = X_train
                    y_train_mc = y_train
                
                try:
                    # Create and train model
                    model = model_factory()
                    model.fit(X_train_mc, y_train_mc)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if model.predict_proba(X_test).shape[1] > 1 else model.predict_proba(X_test).flatten()
                    
                    # Calculate metrics
                    all_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                    all_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                    all_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                    all_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
                    
                    if len(np.unique(y_test)) > 1:
                        all_metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
                        
                except Exception as e:
                    logger.warning(f"MC iteration failed: {e}")
                    continue
        
        # Compute Monte Carlo statistics
        for metric, values in all_metrics.items():
            if values:
                values_array = np.array(values)
                self.monte_carlo_results[metric] = MonteCarloResult(
                    metric_name=metric,
                    mean=np.mean(values_array),
                    std=np.std(values_array),
                    median=np.median(values_array),
                    ci_95_lower=np.percentile(values_array, 2.5),
                    ci_95_upper=np.percentile(values_array, 97.5),
                    min_value=np.min(values_array),
                    max_value=np.max(values_array),
                    n_simulations=len(values_array),
                    distribution=values
                )
        
        return self.monte_carlo_results
    
    def get_stability_score(self) -> float:
        """
        Calculate model stability score (0-1)
        
        Based on coefficient of variation of metrics
        """
        if not self.monte_carlo_results:
            return 0.0
            
        cv_scores = []
        for metric, result in self.monte_carlo_results.items():
            if result.mean != 0:
                cv = result.std / abs(result.mean)
                # Convert CV to stability score (lower CV = higher stability)
                stability = 1 / (1 + cv)
                cv_scores.append(stability)
        
        return np.mean(cv_scores) if cv_scores else 0.0
    
    def is_robust(self, min_stability: float = 0.6) -> bool:
        """Check if model is robust based on Monte Carlo results"""
        return self.get_stability_score() >= min_stability


class PurgedWalkForward:
    """
    Purged Walk-Forward with Embargo
    
    Prevents look-ahead bias by:
    1. Purging: Removing samples between train and test
    2. Embargo: Additional buffer for bootstrap
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size_ratio: float = 0.7,
        purge_pct: float = 0.1,
        embargo_pct: float = 0.05
    ):
        self.n_splits = n_splits
        self.train_size_ratio = train_size_ratio
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate purged splits"""
        n = len(y)
        
        # Calculate window sizes
        window_size = n // self.n_splits
        train_size = int(window_size * self.train_size_ratio)
        test_size = window_size - train_size
        
        # Purge size
        purge_size = int(window_size * self.purge_pct)
        embargo_size = int(train_size * self.embargo_pct)
        
        splits = []
        
        for i in range(self.n_splits - 1):
            # Train window
            train_start = i * test_size
            train_end = train_start + train_size
            
            # Apply embargo to train (remove most recent samples)
            train_end_embargoed = train_end - embargo_size
            
            # Test window with purge
            test_start = train_end + purge_size
            test_end = min(test_start + test_size, n)
            
            if train_end_embargoed - train_start < 50 or test_end - test_start < 10:
                continue
            
            X_train = X[train_start:train_end_embargoed]
            y_train = y[train_start:train_end_embargoed]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            splits.append((X_train, X_test, y_train, y_test))
        
        return splits


# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering for systematic trading
    
    Features:
    - Rolling beta to BTC/ETH
    - Cross-asset correlations
    - Market microstructure
    - Order flow indicators
    """
    
    def __init__(self, lookback_windows: List[int] = None):
        self.lookback_windows = lookback_windows or [5, 10, 20, 50]
        self.scalers: Dict[str, RobustScaler] = {}
        
    def add_rolling_beta_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        benchmark_col: str = 'close',
        benchmark_name: str = 'BTC'
    ) -> pd.DataFrame:
        """
        Calculate rolling beta to benchmark (BTC/ETH)
        
        Beta = Cov(asset, benchmark) / Var(benchmark)
        """
        result = df.copy()
        
        # Returns
        asset_returns = df[target_col].pct_change()
        benchmark_returns = df[benchmark_col].pct_change() if benchmark_col != target_col else asset_returns
        
        for window in self.lookback_windows:
            # Rolling covariance
            rolling_cov = asset_returns.rolling(window).cov(benchmark_returns)
            
            # Rolling variance of benchmark
            rolling_var = benchmark_returns.rolling(window).var()
            
            # Beta
            result[f'beta_{benchmark_name}_{window}'] = rolling_cov / (rolling_var + 1e-10)
            
            # Rolling correlation (for context)
            result[f'corr_{benchmark_name}_{window}'] = asset_returns.rolling(window).corr(benchmark_returns)
        
        return result
    
    def add_cross_asset_correlations(
        self,
        df: pd.DataFrame,
        other_assets: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Add cross-asset correlation features
        """
        result = df.copy()
        
        # Main asset returns
        main_returns = df['close'].pct_change()
        
        for asset_name, asset_prices in other_assets.items():
            # Ensure same index
            if isinstance(asset_prices, pd.Series):
                asset_returns = asset_prices.pct_change()
                
                for window in self.lookback_windows[:3]:  # Limit for performance
                    corr = main_returns.rolling(window).corr(asset_returns)
                    result[f'corr_{asset_name}_{window}'] = corr
                    
                    # Lead-lag relationship
                    lead_corr = main_returns.rolling(window).corr(asset_returns.shift(1))
                    result[f'lead_corr_{asset_name}_{window}'] = lead_corr
        
        return result
    
    def add_microstructure_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Market microstructure features
        
        Features:
        - Amihud illiquidity ratio
        - Volume-price relationship
        - Order flow imbalance
        - Spread estimates
        """
        result = df.copy()
        
        # Returns
        returns = df['close'].pct_change()
        abs_returns = np.abs(returns)
        
        # Amihud Illiquidity Ratio
        if 'volume' in df.columns:
            # |return| / volume
            result['amihud_illiquidity'] = abs_returns / (df['volume'] + 1)
            result['amihud_illiquidity_20'] = result['amihud_illiquidity'].rolling(20).mean()
        
        # Volume-weighted returns
        if 'volume' in df.columns:
            result['vw_ret_5'] = (returns * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
            result['vw_ret_20'] = (returns * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Price impact
        if 'volume' in df.columns:
            # How much does price move per unit volume?
            result['price_impact_5'] = abs_returns.rolling(5).sum() / (df['volume'].rolling(5).sum() + 1)
            result['price_impact_20'] = abs_returns.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1)
        
        # Order flow (if we have bid/ask)
        if 'high' in df.columns and 'low' in df.columns:
            # Spread proxy
            result['spread_proxy'] = (df['high'] - df['low']) / df['close']
            result['spread_proxy_20'] = result['spread_proxy'].rolling(20).mean()
            
            # Close position in range
            result['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            
            # Intraday range
            result['intraday_range'] = (df['high'] - df['low']) / df['close']
            result['intraday_range_20'] = result['intraday_range'].rolling(20).mean()
        
        # Return dispersion
        for window in [5, 10, 20]:
            result[f'return_dispersion_{window}'] = returns.rolling(window).std()
        
        # Kurtosis and skewness
        for window in [20, 50]:
            result[f'return_skew_{window}'] = returns.rolling(window).skew()
            result[f'return_kurt_{window}'] = returns.rolling(window).apply(lambda x: stats.kurtosis(x) if len(x) > 3 else 0)
        
        return result
    
    def add_order_flow_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Order flow indicators
        
        Based on:
        - Volume analysis
        - Price action
        - Accumulation/Distribution
        """
        result = df.copy()
        
        if 'volume' not in df.columns:
            return result
        
        # Accumulation/Distribution Line
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        money_flow_volume = money_flow_multiplier * df['volume']
        result['ad_line'] = money_flow_volume.cumsum()
        result['ad_oscillator'] = result['ad_line'] - result['ad_line'].rolling(20).mean()
        
        # Chaikin Money Flow
        result['cmf_20'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volume rate of change
        result['volume_roc_5'] = df['volume'].pct_change(5)
        result['volume_roc_20'] = df['volume'].pct_change(20)
        
        # On-Balance Volume changes
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        result['obv_roc_5'] = obv.pct_change(5)
        result['obv_roc_20'] = obv.pct_change(20)
        
        # Volume momentum
        result['volume_momentum_5'] = df['volume'] / df['volume'].rolling(5).mean() - 1
        result['volume_momentum_20'] = df['volume'] / df['volume'].rolling(20).mean() - 1
        
        # Price-Volume correlation
        result['pv_corr_10'] = df['close'].pct_change().rolling(10).corr(df['volume'])
        result['pv_corr_20'] = df['close'].pct_change().rolling(20).corr(df['volume'])
        
        return result
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Add sentiment-derived features
        """
        result = df.copy()
        
        if sentiment_data is None:
            # Generate synthetic sentiment from price action
            # This is a placeholder - in production, use actual sentiment data
            returns = df['close'].pct_change()
            
            # Aggregate sentiment (synthetic)
            result['sentiment_synthetic'] = (
                (returns > 0).rolling(10).sum() - 
                (returns < 0).rolling(10).sum()
            ) / 10
            
            result['sentiment_momentum'] = result['sentiment_synthetic'].diff(5)
        else:
            # Use actual sentiment data
            if 'sentiment' in sentiment_data.columns:
                result['sentiment'] = sentiment_data['sentiment']
                result['sentiment_ma'] = result['sentiment'].rolling(10).mean()
                result['sentiment_std'] = result['sentiment'].rolling(10).std()
        
        return result
    
    def add_regime_features(
        self,
        df: pd.DataFrame,
        regime_detector: HiddenMarkovRegimeDetector
    ) -> pd.DataFrame:
        """
        Add regime-based features
        """
        result = df.copy()
        
        # Create regime features
        regime_features = RegimeFeatureGenerator.create_regime_features(df)
        
        # Detect regimes
        X_regime = regime_features.values
        regime_states = regime_detector.detect_regimes(X_regime)
        regime_probas = regime_detector.predict_proba(X_regime)
        
        # Add regime probabilities as features
        for i in range(min(regime_probas.shape[1], 4)):
            result[f'regime_prob_{i}'] = regime_probas[:, i]
        
        # Add regime indicators
        regime_ids = regime_detector.predict(X_regime)
        for i, regime_name in enumerate(regime_detector.regime_names[:4]):
            result[f'regime_{regime_name}'] = (regime_ids == i).astype(int)
        
        return result
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        other_assets: Optional[Dict[str, pd.Series]] = None,
        regime_detector: Optional[HiddenMarkovRegimeDetector] = None,
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create all advanced features
        """
        result = df.copy()
        
        # Rolling beta to benchmark
        if benchmark_data is not None:
            result = self.add_rolling_beta_features(
                result,
                benchmark_col=benchmark_data['close'].values[-len(result):],
                benchmark_name='BENCH'
            )
        
        # Cross-asset correlations
        if other_assets:
            result = self.add_cross_asset_correlations(result, other_assets)
        
        # Microstructure
        result = self.add_microstructure_features(result)
        
        # Order flow
        result = self.add_order_flow_indicators(result)
        
        # Sentiment
        result = self.add_sentiment_features(result, sentiment_data)
        
        # Regime features
        if regime_detector is not None and len(result) > 50:
            try:
                result = self.add_regime_features(result, regime_detector)
            except Exception as e:
                logger.warning(f"Regime features failed: {e}")
        
        # Clean up
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(0)
        
        return result


# =============================================================================
# HEDGE FUND ML PIPELINE
# =============================================================================

class HedgeFundMLPipeline:
    """
    Complete hedge fund research-grade ML pipeline
    
    Combines:
    - Regime detection
    - Meta-labeling
    - Advanced features
    - Monte Carlo walk-forward validation
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        meta_label_type: MetaLabelType = MetaLabelType.TRADE_WORTHINESS,
        use_monte_carlo: bool = True,
        n_monte_carlo: int = 100
    ):
        # Regime detection
        self.regime_detector = HiddenMarkovRegimeDetector(n_regimes=n_regimes)
        self.regime_features = RegimeFeatureGenerator()
        
        # Meta-labeling
        self.meta_config = MetaLabelConfig(label_type=meta_label_type)
        self.meta_generator = MetaLabelGenerator(self.meta_config)
        
        # Advanced features
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Validation
        self.mc_validator = MonteCarloWalkForward(n_monte_carlo=n_monte_carlo) if use_monte_carlo else None
        
        # Models
        self.model = None
        self.is_fitted = False
        
        # State
        self.current_regime: Optional[RegimeState] = None
        self.feature_names: List[str] = []
        
    def prepare_features(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        other_assets: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """Prepare all features"""
        
        # First, fit regime detector
        regime_feats = self.regime_features.create_regime_features(df)
        
        # Train regime detector
        if len(df) > 100 and regime_feats.shape[1] > 0:
            try:
                X_regime = regime_feats.dropna()
                if len(X_regime) > 50:
                    self.regime_detector.fit(X_regime.values)
                    self.current_regime = self.regime_detector.get_current_regime(X_regime.values[-1:]).current_regime
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")
        
        # Create advanced features
        features = self.feature_engineer.create_all_features(
            df,
            benchmark_data=benchmark_df,
            other_assets=other_assets,
            regime_detector=self.regime_detector if self.regime_detector.is_fitted_ else None
        )
        
        # Store feature names
        self.feature_names = [c for c in features.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        
        return features
    
    def create_labels(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        horizon: int = 1
    ) -> np.ndarray:
        """Create meta-labels"""
        
        # Calculate actual returns
        actual_returns = df['close'].pct_change(horizon).shift(-horizon).values
        
        # Get regimes for labels
        regimes = None
        if self.regime_detector.is_fitted_:
            try:
                regime_feats = self.regime_features.create_regime_features(df)
                regime_states = self.regime_detector.detect_regimes(regime_feats.dropna().values)
                regimes = [s.regime for s in regime_states[:len(predictions)]]
            except:
                pass
        
        # Create labels
        labels = self.meta_generator.create_risk_based_labels(
            df, predictions, actual_returns, regimes=regimes
        )
        
        return labels
    
    def train(
        self,
        df: pd.DataFrame,
        initial_predictions: np.ndarray,
        benchmark_df: Optional[pd.DataFrame] = None,
        other_assets: Optional[Dict[str, pd.Series]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the full pipeline
        
        Args:
            df: Price data
            initial_predictions: Initial ML predictions (from base model)
            benchmark_df: Benchmark data (e.g., BTC)
            other_assets: Other asset price series
            test_size: Test set proportion
        """
        
        # Prepare features
        X = self.prepare_features(df, benchmark_df, other_assets)
        
        # Create meta-labels
        y = self.create_labels(df, initial_predictions)
        
        # Align
        valid = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid]
        y = y[valid]
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx].values
        X_test = X.iloc[split_idx:].values
        y_train = y.iloc[:split_idx].values
        y_test = y.iloc[split_idx:].values
        
        # Train meta-model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if len(self.model.classes_) > 1 else self.model.predict_proba(X_test).flatten()
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
        }
        
        if len(np.unique(y_test)) > 1:
            results['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        # Run Monte Carlo validation if enabled
        if self.mc_validator is not None:
            mc_results = self.mc_validator.run_validation(
                X_train, y_train,
                lambda: GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=42
                )
            )
            results['monte_carlo'] = {
                k: {'mean': v.mean, 'std': v.std, 'ci_95': (v.ci_95_lower, v.ci_95_upper)}
                for k, v in mc_results.items()
            }
            results['stability_score'] = self.mc_validator.get_stability_score()
            results['is_robust'] = self.mc_validator.is_robust()
        
        return results
    
    def predict(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        other_assets: Optional[Dict[str, pd.Series]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading decisions
        
        Returns:
            (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Prepare features
        X = self.prepare_features(df, benchmark_df, other_assets)
        
        # Align with model
        X = X[self.feature_names] if self.feature_names else X
        
        # Predict
        predictions = self.model.predict(X.values)
        probabilities = self.model.predict_proba(X.values)[:, 1] if len(self.model.classes_) > 1 else self.model.predict_proba(X.values).flatten()
        
        return predictions, probabilities
    
    def get_regime(self) -> Optional[RegimeState]:
        """Get current market regime"""
        return self.current_regime
    
    def get_regime_adaptive_threshold(
        self,
        base_threshold: float = 0.5
    ) -> float:
        """
        Get regime-adaptive threshold
        
        Adjusts trading threshold based on regime:
        - High volatility: more conservative (higher threshold)
        - Trending: follow signals more closely
        - Mean-reverting: require stronger signals
        """
        if self.current_regime is None:
            return base_threshold
        
        regime_adjustments = {
            MarketRegime.HIGH_VOLATILITY: 0.1,
            MarketRegime.LOW_VOLATILITY: -0.05,
            MarketRegime.TRENDING_UP: -0.1,
            MarketRegime.TRENDING_DOWN: -0.1,
            MarketRegime.MEAN_REVERTING: 0.05,
            MarketRegime.CONSOLIDATION: 0.1,
            MarketRegime.UNKNOWN: 0.0
        }
        
        adjustment = regime_adjustments.get(self.current_regime.regime, 0.0)
        
        # Adjust based on confidence
        confidence_adjustment = (self.current_regime.probability - 0.5) * 0.1
        
        return base_threshold + adjustment + confidence_adjustment
    
    def save(self, filepath: str):
        """Save pipeline"""
        data = {
            'regime_detector': self.regime_detector,
            'model': self.model,
            'feature_names': self.feature_names,
            'meta_config': self.meta_config,
            'is_fitted': self.is_fitted
        }
        joblib.dump(data, filepath)
    
    def load(self, filepath: str):
        """Load pipeline"""
        data = joblib.load(filepath)
        self.regime_detector = data['regime_detector']
        self.model = data['model']
        self.feature_names = data.get('feature_names', [])
        self.meta_config = data.get('meta_config', MetaLabelConfig())
        self.is_fitted = data.get('is_fitted', False)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
        
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity: np.ndarray) -> Tuple[float, int, int]:
    """Calculate maximum drawdown"""
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    
    max_dd = np.min(drawdown)
    end_idx = np.argmin(drawdown)
    start_idx = np.argmax(equity[:end_idx+1])
    
    return max_dd, start_idx, end_idx


def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)"""
    annual_return = np.mean(returns) * periods_per_year
    
    equity = (1 + returns).cumprod()
    max_dd, _, _ = calculate_max_drawdown(equity)
    
    if max_dd == 0:
        return 0.0
        
    return annual_return / abs(max_dd)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of how to use the hedge fund ML pipeline"""
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n) * 2),
        'high': 0,
        'low': 0,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    df['high'] = df['close'] * (1 + np.abs(np.random.randn(n) * 0.01))
    df['low'] = df['close'] * (1 - np.abs(np.random.randn(n) * 0.01))
    
    # Benchmark (BTC simulation)
    benchmark_df = pd.DataFrame({
        'close': 20000 + np.cumsum(np.random.randn(n) * 500)
    }, index=dates)
    
    # Initial predictions from base model (simulated)
    initial_predictions = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    
    # Create pipeline
    pipeline = HedgeFundMLPipeline(
        n_regimes=4,
        meta_label_type=MetaLabelType.TRADE_WORTHINESS,
        use_monte_carlo=True,
        n_monte_carlo=50  # Reduced for demo
    )
    
    # Train
    print("Training hedge fund ML pipeline...")
    results = pipeline.train(df, initial_predictions, benchmark_df)
    
    print("\n=== Training Results ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"  {k2}: {v2}")
        else:
            print(f"{k}: {v}")
    
    # Get current regime
    regime = pipeline.get_regime()
    if regime:
        print(f"\nCurrent Regime: {regime.regime.value} (confidence: {regime.probability:.2f})")
    
    # Get adaptive threshold
    adaptive_threshold = pipeline.get_regime_adaptive_threshold()
    print(f"Adaptive Threshold: {adaptive_threshold:.2f}")
    
    print("\nHedge fund ML pipeline ready!")
    
    return pipeline, results


if __name__ == "__main__":
    example_usage()
