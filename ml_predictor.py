"""
Machine Learning Prediction Module - IMPROVED VERSION V2
=========================================================
Advanced ML predictor with better feature engineering and meta-labeling

Key Improvements:
1. Extended features (28 features vs original 7)
2. Risk-adjusted labeling with stop-loss/take-profit
3. Walk-forward cross-validation
4. XGBoost + ensemble for better performance
5. Probability calibration

Author: AI Trading System
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier, 
        AdaBoostClassifier,
        ExtraTreesClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        roc_auc_score,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML predictions disabled.")

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using sklearn alternatives")


class ImprovedPricePredictor:
    """
    Improved Machine Learning-based price movement predictor
    Uses ensemble methods with advanced feature engineering
    """
    
    def __init__(self, use_meta_labeling: bool = True):
        """Initialize the improved predictor"""
        self.scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        self.rf_model = None
        self.gb_model = None
        self.xgb_model = None
        self.et_model = None
        self.lr_model = None
        self.is_trained = False
        self.use_meta_labeling = use_meta_labeling
        
        # Extended feature set (28 features)
        self.feature_names = [
            # Momentum indicators
            'rsi_14', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d',
            # Trend indicators
            'sma_9_ratio', 'sma_21_ratio', 'sma_50_ratio', 'sma_200_ratio',
            'ema_12_ratio', 'ema_26_ratio',
            'adx', 'atr_ratio',
            # Volatility indicators
            'bb_position', 'bb_width',
            'volatility_10', 'volatility_20',
            # Volume indicators
            'volume_ratio', 'volume_ma_ratio', 'obv_change',
            # Price action
            'price_momentum_3', 'price_momentum_5', 'price_momentum_10',
            'high_low_ratio', 'close_open_ratio'
        ]
        
        # Training metrics
        self.training_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'n_samples': 0
        }
        
        # Model weights for ensemble (learned from CV)
        self.model_weights = {
            'rf': 0.25,
            'gb': 0.25,
            'xgb': 0.25,
            'et': 0.25
        }
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare EXTENDED features for ML model
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            DataFrame with 28 features
        """
        if not SKLEARN_AVAILABLE:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']
        
        # === MOMENTUM INDICATORS ===
        
        # RSI (3 periods)
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Stochastic Oscillator
        low14 = low.rolling(window=14).min()
        high14 = high.rolling(window=14).max()
        features['stoch_k'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
        features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
        
        # Moving Average Ratios
        for period in [9, 21, 50, 200]:
            sma = close.rolling(window=period).mean()
            features[f'sma_{period}_ratio'] = close / sma
        
        for period in [12, 26]:
            ema = close.ewm(span=period, adjust=False).mean()
            features[f'ema_{period}_ratio'] = close / ema
        
        # ADX (Average Directional Index)
        high_diff = high.diff()
        low_diff = -low.diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        atr = (high - low).rolling(window=14).mean()
        features['adx'] = abs(plus_dm - minus_dm) / (atr + 1e-10) * 100
        features['atr_ratio'] = atr / close
        
        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        features['bb_width'] = (bb_upper - bb_lower) / sma20
        
        # Volatility
        features['volatility_10'] = close.pct_change().rolling(window=10).std()
        features['volatility_20'] = close.pct_change().rolling(window=20).std()
        
        # Volume features
        features['volume_ratio'] = volume / volume.rolling(window=20).mean()
        features['volume_ma_ratio'] = volume / volume.rolling(window=5).mean()
        features['obv_change'] = (volume * np.sign(close.diff())).rolling(window=10).sum()
        
        # Price momentum
        for period in [3, 5, 10]:
            features[f'price_momentum_{period}'] = close.pct_change(periods=period)
        
        # High/Low ratio
        features['high_low_ratio'] = (high - low) / close
        features['close_open_ratio'] = (close - open_price) / open_price
        
        return features.fillna(0).replace([np.inf, -np.inf], 0)
