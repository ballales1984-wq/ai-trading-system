"""
ML Predictor V2 - Improved Version
====================================
Advanced features and meta-labeling for better trading predictions

Key Improvements over V1:
1. 28 features (vs 7 original)
2. Meta-labeling with risk management
3. Walk-forward validation
4. XGBoost ensemble
5. Dynamic threshold adjustment
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")


class ImprovedPricePredictor:
    """Improved ML predictor with advanced features"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'min_confidence': 0.55,
            'use_meta_labeling': True
        }
        
        self.scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        self.models = {}
        self.is_trained = False
        self.feature_names = self._get_feature_names()
        self.metrics = {'accuracy': 0.0, 'cv_mean': 0.0, 'cv_std': 0.0, 'n_features': 0}
    
    def _get_feature_names(self) -> List[str]:
        return [
            'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d',
            'sma_9_r', 'sma_21_r', 'sma_50_r', 'sma_200_r',
            'ema_12_r', 'ema_26_r',
            'adx', 'atr_r',
            'bb_pos', 'bb_width',
            'vol_10', 'vol_20',
            'vol_ratio', 'vol_ma_r', 'obv_ch',
            'mom_3', 'mom_5', 'mom_10',
            'hl_r', 'co_r'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create 28 features from OHLCV data"""
        if not SKLEARN_AVAILABLE or df is None or len(df) < 50:
            return pd.DataFrame()
        
        f = pd.DataFrame(index=df.index)
        c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
        
        # RSI
        for p in [7, 14]:
            d = c.diff()
            g = d.where(d > 0, 0).rolling(p).mean()
            loss = (-d.where(d < 0, 0)).rolling(p).mean()
            f[f'rsi_{p}'] = 100 - (100 / (1 + g / (loss + 1e-10)))
        
        # MACD
        e12, e26 = c.ewm(12).mean(), c.ewm(26).mean()
        f['macd'] = e12 - e26
        f['macd_signal'] = f['macd'].ewm(9).mean()
        f['macd_hist'] = f['macd'] - f['macd_signal']
        
        # Stochastic
        low14, high14 = l.rolling(14).min(), h.rolling(14).max()
        f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-10)
        f['stoch_d'] = f['stoch_k'].rolling(3).mean()
        
        # SMA/EMA ratios
        for p, name in [(9