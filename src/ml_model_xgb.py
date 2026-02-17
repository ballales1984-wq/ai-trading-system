"""
XGBoost Signal Model - Advanced ML for Trading
============================================
XGBoost-based model for financial signal generation.
Optimized for time series with proper feature engineering.

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


@dataclass
class XGBResult:
    """XGBoost model results"""
    predictions: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    feature_importance: Dict[str, float]
    tree_model: object


class XGBSignalModel:
    """
    XGBoost-based trading signal model.
    
    Optimized for financial time series with:
    - Proper feature engineering
    - Time-series split validation
    - Feature importance analysis
    """
    
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        """Initialize XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            use_label_encoder=False
        )
        
        self.is_fitted = False
        self.feature_names = []
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive feature set for XGBoost.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicators
        
        Returns:
        --------
        pd.DataFrame : Feature matrix
        """
        features = pd.DataFrame(index=df.index)
        
        # === Price-based features ===
        # Returns at different horizons
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'ret_{lag}'] = df['close'].pct_change(lag)
        
        # Log returns
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # === Volatility features ===
        for window in [5, 10, 20, 50]:
            features[f'vol_{window}'] = df['close'].pct_change().rolling(window).std()
        
        # Volatility ratio
        features['vol_ratio'] = features['vol_10'] / features['vol_20']
        
        # === Trend features ===
        for window in [10, 20, 50, 100, 200]:
            features[f'sma_{window}'] = df['close'].rolling(window).mean()
            features[f'sma_ratio_{window}'] = df['close'] / features[f'sma_{window}']
        
        # EMA features
        for window in [9, 21, 50]:
            if f'ema_{window}' in df.columns:
                features[f'ema_{window}'] = df[f'ema_{window}']
                features[f'ema_ratio_{window}'] = df['close'] / df[f'ema_{window}']
        
        # === Technical indicator features ===
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
            features['rsi_ma'] = df['rsi'].rolling(5).mean()
            features['rsi_std'] = df['rsi'].rolling(10).std()
        
        if 'macd' in df.columns:
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
            features['macd_hist'] = df['macd_histogram']
        
        if 'bb_position' in df.columns:
            features['bb_position'] = df['bb_position']
            features['bb_width'] = df.get('bb_width', 0)
        
        # === Momentum features ===
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Rate of change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # === Volume features ===
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_ma_5'] = df['volume'].rolling(5).mean()
            features['volume_ma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma_20']
        
        if 'obv' in df.columns:
            features['obv'] = df['obv']
            features['obv_ma'] = df['obv'].rolling(10).mean()
        
        # === ATR features ===
        if 'atr' in df.columns:
            features['atr'] = df['atr']
            features['atr_pct'] = df['atr'] / df['close']
            features['atr_ma'] = df['atr'].rolling(10).mean()
        
        # === Regime features ===
        if 'regime' in df.columns:
            regime_dummies = pd.get_dummies(df['regime'], prefix='regime')
            features = pd.concat([features, regime_dummies], axis=1)
        
        # === Sentiment features ===
        if 'sentiment' in df.columns:
            features['sentiment'] = df['sentiment']
            features['sentiment_ma'] = df['sentiment'].rolling(3).mean()
            features['sentiment_std'] = df['sentiment'].rolling(5).std()
        
        # === Cross-asset features ===
        if 'btc_close' in df.columns:
            features['btc_corr'] = df['close'].corr(df['btc_close'])
        
        # === Time features ===
        if hasattr(df.index, 'dayofweek'):
            features['day_of_week'] = df.index.dayofweek
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def build_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Build target variable for supervised learning.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        horizon : int
            Forward periods to predict
        threshold : float
            Return threshold for classification
        
        Returns:
        --------
        pd.Series : Binary target (1 = buy, 0 = no position)
        """
        future_return = df['close'].pct_change(horizon).shift(-horizon)
        target = (future_return > threshold).astype(int)
        
        return target
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple:
        """Prepare train/test split."""
        X = self.build_features(df)
        y = self.build_target(df)
        
        # Align
        valid_idx = X.index.intersection(y.index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Remove NaN targets
        valid = ~y.isna()
        X = X[valid]
        y = y[valid]
        
        # Time-series split (no shuffle!)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Fit XGBoost model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        test_size : float
            Test set proportion
        
        Returns:
        --------
        Dict : Training metrics
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        
        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        self.is_fitted = True
        
        return {
            'accuracy': accuracy,
            'test_size': len(y_test),
            'feature_importance': importance
        }
    
    def predict(self, df: pd.DataFrame) -> XGBResult:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.build_features(df)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return XGBResult(
            predictions=predictions,
            probabilities=probabilities,
            accuracy=0.0,
            feature_importance=importance,
            tree_model=self.model
        )
    
    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from predictions."""
        result = self.predict(df)
        
        signals = pd.Series('HOLD', index=df.index)
        
        # Buy when probability > 0.55
        signals[result.probabilities > 0.55] = 'BUY'
        # Sell when probability < 0.45
        signals[result.probabilities < 0.45] = 'SELL'
        
        return signals
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top N most important features."""
        if not self.is_fitted:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(n)


class EnsembleModels:
    """Ensemble of XGBoost + Random Forest for robust signals."""
    
    def __init__(self):
        self.xgb_model = XGBSignalModel()
        self.rf_model = None  # Would use sklearn RandomForest
    
    def fit(self, df: pd.DataFrame) -> Dict:
        """Fit all models."""
        xgb_metrics = self.xgb_model.fit(df)
        
        return {
            'xgboost': xgb_metrics
        }
    
    def predict_ensemble(self, df: pd.DataFrame) -> np.ndarray:
        """Average predictions from all models."""
        xgb_result = self.xgb_model.predict(df)
        
        # Could add RF predictions here
        return xgb_result.probabilities
    
    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate ensemble signals."""
        avg_prob = self.predict_ensemble(df)
        
        signals = pd.Series('HOLD', index=df.index)
        signals[avg_prob > 0.55] = 'BUY'
        signals[avg_prob < 0.45] = 'SELL'
        
        return signals
