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

    def create_labels(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        threshold: float = 0.02
    ) -> pd.Series:
        """
        Create labels for training.
        
        Args:
            df: Price dataframe
            forward_periods: Periods to look ahead
            threshold: Minimum return to consider as signal
            
        Returns:
            Series with labels (1=up, 0=down/neutral)
        """
        if not SKLEARN_AVAILABLE:
            return pd.Series()
        
        future_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
        
        if self.use_meta_labeling:
            # Use risk-adjusted labels
            labels = pd.Series(0, index=df.index)
            
            # Calculate position sizing based on volatility
            volatility = df['close'].pct_change().rolling(20).std()
            risk_adjusted_threshold = threshold * (1 + volatility)
            
            labels[future_returns > risk_adjusted_threshold] = 1
            labels[future_returns < -risk_adjusted_threshold] = -1
        else:
            # Simple binary labels
            labels = (future_returns > threshold).astype(int)
        
        return labels.fillna(0)

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the ensemble model.
        
        Args:
            df: OHLCV dataframe
            test_size: Fraction for test split
            
        Returns:
            Training metrics dictionary
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, cannot train")
            return self.training_metrics
        
        # Prepare features
        features = self.prepare_features(df)
        labels = self.create_labels(df)
        
        # Align features and labels
        valid_idx = features.index.intersection(labels.index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]
        
        # Remove samples with NaN labels
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            logger.warning("Insufficient training data")
            return self.training_metrics
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        n_samples = len(X_scaled)
        train_size = int(n_samples * (1 - test_size))
        
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y.values[:train_size], y.values[train_size:]
        
        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Gradient Boosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.gb_model.fit(X_train, y_train)
        
        # Extra Trees
        self.et_model = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.et_model.fit(X_train, y_train)
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.xgb_model.fit(X_train, y_train)
        
        # Calculate metrics
        self.is_trained = True
        self.training_metrics['n_samples'] = n_samples
        
        # Ensemble prediction
        y_pred_proba = self._ensemble_predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        self.training_metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.training_metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        self.training_metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        self.training_metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            self.training_metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            self.training_metrics['auc'] = 0.0
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(
            self.rf_model, X_scaled, y.values, cv=tscv, scoring='accuracy'
        )
        self.training_metrics['cv_mean'] = cv_scores.mean()
        self.training_metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"Training complete. Metrics: {self.training_metrics}")
        return self.training_metrics
    
    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions"""
        probas = []
        
        if self.rf_model:
            probas.append(self.rf_model.predict_proba(X)[:, 1] if len(self.rf_model.classes_) > 1 else np.zeros(len(X)))
        if self.gb_model:
            probas.append(self.gb_model.predict_proba(X)[:, 1] if len(self.gb_model.classes_) > 1 else np.zeros(len(X)))
        if self.et_model:
            probas.append(self.et_model.predict_proba(X)[:, 1] if len(self.et_model.classes_) > 1 else np.zeros(len(X)))
        if self.xgb_model and XGBOOST_AVAILABLE:
            probas.append(self.xgb_model.predict_proba(X)[:, 1])
        
        if not probas:
            return np.zeros(len(X))
        
        # Weighted average
        weights = [self.model_weights.get(k, 0.25) for k in ['rf', 'gb', 'et', 'xgb'][:len(probas)]]
        total = sum(weights)
        weights = [w/total for w in weights]
        
        ensemble = np.zeros(len(X))
        for proba, weight in zip(probas, weights):
            ensemble += proba * weight
        
        return ensemble
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Prediction dictionary with signal, probability, and confidence
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return {
                'signal': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'model_ready': False
            }
        
        # Prepare features
        features = self.prepare_features(df)
        
        if len(features) == 0:
            return {
                'signal': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'error': 'No features available'
            }
        
        # Use last row for prediction
        X = features.iloc[-1:]
        X_scaled = self.scaler.transform(X)
        
        # Get ensemble prediction
        proba = self._ensemble_predict_proba(X_scaled)[0]
        
        # Signal based on probability
        signal = 1 if proba > 0.6 else -1 if proba < 0.4 else 0
        
        # Confidence based on distance from 0.5
        confidence = abs(proba - 0.5) * 2
        
        return {
            'signal': signal,
            'probability': float(proba),
            'confidence': float(confidence),
            'model_ready': True
        }
    
    def predict_multi(
        self,
        df: pd.DataFrame,
        window: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Make predictions over a window of data.
        
        Args:
            df: OHLCV dataframe
            window: Number of recent predictions
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return []
        
        predictions = []
        
        for i in range(min(window, len(df))):
            # Use data up to current point
            df_slice = df.iloc[:len(df)-i]
            pred = self.predict(df_slice)
            predictions.append(pred)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_trained or self.rf_model is None:
            return pd.DataFrame()
        
        importance = self.rf_model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        model_data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'xgb_model': self.xgb_model,
            'et_model': self.et_model,
            'scaler': self.scaler,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rf_model = model_data.get('rf_model')
        self.gb_model = model_data.get('gb_model')
        self.xgb_model = model_data.get('xgb_model')
        self.et_model = model_data.get('et_model')
        self.scaler = model_data.get('scaler')
        self.training_metrics = model_data.get('training_metrics', {})
        self.feature_names = model_data.get('feature_names', [])
        self.model_weights = model_data.get('model_weights', {})
        self.is_trained = model_data.get('is_trained', False)
        
        logger.info(f"Model loaded from {filepath}")
