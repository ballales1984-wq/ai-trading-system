"""
ML Signal Model for Crypto Trading System
========================================
Real Machine Learning supervised model for signal generation.
Uses ensemble methods with proper feature engineering.

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os


@dataclass
class ModelResult:
    """Container for ML model results"""
    prediction: np.ndarray
    probability: np.ndarray
    accuracy: float
    feature_importance: Dict[str, float]


class MLSignalModel:
    """
    Machine Learning model for generating trading signals.
    Uses Random Forest and Gradient Boosting ensemble methods.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42
    ):
        """
        Initialize ML Signal Model.
        
        Parameters:
        -----------
        model_type : str
            Model type: 'random_forest' or 'gradient_boosting'
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum tree depth
        random_state : int
            Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicators
        
        Returns:
        --------
        pd.DataFrame : Features ready for ML
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns features
        if 'close' in df.columns:
            features['returns'] = df['close'].pct_change()
            features['returns_lag_1'] = features['returns'].shift(1)
            features['returns_lag_2'] = features['returns'].shift(2)
            features['returns_lag_3'] = features['returns'].shift(3)
        
        # Technical indicator features
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
            features['rsi_ma'] = df['rsi'].rolling(5).mean()
        
        if 'macd' in df.columns:
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
            features['macd_histogram'] = df['macd_histogram']
        
        if 'bb_position' in df.columns:
            features['bb_position'] = df['bb_position']
            features['bb_width'] = df.get('bb_width', 0)
        
        # Moving average features
        for col in ['ema_9', 'ema_21', 'ema_50']:
            if col in df.columns:
                features[col] = df[col]
                # Price relative to EMA
                if 'close' in df.columns:
                    features[f'{col}_ratio'] = df['close'] / df[col]
        
        # Volatility features
        if 'atr' in df.columns and 'close' in df.columns:
            features['atr_pct'] = df['atr'] / df['close']
            features['atr_pct_ma'] = features['atr_pct'].rolling(5).mean()
        
        # Volume features
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_ma'] = df['volume'].rolling(5).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # Trend features
        if 'close' in df.columns:
            for window in [5, 10, 20]:
                features[f'close_ma_{window}'] = df['close'].rolling(window).mean()
                features[f'close_ma_ratio_{window}'] = df['close'] / features[f'close_ma_{window}']
        
        # Momentum features
        if 'close' in df.columns:
            for period in [5, 10, 20]:
                features[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Fill NaN and replace infinities
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def create_target(
        self,
        df: pd.DataFrame,
        forward_periods: int = 1,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Create target variable for supervised learning.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        forward_periods : int
            Number of periods to look ahead
        threshold : float
            Return threshold for classification
        
        Returns:
        --------
        pd.Series : Target variable (1 = buy, 0 = hold/sell)
        """
        future_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
        
        # Binary classification: 1 if positive return, 0 otherwise
        target = (future_returns > threshold).astype(int)
        
        return target
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        forward_periods: int = 1,
        test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        forward_periods : int
            Forward periods for target
        test_size : float
            Test set proportion
        
        Returns:
        --------
        Tuple : (X_train, X_test, y_train, y_test)
        """
        # Prepare features
        X = self.prepare_features(df)
        
        # Create target
        y = self.create_target(df, forward_periods)
        
        # Align X and y
        valid_idx = X.index.intersection(y.index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Remove rows with NaN in target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Time-series split (no shuffle!)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.3
    ) -> Dict[str, float]:
        """
        Train the ML model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with features
        test_size : float
            Test set proportion
        
        Returns:
        --------
        Dict[str, float] : Training metrics
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size=test_size)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        self.is_trained = True
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> ModelResult:
        """
        Generate predictions using trained model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for prediction
        
        Returns:
        --------
        ModelResult : Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale
        X_scaled = self.scaler.transform(X.values)
        
        # Predict
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled)[:, 1]
        
        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return ModelResult(
            prediction=prediction,
            probability=probability,
            accuracy=0.0,  # Calculated during training
            feature_importance=importance
        )
    
    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from predictions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for prediction
        
        Returns:
        --------
        pd.Series : Trading signals (BUY/SELL/HOLD)
        """
        result = self.predict(df)
        
        signals = pd.Series('HOLD', index=df.index)
        
        # Buy when probability > 0.6
        signals[result.probability > 0.6] = 'BUY'
        # Sell when probability < 0.4
        signals[result.probability < 0.4] = 'SELL'
        
        return signals
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Time-series cross-validation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for CV
        n_splits : int
            Number of splits
        
        Returns:
        --------
        Dict[str, List[float]] : CV scores for each fold
        """
        X, y = self.prepare_features(df), self.create_target(df)
        
        # Align
        valid_idx = X.index.intersection(y.index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and predict
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        return scores
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.is_trained = data['is_trained']


class EnsembleMLModel:
    """Ensemble of multiple ML models for better predictions."""
    
    def __init__(self):
        self.models = {
            'rf': MLSignalModel('random_forest'),
            'gb': MLSignalModel('gradient_boosting')
        }
    
    def train_all(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train all models."""
        results = {}
        for name, model in self.models.items():
            metrics = model.train(df)
            results[name] = metrics
        return results
    
    def predict_ensemble(self, df: pd.DataFrame) -> np.ndarray:
        """Average predictions from all models."""
        probabilities = []
        
        for model in self.models.values():
            result = model.predict(df)
            probabilities.append(result.probability)
        
        # Average probabilities
        return np.mean(probabilities, axis=0)
    
    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate ensemble signals."""
        avg_prob = self.predict_ensemble(df)
        
        signals = pd.Series('HOLD', index=df.index)
        signals[avg_prob > 0.55] = 'BUY'
        signals[avg_prob < 0.45] = 'SELL'
        
        return signals
