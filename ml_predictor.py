"""
Machine Learning Prediction Module
Uses historical price data to predict future price movements
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML predictions disabled.")


class PricePredictor:
    """
    Machine Learning-based price movement predictor
    Uses Random Forest and Gradient Boosting for classification
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.rf_model = None
        self.gb_model = None
        self.is_trained = False
        self.feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio',
            'price_momentum', 'volatility', 'trend_strength'
        ]
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        
        # Bollinger Bands position
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume ratio
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price momentum
        features['price_momentum'] = df['close'].pct_change(periods=5)
        
        # Volatility
        features['volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        
        # Trend strength (simplified ADX)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features['trend_strength'] = atr / df['close']
        
        return features.fillna(0)
    
    def create_labels(self, df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """
        Create labels: 1 = price goes up > threshold, 0 = otherwise
        
        Args:
            df: OHLCV dataframe
            threshold: Minimum price change to consider as 'up'
            
        Returns:
            Series with labels
        """
        future_returns = df['close'].shift(-1) / df['close'] - 1
        labels = (future_returns > threshold).astype(int)
        return labels.fillna(0)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the ML models
        
        Args:
            df: OHLCV dataframe with sufficient history
            
        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE:
            return {'status': 'ERROR', 'message': 'scikit-learn not available'}
        
        try:
            # Prepare features
            X = self.prepare_features(df)
            y = self.create_labels(df)
            
            # Remove rows with NaN labels
            valid_idx = y != 0
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 50:
                return {'status': 'ERROR', 'message': 'Insufficient data for training'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train_scaled, y_train)
            rf_accuracy = self.rf_model.score(X_test_scaled, y_test)
            
            # Train Gradient Boosting
            self.gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.gb_model.fit(X_train_scaled, y_train)
            gb_accuracy = self.gb_model.score(X_test_scaled, y_test)
            
            self.is_trained = True
            
            return {
                'status': 'OK',
                'rf_accuracy': rf_accuracy,
                'gb_accuracy': gb_accuracy,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict next price movement
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            # Try to train first
            train_result = self.train(df)
            if train_result.get('status') != 'OK':
                return train_result
        
        try:
            # Prepare features
            X = self.prepare_features(df)
            X_last = X.iloc[-1:].values
            
            # Scale
            X_scaled = self.scaler.transform(X_last)
            
            # Predictions
            rf_pred = self.rf_model.predict(X_scaled)[0]
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            
            gb_pred = self.gb_model.predict(X_scaled)[0]
            gb_proba = self.gb_model.predict_proba(X_scaled)[0]
            
            # Ensemble prediction (average)
            avg_proba = (rf_proba[1] + gb_proba[1]) / 2
            ensemble_pred = 1 if avg_proba > 0.5 else 0
            
            # Confidence
            confidence = max(avg_proba, 1 - avg_proba)
            
            return {
                'status': 'OK',
                'prediction': 'UP' if ensemble_pred == 1 else 'DOWN',
                'confidence': confidence,
                'rf_probability': rf_proba[1],
                'gb_probability': gb_proba[1],
                'model': 'Ensemble (RF + GB)'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from trained models
        
        Returns:
            DataFrame with feature importance or None
        """
        if not self.is_trained:
            return None
            
        try:
            rf_importance = self.rf_model.feature_importances_
            gb_importance = self.gb_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'random_forest': rf_importance,
                'gradient_boosting': gb_importance,
                'average': (rf_importance + gb_importance) / 2
            }).sort_values('average', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return None
    
    def save_model(self, filepath: str) -> bool:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            logger.warning("Cannot save: model not trained")
            return False
        
        try:
            import pickle
            model_data = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.rf_model = model_data['rf_model']
            self.gb_model = model_data['gb_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """
        Export model configuration as dictionary
        
        Returns:
            Dict with model configuration
        """
        return {
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'has_rf_model': self.rf_model is not None,
            'has_gb_model': self.gb_model is not None,
            'has_scaler': self.scaler is not None
        }


class SimpleMovingAveragePredictor:
    """
    Simple predictor using moving averages for trend prediction
    No ML required - uses technical analysis rules
    """
    
    def __init__(self):
        """Initialize predictor"""
        self.name = "SMA Crossover Predictor"
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict based on moving average crossovers
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Prediction results
        """
        try:
            # Calculate moving averages
            sma_short = df['close'].rolling(window=9).mean()
            sma_medium = df['close'].rolling(window=21).mean()
            sma_long = df['close'].rolling(window=50).mean()
            
            current_price = df['close'].iloc[-1]
            
            # Buy signal: short MA above medium MA, both above long MA
            buy_condition = (sma_short.iloc[-1] > sma_medium.iloc[-1] > sma_long.iloc[-1])
            
            # Sell signal: short MA below medium MA
            sell_condition = sma_short.iloc[-1] < sma_medium.iloc[-1]
            
            if buy_condition:
                prediction = 'UP'
                confidence = 0.7
                reason = "Bullish crossover (9 > 21 > 50 SMA)"
            elif sell_condition:
                prediction = 'DOWN'
                confidence = 0.7
                reason = "Bearish crossover (9 < 21 SMA)"
            else:
                prediction = 'SIDEWAYS'
                confidence = 0.5
                reason = "No clear trend"
            
            return {
                'status': 'OK',
                'prediction': prediction,
                'confidence': confidence,
                'reason': reason,
                'current_price': current_price,
                'sma_9': sma_short.iloc[-1],
                'sma_21': sma_medium.iloc[-1],
                'sma_50': sma_long.iloc[-1] if not pd.isna(sma_long.iloc[-1]) else None,
                'model': self.name
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'status': 'ERROR', 'message': str(e)}


# Singleton instances
_ml_predictor = None
_sma_predictor = None

def get_ml_predictor() -> PricePredictor:
    """Get ML predictor singleton"""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = PricePredictor()
    return _ml_predictor

def get_sma_predictor() -> SimpleMovingAveragePredictor:
    """Get SMA predictor singleton"""
    global _sma_predictor
    if _sma_predictor is None:
        _sma_predictor = SimpleMovingAveragePredictor()
    return _sma_predictor


if __name__ == "__main__":
    # Test the module
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
    np.random.seed(42)
    
    prices = 45000 + np.cumsum(np.random.randn(100) * 500)
    df = pd.DataFrame({
        'open': prices + np.random.randn(100) * 100,
        'high': prices + abs(np.random.randn(100) * 200),
        'low': prices - abs(np.random.randn(100) * 200),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    print("=" * 50)
    print("ML PRICE PREDICTION TEST")
    print("=" * 50)
    
    # Test SMA predictor (works without sklearn)
    sma_pred = get_sma_predictor()
    result = sma_pred.predict(df)
    
    print(f"\nSMA Predictor:")
    print(f"  Prediction: {result.get('prediction')}")
    print(f"  Confidence: {result.get('confidence'):.0%}")
    print(f"  Reason: {result.get('reason')}")
    print(f"  Current Price: ${result.get('current_price'):,.2f}")
    
    # Test ML predictor (requires sklearn)
    if SKLEARN_AVAILABLE:
        ml_pred = get_ml_predictor()
        train_result = ml_pred.train(df)
        print(f"\nML Model Training:")
        print(f"  Status: {train_result.get('status')}")
        print(f"  RF Accuracy: {train_result.get('rf_accuracy', 0):.1%}")
        print(f"  GB Accuracy: {train_result.get('gb_accuracy', 0):.1%}")
        
        pred_result = ml_pred.predict(df)
        print(f"\nML Predictor:")
        print(f"  Prediction: {pred_result.get('prediction')}")
        print(f"  Confidence: {pred_result.get('confidence', 0):.0%}")
    else:
        print("\nML Predictor requires scikit-learn installation")
