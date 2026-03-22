#!/usr/bin/env python3
"""
Enhanced ML Training Script with Walk-Forward Validation
=======================================================
Advanced training script with:
- Walk-forward cross-validation for time series
- Multiple symbol training
- Model versioning
- Feature importance analysis
- Better hyperparameter tuning

Usage:
    python train_ml_enhanced.py --symbol BTCUSDT --interval 1h --candles 1000
    python train_ml_enhanced.py --all-symbols
"""

import sys
import os
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ML imports
try:
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier, 
        ExtraTreesClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, classification_report
    )
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn not available")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using sklearn alternatives")

API_BASE = os.getenv("AI_TRADING_API_URL", "http://localhost:8000")


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_candle_data(symbol: str, interval: str = "1h", limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fetch candle data from API"""
    try:
        url = f"{API_BASE}/api/v1/market/candles/{symbol}"
        params = {"interval": interval, "limit": limit}
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Clean data
                df = df.dropna()
                return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
    return None


# =============================================================================
# Feature Engineering
# =============================================================================

def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # RSI (multiple periods)
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic
    low14 = low.rolling(window=14).min()
    high14 = high.rolling(window=14).max()
    df['stoch_k'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Moving Average Ratios
    for period in [9, 21, 50, 200]:
        sma = close.rolling(window=period).mean()
        df[f'sma_{period}_ratio'] = close / sma
    
    for period in [12, 26]:
        ema = close.ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_ratio'] = close / ema
    
    # ADX
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    atr = (high - low).rolling(window=14).mean()
    df['adx'] = abs(plus_dm - minus_dm) / (atr + 1e-10) * 100
    df['atr_ratio'] = atr / close
    
    # Bollinger Bands
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    bb_upper = sma20 + (std20 * 2)
    bb_lower = sma20 - (std20 * 2)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df['bb_width'] = (bb_upper - bb_lower) / sma20
    
    # Volatility
    df['volatility_10'] = close.pct_change().rolling(window=10).std()
    df['volatility_20'] = close.pct_change().rolling(window=20).std()
    
    # Volume indicators
    df['volume_ratio'] = volume / volume.rolling(window=20).mean()
    df['volume_ma_ratio'] = volume / volume.rolling(window=5).mean()
    df['obv_change'] = (volume * np.sign(close.diff())).rolling(window=10).sum()
    
    # Price momentum
    for period in [3, 5, 10]:
        df[f'price_momentum_{period}'] = close.pct_change(periods=period)
    
    # High/Low ratio
    df['high_low_ratio'] = (high - low) / close
    df['close_open_ratio'] = (close - open_price) / open_price
    
    # Additional features
    # On-Balance Volume ratio
    df['obv_ratio'] = df['obv_change'] / (volume + 1e-10)
    
    # VWAP approximation
    df['vwap'] = ((high + low + close) / 3 * volume).rolling(window=20).sum() / (volume.rolling(window=20).sum() + 1e-10)
    df['vwap_ratio'] = close / df['vwap']
    
    # Williams %R
    df['williams_r'] = -100 * (high14 - close) / (high14 - low14 + 1e-10)
    
    # CCI
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    return df


# =============================================================================
# Label Creation
# =============================================================================

def create_labels(df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.02) -> pd.Series:
    """Create trading labels with risk-adjusted threshold"""
    future_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
    
    # Risk-adjusted threshold based on volatility
    volatility = df['close'].pct_change().rolling(20).std()
    risk_adjusted_threshold = threshold * (1 + volatility)
    
    labels = pd.Series(0, index=df.index)
    labels[future_returns > risk_adjusted_threshold] = 1   # Buy signal
    labels[future_returns < -risk_adjusted_threshold] = -1  # Sell signal
    
    return labels.fillna(0)


# =============================================================================
# Feature Preparation
# =============================================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and labels"""
    # Calculate indicators
    df = calculate_advanced_indicators(df)
    
    # Create labels
    labels = create_labels(df)
    
    # Select features
    feature_columns = [
        'rsi_7', 'rsi_14', 'rsi_21',
        'macd', 'macd_signal', 'macd_histogram',
        'stoch_k', 'stoch_d',
        'sma_9_ratio', 'sma_21_ratio', 'sma_50_ratio', 'sma_200_ratio',
        'ema_12_ratio', 'ema_26_ratio',
        'adx', 'atr_ratio',
        'bb_position', 'bb_width',
        'volatility_10', 'volatility_20',
        'volume_ratio', 'volume_ma_ratio', 'obv_change',
        'price_momentum_3', 'price_momentum_5', 'price_momentum_10',
        'high_low_ratio', 'close_open_ratio',
        'vwap_ratio', 'williams_r', 'cci'
    ]
    
    # Filter available columns
    available_features = [col for col in feature_columns if col in df.columns]
    features = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    
    return features, labels


# =============================================================================
# Walk-Forward Validation
# =============================================================================

def walk_forward_validation(
    X: np.ndarray, 
    y: np.ndarray,
    n_splits: int = 5,
    test_size: int = 50
) -> Dict[str, Any]:
    """
    Perform walk-forward cross-validation for time series
    
    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of walk-forward splits
        test_size: Size of test window
        
    Returns:
        Dictionary with validation metrics
    """
    if not SKLEARN_AVAILABLE:
        return {}
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Ensure we have enough data
        if len(test_idx) < 10:
            continue
            
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Skip if not enough samples
        if len(np.unique(y_train)) < 2:
            continue
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Map labels to 0,1,2 for multiclass
        y_train_mapped = y_train.copy()
        y_test_mapped = y_test.copy()
        
        unique_labels = np.unique(y_train)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_train_mapped = np.array([label_map.get(v, 1) for v in y_train])
        y_test_mapped = np.array([label_map.get(v, 1) for v in y_test])
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42 + fold,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train_mapped)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test_mapped, y_pred)
        prec = precision_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
        
        fold_metrics.append({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
    
    # Average metrics
    if fold_metrics:
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'precision': np.mean([m['precision'] for m in fold_metrics]),
            'recall': np.mean([m['recall'] for m in fold_metrics]),
            'f1': np.mean([m['f1'] for m in fold_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
            'n_folds': len(fold_metrics)
        }
    else:
        avg_metrics = {}
    
    return avg_metrics


# =============================================================================
# Model Training
# =============================================================================

def train_enhanced_model(
    df: pd.DataFrame,
    symbol: str,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Train enhanced ML model with ensemble"""
    if not SKLEARN_AVAILABLE:
        return {}
    
    # Prepare features
    features, labels = prepare_features(df)
    
    # Align features and labels
    valid_idx = features.index.intersection(labels.index)
    X = features.loc[valid_idx].values
    y = labels.loc[valid_idx].values
    
    # Remove NaN labels
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 100:
        logger.warning(f"Insufficient data for {symbol}")
        return {}
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Map labels
    unique_labels = np.unique(y_train)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    reverse_map = {new: old for old, new in label_map.items()}
    y_train_mapped = np.array([label_map.get(v, 1) for v in y_train])
    y_test_mapped = np.array([label_map.get(v, 1) for v in y_test])
    
    # Train ensemble models
    models = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_mapped)
    models['rf'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train_mapped)
    models['gb'] = gb
    
    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    et.fit(X_train, y_train_mapped)
    models['et'] = et
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        xgb.fit(X_train, y_train_mapped)
        models['xgb'] = xgb
    
    # Ensemble prediction
    probas = []
    for name, model in models.items():
        probas.append(model.predict_proba(X_test))
    
    # Weighted average
    ensemble_proba = np.mean(probas, axis=0)
    y_pred = np.argmax(ensemble_proba, axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_mapped, y_pred),
        'precision': precision_score(y_test_mapped, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test_mapped, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test_matched, y_pred, average='weighted', zero_division=0),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    # Walk-forward validation
    wf_metrics = walk_forward_validation(X_scaled, y)
    metrics['walk_forward'] = wf_metrics
    
    # Feature importance (from RF)
    feature_names = features.columns.tolist()
    importance = rf.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:15]
    metrics['feature_importance'] = feature_importance
    
    # Save model
    model_data = {
        'models': models,
        'scaler': scaler,
        'label_map': label_map,
        'reverse_map': reverse_map,
        'feature_names': feature_names,
        'symbol': symbol,
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    model_path = f"data/ml_model_{symbol.replace('/', '_')}_enhanced.pkl"
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced ML Training with Walk-Forward Validation")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", default="1h", help="Timeframe")
    parser.add_argument("--candles", type=int, default=1000, help="Number of candles")
    parser.add_argument("--all-symbols", action="store_true", help="Train on all symbols")
    parser.add_argument("--forward-periods", type=int, default=5, help="Forward periods for labels")
    parser.add_argument("--threshold", type=float, default=0.02, help="Label threshold")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENHANCED ML TRAINING - Walk-Forward Validation")
    print("=" * 70)
    
    if args.all_symbols:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']
    else:
        symbols = [args.symbol]
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Training model for {symbol}")
        print(f"{'='*50}")
        
        # Fetch data
        df = fetch_candle_data(symbol, args.interval, args.candles)
        
        if df is None or len(df) < 100:
            print(f"Could not fetch enough data for {symbol}")
            continue
        
        print(f"Fetched {len(df)} candles")
        
        # Train model
        metrics = train_enhanced_model(df, symbol)
        
        if metrics:
            print(f"\n{symbol} Results:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
            print(f"  Precision: {metrics.get('precision', 0):.2%}")
            print(f"  Recall: {metrics.get('recall', 0):.2%}")
            print(f"  F1 Score: {metrics.get('f1', 0):.2%}")
            
            if 'walk_forward' in metrics and metrics['walk_forward']:
                wf = metrics['walk_forward']
                print(f"\n  Walk-Forward Validation ({wf.get('n_folds', 0)} folds):")
                print(f"    Accuracy: {wf.get('accuracy', 0):.2%} (+/- {wf.get('std_accuracy', 0):.2%})")
                print(f"    F1: {wf.get('f1', 0):.2%}")
            
            print(f"\n  Top Features:")
            for feat, imp in metrics.get('feature_importance', [])[:5]:
                print(f"    {feat}: {imp:.3f}")
            
            all_results[symbol] = metrics
        else:
            print(f"Training failed for {symbol}")
    
    # Save results summary
    results_path = "data/training_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"Results saved to {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
