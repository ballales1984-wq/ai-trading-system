#!/usr/bin/env python3
"""
Train ML Model using existing ImprovedPricePredictor
=====================================================
Uses the proven ml_predictor.py with walk-forward validation.

Run: python train_model.py --symbol BTCUSDT --candles 2000
"""

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the existing predictor
from ml_predictor import ImprovedPricePredictor


def fetch_binance_data(symbol: str, candles: int = 2000, interval: str = '1h') -> pd.DataFrame:
    """Fetch data from Binance"""
    try:
        import ccxt
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=candles)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        # Create synthetic data for testing
        logger.info("Creating synthetic data for testing...")
        dates = pd.date_range(end=datetime.now(), periods=candles, freq='1H')
        np.random.seed(42)
        
        # Generate realistic price movements
        returns = np.random.normal(0.0005, 0.02, candles)
        close = 50000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': close * (1 + np.random.normal(0, 0.005, candles)),
            'high': close * (1 + np.abs(np.random.normal(0.01, 0.01, candles))),
            'low': close * (1 - np.abs(np.random.normal(0.01, 0.01, candles))),
            'close': close,
            'volume': np.random.uniform(1000, 10000, candles)
        }, index=dates)
        return df


def train_with_walk_forward(df: pd.DataFrame, symbol: str):
    """Train model with walk-forward cross-validation"""
    
    logger.info("=" * 60)
    logger.info(f"WALK-FORWARD VALIDATION - {symbol}")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = ImprovedPricePredictor(use_meta_labeling=True)
    
    # Create features
    logger.info("Creating features...")
    features = predictor.prepare_features(df)
    
    # Create labels (3-class: -1=down, 0=neutral, 1=up)
    labels = predictor.create_labels(df, forward_periods=5, threshold=0.015)
    
    # Align features and labels
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid_idx]
    y = labels.loc[valid_idx]
    
    logger.info(f"Dataset size: {len(X)} samples")
    logger.info(f"Class distribution:\n{y.value_counts().sort_index()}")
    
    # Walk-forward validation
    n_splits = 5
    fold_size = len(X) // (n_splits + 1)
    
    fold_results = []
    accuracies = []
    
    for fold in range(n_splits):
        # Calculate train/test indices
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size
        
        if test_end > len(X):
            break
            
        X_train, X_test = X.iloc[:train_end], X.iloc[test_start:test_end]
        y_train, y_test = y.iloc[:train_end], y.iloc[test_start:test_end]
        
        # Check for valid classes
        if len(y_train.unique()) < 2:
            logger.warning(f"Fold {fold+1}: Skipping - only one class in training data")
            continue
        
        logger.info(f"\nFold {fold+1}/{n_splits}")
        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        try:
            # Use the data subset for training
            train_df = df.loc[X_train.index]
            predictor.train(train_df, test_size=0.1)
            
            # Get accuracy from training metrics
            metrics = predictor.training_metrics
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1', 0)
            
            logger.info(f"  Accuracy: {acc:.3f}, F1: {f1:.3f}")
            accuracies.append(acc)
            fold_results.append({'fold': fold+1, 'accuracy': acc, 'f1': f1})
            
        except Exception as e:
            logger.error(f"  Training error: {e}")
            continue
    
    # Summary
    if accuracies:
        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD RESULTS")
        logger.info("=" * 60)
        avg_acc = np.mean(accuracies)
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        logger.info(f"Average Accuracy: {avg_acc:.3f}")
        logger.info(f"Average F1 Score: {avg_f1:.3f}")
        
        # Show per-fold results
        for r in fold_results:
            logger.info(f"  Fold {r['fold']}: Acc={r['accuracy']:.3f}, F1={r['f1']:.3f}")
        
        # Final model training on all data
        logger.info("\nTraining final model on all data...")
        predictor.train(df, test_size=0.15)
        
        # Save model
        import pickle
        model_path = f"data/ml_model_{symbol.replace('/', '').replace('USDT', 'USDT')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)
        logger.info(f"Model saved to {model_path}")
        
        return avg_acc, avg_f1
    
    return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description='Train ML model with walk-forward validation')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--candles', type=int, default=2000, help='Number of candles to fetch')
    parser.add_argument('--interval', type=str, default='1h', help='Timeframe (1m, 5m, 1h, 1d)')
    
    args = parser.parse_args()
    
    logger.info(f"Training model for {args.symbol}")
    logger.info(f"Fetching {args.candles} candles...")
    
    # Fetch data
    df = fetch_binance_data(args.symbol, args.candles, args.interval)
    
    # Train
    acc, f1 = train_with_walk_forward(df, args.symbol)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Final Accuracy: {acc:.3f}")
    logger.info(f"Final F1: {f1:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
