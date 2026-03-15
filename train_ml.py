#!/usr/bin/env python3
"""
Simple ML Training Script
Trains ML models using data from the local API
"""
import sys
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import ML modules
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Error: scikit-learn not available")
    sys.exit(1)

API_BASE = "http://localhost:8000"

def fetch_candle_data(symbol: str, interval: str = "1h", limit: int = 200):
    """Fetch candle data from API"""
    try:
        url = f"{API_BASE}/api/v1/market/candles/{symbol}"
        params = {"interval": interval, "limit": limit}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                # Parse timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Moving averages
    df['sma20'] = sma20
    df['sma50'] = df['close'].rolling(window=50).mean()
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price momentum
    df['momentum'] = df['close'].pct_change(periods=5)
    df['volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
    
    return df

def create_labels(df: pd.DataFrame, lookahead: int = 4):
    """Create trading labels (1=buy, 0=hold, -1=sell)"""
    # Future return
    df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
    
    # Create labels based on future return
    df['label'] = 0  # hold
    df.loc[df['future_return'] > 0.02, 'label'] = 1  # buy (>2%)
    df.loc[df['future_return'] < -0.02, 'label'] = -1  # sell (<-2%)
    
    return df

def train_model(df: pd.DataFrame, symbol: str):
    """Train ML model for a symbol"""
    # Features
    features = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'momentum', 'volatility']
    
    # Drop NaN
    df_train = df[features + ['label']].dropna()
    
    if len(df_train) < 50:
        print(f"Not enough data for {symbol}")
        return None
    
    X = df_train[features]
    y = df_train['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Time-based split
    )
    
    if len(X_train) < 20:
        print(f"Not enough training data for {symbol}")
        return None
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{symbol} Model Results:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Feature importances:")
    for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.3f}")
    
    # Save model
    model_path = f"data/ml_model_{symbol.replace('/', '_')}.pkl"
    import joblib
    joblib.dump({
        'model': rf,
        'features': features,
        'scaler': StandardScaler().fit(X_train),
        'trained_at': datetime.now().isoformat()
    }, model_path)
    print(f"  Model saved to: {model_path}")
    
    return rf

def main():
    print("=" * 60)
    print("ML TRAINING SCRIPT - Using Real Market Data")
    print("=" * 60)
    
    # Symbols to train on
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"Training model for {symbol}")
        print(f"{'='*40}")
        
        # Fetch data
        df = fetch_candle_data(symbol, interval="1h", limit=200)
        if df is None or len(df) < 50:
            print(f"Could not fetch enough data for {symbol}")
            continue
        
        print(f"Fetched {len(df)} candles")
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Create labels
        df = create_labels(df, lookahead=4)
        
        # Train
        train_model(df, symbol)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
