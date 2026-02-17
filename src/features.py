# src/features.py
"""
Advanced Features Module for Quantitative Trading
Features that improve ML model predictive power
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. REALIZED VOLATILITY
# ---------------------------------------------------------

def realized_volatility(df, window=14):
    """Calculate realized volatility (annualized)"""
    returns = np.log(df['close'] / df['close'].shift(1))
    df[f'rvol_{window}'] = returns.rolling(window).std() * np.sqrt(365)
    return df


# ---------------------------------------------------------
# 2. NORMALIZED MOMENTUM
# ---------------------------------------------------------

def normalized_momentum(df, window=14):
    """Calculate normalized momentum (risk-adjusted)"""
    returns = df['close'].pct_change(window)
    vol = df['close'].pct_change().rolling(window).std()
    df[f'norm_mom_{window}'] = returns / vol
    return df


# ---------------------------------------------------------
# 3. TREND REGIME (ADX + EMA SLOPE)
# ---------------------------------------------------------

def trend_regime(df, ema_window=20, slope_window=10):
    """Detect trend regime: +1 (bull), -1 (bear), 0 (range)"""
    df[f'ema_{ema_window}'] = df['close'].ewm(span=ema_window).mean()
    df[f'ema_slope_{ema_window}'] = df[f'ema_{ema_window}'].diff(slope_window)
    
    # regime:
    # +1 = uptrend
    # -1 = downtrend
    # 0 = range
    df['trend_regime'] = 0
    df.loc[df[f'ema_slope_{ema_window}'] > 0, 'trend_regime'] = 1
    df.loc[df[f'ema_slope_{ema_window}'] < 0, 'trend_regime'] = -1
    
    return df


# ---------------------------------------------------------
# 4. VOLUME IMBALANCE
# ---------------------------------------------------------

def volume_imbalance(df, window=10):
    """Calculate volume imbalance (buy vs sell pressure)"""
    df['vol_buy'] = df['volume'] * (df['close'] > df['open']).astype(int)
    df['vol_sell'] = df['volume'] * (df['close'] < df['open']).astype(int)
    
    df[f'vol_imbalance_{window}'] = (
        df['vol_buy'].rolling(window).sum() -
        df['vol_sell'].rolling(window).sum()
    )
    
    return df.drop(columns=['vol_buy', 'vol_sell'])


# ---------------------------------------------------------
# 5. ROLLING CORRELATIONS
# ---------------------------------------------------------

def rolling_correlation(df, other_df, window=30, label="corr"):
    """Calculate rolling correlation with another asset"""
    merged = df[['close']].join(other_df[['close']], lsuffix='_a', rsuffix='_b')
    merged[label] = merged['close_a'].rolling(window).corr(merged['close_b'])
    df[label] = merged[label]
    return df


# ---------------------------------------------------------
# 6. MULTI-TIMEFRAME FEATURES
# ---------------------------------------------------------

def multi_timeframe_ema(df, ema_short=20, ema_mid=50, ema_long=200):
    """Calculate EMAs across multiple timeframes"""
    df[f'ema_{ema_short}'] = df['close'].ewm(span=ema_short).mean()
    df[f'ema_{ema_mid}'] = df['close'].ewm(span=ema_mid).mean()
    df[f'ema_{ema_long}'] = df['close'].ewm(span=ema_long).mean()
    
    df['ema_ratio_short_mid'] = df[f'ema_{ema_short}'] / df[f'ema_{ema_mid}']
    df['ema_ratio_mid_long'] = df[f'ema_{ema_mid}'] / df[f'ema_{ema_long}']
    
    return df


# ---------------------------------------------------------
# 7. PRICE MOMENTUM OSCILLATORS
# ---------------------------------------------------------

def momentum_oscillators(df):
    """Calculate RSI, Stochastic, MACD"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    low14 = df['low'].rolling(window=14).min()
    high14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


# ---------------------------------------------------------
# 8. ATR (Average True Range)
# ---------------------------------------------------------

def atr(df, window=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window).mean()
    df['atr_percent'] = (df['atr'] / df['close']) * 100
    
    return df


# ---------------------------------------------------------
# 9. ON-BALANCE VOLUME (OBV)
# ---------------------------------------------------------

def on_balance_volume(df):
    """Calculate On-Balance Volume"""
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    return df


# ---------------------------------------------------------
# 10. COMPLETE PIPELINE
# ---------------------------------------------------------

def generate_advanced_features(df, other_assets=None):
    """
    Apply all advanced features in one call.
    
    Args:
        df: DataFrame with OHLCV data
        other_assets: dict e.g. {"ETH": eth_df, "SOL": sol_df}
    
    Returns:
        DataFrame with added features
    """
    df = realized_volatility(df)
    df = normalized_momentum(df)
    df = trend_regime(df)
    df = volume_imbalance(df)
    df = multi_timeframe_ema(df)
    df = momentum_oscillators(df)
    df = atr(df)
    df = on_balance_volume(df)
    
    if other_assets:
        for name, other_df in other_assets.items():
            df = rolling_correlation(df, other_df, label=f"corr_{name}")
    
    return df
