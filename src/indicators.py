"""
Technical Indicators Module for Crypto Trading System
====================================================
Professional technical analysis indicators including:
- RSI (Relative Strength Index)
- EMA (Exponential Moving Average)
- SMA (Simple Moving Average)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- VWAP (Volume Weighted Average Price)
- ATR (Average True Range)
- Stochastic Oscillator
- ADX (Average Directional Index)
- OBV (On Balance Volume)

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    period : int
        RSI period (default 14)
    
    Returns:
    --------
    pd.Series : RSI values (0-100)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use exponential moving average for smoother results
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    period : int
        EMA period
    
    Returns:
    --------
    pd.Series : EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    period : int
        SMA period
    
    Returns:
    --------
    pd.Series : SMA values
    """
    return series.rolling(window=period).mean()


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    fast_period : int
        Fast EMA period (default 12)
    slow_period : int
        Slow EMA period (default 26)
    signal_period : int
        Signal line period (default 9)
    
    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series] : (MACD line, Signal line, Histogram)
    """
    ema_fast = ema(series, fast_period)
    ema_slow = ema(series, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    window : int
        Window size (default 20)
    num_std : float
        Number of standard deviations (default 2.0)
    
    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series] : (Upper band, Middle band, Lower band)
    """
    middle_band = sma(series, window)
    std = series.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'high', 'low', 'close', 'volume' columns
    
    Returns:
    --------
    pd.Series : VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return vwap


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'high', 'low', 'close' columns
    period : int
        ATR period (default 14)
    
    Returns:
    --------
    pd.Series : ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(com=period-1, min_periods=period).mean()
    
    return atr


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'high', 'low', 'close' columns
    k_period : int
        %K period (default 14)
    d_period : int
        %D period (default 3)
    
    Returns:
    --------
    Tuple[pd.Series, pd.Series] : (%K, %D)
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k_percent = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'high', 'low', 'close' columns
    period : int
        ADX period (default 14)
    
    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series] : (ADX, +DI, -DI)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = atr(df, period)
    
    plus_di = 100 * (plus_dm.ewm(com=period-1).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(com=period-1).mean() / tr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(com=period-1).mean()
    
    return adx, plus_di, minus_di


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On Balance Volume (OBV).
    
    Parameters:
    -----------
    close : pd.Series
        Close price series
    volume : pd.Series
        Volume series
    
    Returns:
    --------
    pd.Series : OBV values
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    Parameters:
    -----------
    high : pd.Series
        High price series
    low : pd.Series
        Low price series
    close : pd.Series
        Close price series
    period : int
        CCI period (default 20)
    
    Returns:
    --------
    pd.Series : CCI values
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-9)
    
    return cci


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R.
    
    Parameters:
    -----------
    high : pd.Series
        High price series
    low : pd.Series
        Low price series
    close : pd.Series
        Close price series
    period : int
        Period (default 14)
    
    Returns:
    --------
    pd.Series : Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
    
    return williams_r


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI).
    
    Parameters:
    -----------
    high : pd.Series
        High price series
    low : pd.Series
        Low price series
    close : pd.Series
        Close price series
    volume : pd.Series
        Volume series
    period : int
        Period (default 14)
    
    Returns:
    --------
    pd.Series : MFI values (0-100)
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-9)))
    
    return mfi


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators and add to DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    
    Returns:
    --------
    pd.DataFrame : DataFrame with all indicators added
    """
    result = df.copy()
    
    # Price-based indicators
    result['rsi'] = rsi(df['close'])
    result['ema_9'] = ema(df['close'], 9)
    result['ema_21'] = ema(df['close'], 21)
    result['ema_50'] = ema(df['close'], 50)
    result['ema_200'] = ema(df['close'], 200)
    result['sma_20'] = sma(df['close'], 20)
    result['sma_50'] = sma(df['close'], 50)
    result['sma_200'] = sma(df['close'], 200)
    
    # MACD
    macd_line, signal_line, histogram = macd(df['close'])
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    result['bb_width'] = (bb_upper - bb_lower) / bb_middle
    result['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # Volume indicators
    if 'volume' in df.columns:
        result['obv'] = obv(df['close'], df['volume'])
        result['vwap'] = vwap(df)
        result['mfi'] = mfi(df['high'], df['low'], df['close'], df['volume'])
    
    # Range indicators
    result['atr'] = atr(df)
    result['atr_pct'] = result['atr'] / df['close'] * 100
    
    # Stochastic
    if 'high' in df.columns and 'low' in df.columns:
        k, d = stochastic(df)
        result['stoch_k'] = k
        result['stoch_d'] = d
    
    # ADX
    if 'high' in df.columns and 'low' in df.columns:
        adx_val, plus_di, minus_di = adx(df)
        result['adx'] = adx_val
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
    
    return result
