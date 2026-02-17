"""
Signal Engine Module for Crypto Trading System
==============================================
Professional signal generation combining:
- Technical indicator signals
- Sentiment analysis
- Multi-timeframe analysis
- Signal filtering and confirmation

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Signal:
    """Trading signal container"""
    action: str  # BUY, SELL, HOLD
    strength: float  # Signal strength (0-1)
    confidence: float  # Confidence level (0-1)
    reasons: List[str]  # Explanation of signal
    timestamp: Optional[pd.Timestamp] = None


def generate_signal(
    price: float,
    rsi_value: float,
    sentiment: float = 0.0,
    macd_histogram: float = 0.0,
    bb_position: float = 0.5,
    trend: str = "neutral"
) -> Signal:
    """
    Generate trading signal from technical indicators and sentiment.
    
    Parameters:
    -----------
    price : float
        Current price
    rsi_value : float
        RSI value (0-100)
    sentiment : float
        Sentiment score (-1 to 1)
    macd_histogram : float
        MACD histogram value
    bb_position : float
        Bollinger Band position (0-1)
    trend : str
        Current trend direction
    
    Returns:
    --------
    Signal : Trading signal object
    """
    score = 0.0
    reasons = []
    
    # RSI signals
    if rsi_value < 25:
        score += 0.4
        reasons.append("RSI oversold (<25)")
    elif rsi_value < 35:
        score += 0.2
        reasons.append("RSI slightly oversold (<35)")
    elif rsi_value > 75:
        score -= 0.4
        reasons.append("RSI overbought (>75)")
    elif rsi_value > 65:
        score -= 0.2
        reasons.append("RSI slightly overbought (>65)")
    
    # Sentiment signals
    if sentiment > 0.3:
        score += 0.2 * sentiment
        reasons.append(f"Strong positive sentiment ({sentiment:.2f})")
    elif sentiment < -0.3:
        score += 0.2 * sentiment  # negative adds to negative
        reasons.append(f"Strong negative sentiment ({sentiment:.2f})")
    
    # MACD signals
    if macd_histogram > 0:
        score += 0.15
        reasons.append("MACD bullish divergence")
    else:
        score -= 0.15
        reasons.append("MACD bearish divergence")
    
    # Bollinger Bands signals
    if bb_position < 0.1:
        score += 0.2
        reasons.append("Price at lower Bollinger Band")
    elif bb_position > 0.9:
        score -= 0.2
        reasons.append("Price at upper Bollinger Band")
    
    # Trend alignment
    if trend == "uptrend" and score > 0:
        score *= 1.2
        reasons.append("Aligned with uptrend")
    elif trend == "downtrend" and score < 0:
        score *= 1.2
        reasons.append("Aligned with downtrend")
    
    # Determine action
    if score > 0.3:
        action = "BUY"
    elif score < -0.3:
        action = "SELL"
    else:
        action = "HOLD"
    
    # Calculate confidence
    confidence = min(abs(score) / 1.0, 1.0)
    strength = min(abs(score) / 0.6, 1.0)
    
    return Signal(
        action=action,
        strength=strength,
        confidence=confidence,
        reasons=reasons
    )


def generate_ema_crossover_signal(
    df: pd.DataFrame,
    fast_ema: str = 'ema_9',
    slow_ema: str = 'ema_21'
) -> pd.Series:
    """
    Generate signals based on EMA crossover.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with EMA columns
    fast_ema : str
        Fast EMA column name
    slow_ema : str
        Slow EMA column name
    
    Returns:
    --------
    pd.Series : Signals (BUY/SELL/HOLD)
    """
    signals = pd.Series('HOLD', index=df.index)
    
    # Bullish crossover: fast EMA crosses above slow EMA
    bullish_cross = (df[fast_ema] > df[slow_ema]) & (df[fast_ema].shift(1) <= df[slow_ema].shift(1))
    # Bearish crossover: fast EMA crosses below slow EMA
    bearish_cross = (df[fast_ema] < df[slow_ema]) & (df[fast_ema].shift(1) >= df[slow_ema].shift(1))
    
    signals[bullish_cross] = 'BUY'
    signals[bearish_cross] = 'SELL'
    
    return signals


def generate_rsi_signal(rsi: pd.Series, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """
    Generate signals based on RSI levels.
    
    Parameters:
    -----------
    rsi : pd.Series
        RSI values
    oversold : float
        Oversold threshold
    overbought : float
        Overbought threshold
    
    Returns:
    --------
    pd.Series : Signals
    """
    signals = pd.Series('HOLD', index=rsi.index)
    
    signals[rsi < oversold] = 'BUY'
    signals[rsi > overbought] = 'SELL'
    
    return signals


def generate_macd_signal(macd: pd.Series, signal: pd.Series) -> pd.Series:
    """
    Generate signals based on MACD crossovers.
    
    Parameters:
    -----------
    macd : pd.Series
        MACD line
    signal : pd.Series
        Signal line
    
    Returns:
    --------
    pd.Series : Signals
    """
    signals = pd.Series('HOLD', index=macd.index)
    
    # Bullish crossover
    bullish = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    # Bearish crossover
    bearish = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    
    signals[bullish] = 'BUY'
    signals[bearish] = 'SELL'
    
    return signals


def generate_bb_signal(
    price: pd.Series,
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    bb_middle: pd.Series
) -> pd.Series:
    """
    Generate signals based on Bollinger Bands.
    
    Parameters:
    -----------
    price : pd.Series
        Price series
    bb_upper : pd.Series
        Upper Bollinger Band
    bb_lower : pd.Series
        Lower Bollinger Band
    bb_middle : pd.Series
        Middle Bollinger Band
    
    Returns:
    --------
    pd.Series : Signals
    """
    signals = pd.Series('HOLD', index=price.index)
    
    # Price at lower band - potential bounce
    signals[price <= bb_lower] = 'BUY'
    # Price at upper band - potential reversal
    signals[price >= bb_upper] = 'SELL'
    
    return signals


def generate_composite_signal(
    df: pd.DataFrame,
    sentiment: Optional[pd.Series] = None,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Generate composite signal from multiple indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with technical indicators
    sentiment : pd.Series, optional
        Sentiment scores
    weights : Dict[str, float], optional
        Custom weights for each indicator
    
    Returns:
    --------
    pd.Series : Composite signals
    """
    if weights is None:
        weights = {
            'rsi': 0.25,
            'macd': 0.25,
            'ema': 0.20,
            'bb': 0.15,
            'sentiment': 0.15
        }
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    signals = pd.Series(0.0, index=df.index)
    
    # RSI component
    if 'rsi' in df.columns:
        rsi_signal = pd.Series(0.0, index=df.index)
        rsi_signal[df['rsi'] < 30] = 1.0
        rsi_signal[df['rsi'] > 70] = -1.0
        signals += weights.get('rsi', 0) * rsi_signal
    
    # MACD component
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_signal = pd.Series(0.0, index=df.index)
        macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        macd_signal[macd_bullish] = 1.0
        macd_signal[macd_bearish] = -1.0
        signals += weights.get('macd', 0) * macd_signal
    
    # EMA crossover component
    if 'ema_9' in df.columns and 'ema_21' in df.columns:
        ema_signal = pd.Series(0.0, index=df.index)
        ema_bullish = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
        ema_bearish = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
        ema_signal[ema_bullish] = 1.0
        ema_signal[ema_bearish] = -1.0
        signals += weights.get('ema', 0) * ema_signal
    
    # Bollinger Bands component
    if 'bb_position' in df.columns:
        bb_signal = pd.Series(0.0, index=df.index)
        bb_signal[df['bb_position'] < 0.1] = 1.0
        bb_signal[df['bb_position'] > 0.9] = -1.0
        signals += weights.get('bb', 0) * bb_signal
    
    # Sentiment component
    if sentiment is not None:
        signals += weights.get('sentiment', 0) * sentiment
    
    # Convert to signals
    result = pd.Series('HOLD', index=df.index)
    result[signals > 0.3] = 'BUY'
    result[signals < -0.3] = 'SELL'
    
    return result


def detect_trend(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Detect trend direction using price vs SMA.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price and SMA
    period : int
        Period for trend detection
    
    Returns:
    --------
    pd.Series : Trend direction (uptrend/downtrend/neutral)
    """
    sma_col = f'sma_{period}' if f'sma_{period}' in df.columns else 'sma_50'
    
    if sma_col not in df.columns:
        # Use EMA if SMA not available
        ema_col = f'ema_{period}' if f'ema_{period}' in df.columns else 'ema_50'
        if ema_col not in df.columns:
            return pd.Series('neutral', index=df.index)
        trend = pd.Series('neutral', index=df.index)
        trend[df['close'] > df[ema_col]] = 'uptrend'
        trend[df['close'] < df[ema_col]] = 'downtrend'
    else:
        trend = pd.Series('neutral', index=df.index)
        trend[df['close'] > df[sma_col]] = 'uptrend'
        trend[df['close'] < df[sma_col]] = 'downtrend'
    
    return trend


def filter_signals_by_volatility(
    signals: pd.Series,
    atr: pd.Series,
    atr_threshold: float = 1.5
) -> pd.Series:
    """
    Filter signals based on volatility (ATR).
    
    Parameters:
    -----------
    signals : pd.Series
        Trading signals
    atr : pd.Series
        Average True Range
    atr_threshold : float
        Minimum ATR threshold
    
    Returns:
    --------
    pd.Series : Filtered signals
    """
    filtered = signals.copy()
    filtered[atr < atr_threshold] = 'HOLD'
    
    return filtered


def multi_timeframe_signal(
    daily_signal: pd.Series,
    weekly_signal: pd.Series,
    alignment_weight: float = 0.3
) -> pd.Series:
    """
    Combine signals from multiple timeframes.
    
    Parameters:
    -----------
    daily_signal : pd.Series
        Daily timeframe signals
    weekly_signal : pd.Series
        Weekly timeframe signals
    alignment_weight : float
        Weight for alignment bonus
    
    Returns:
    --------
    pd.Series : Combined signals
    """
    # Convert to numeric for calculation
    signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
    daily_num = daily_signal.map(signal_map).fillna(0)
    weekly_num = weekly_signal.map(signal_map).fillna(0)
    
    # Combine signals
    combined = daily_num + alignment_weight * weekly_num
    
    # Convert back to signals
    result = pd.Series('HOLD', index=daily_signal.index)
    result[combined > 0.5] = 'BUY'
    result[combined < -0.5] = 'SELL'
    
    return result
