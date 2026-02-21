"""
Technical Analysis Module
Provides technical indicators and analysis for crypto/commodity markets
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

import pandas as pd
import numpy as np

import config

# Configure logging
logger = logging.getLogger(__name__)


# ==================== DATA STRUCTURES ====================

@dataclass
class IndicatorResult:
    """Result of a technical indicator calculation"""
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'neutral'
    description: str = ""


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis for an asset"""
    symbol: str
    timestamp: datetime
    current_price: float
    
    # Trend indicators
    ema_short: float = 0.0
    ema_medium: float = 0.0
    ema_long: float = 0.0
    trend: str = "neutral"  # 'bullish', 'bearish', 'neutral'
    
    # Momentum indicators
    rsi: float = 50.0
    rsi_signal: str = "neutral"
    
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_signal_line: str = "neutral"
    
    # Volatility indicators
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5  # 0 = at lower band, 1 = at upper band
    bb_signal: str = "neutral"
    
    atr: float = 0.0
    
    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_signal: str = "neutral"
    
    # Overall scores
    momentum_score: float = 0.5  # 0-1
    technical_score: float = 0.5  # 0-1
    
    # Raw indicator values
    indicators: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'trend': self.trend,
            'rsi': self.rsi,
            'rsi_signal': self.rsi_signal,
            'macd': self.macd,
            'macd_histogram': self.macd_histogram,
            'bb_position': self.bb_position,
            'technical_score': self.technical_score,
            'momentum_score': self.momentum_score,
        }


# ==================== TECHNICAL ANALYSIS CLASS ====================

class TechnicalAnalyzer:
    """
    Calculate technical indicators for financial data.
    Includes RSI, EMA, Bollinger Bands, MACD, Stochastic, ATR
    """
    
    def __init__(self):
        """Initialize the technical analyzer"""
        self.settings = config.INDICATOR_SETTINGS
        logger.info("TechnicalAnalyzer initialized")
    
    # ==================== MAIN ANALYSIS METHOD ====================
    
    def analyze(self, df: pd.DataFrame, symbol: str = "") -> TechnicalAnalysis:
        """
        Perform complete technical analysis on price data.
        
        Args:
            df: DataFrame with OHLCV data (must have: open, high, low, close, volume)
            symbol: Trading symbol for reference
            
        Returns:
            TechnicalAnalysis object with all indicators
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for analysis")
            return TechnicalAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=0.0
            )
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Got: {df.columns.tolist()}")
            return TechnicalAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=0.0
            )
        
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0.0
        
        analysis = TechnicalAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price
        )
        
        try:
            # Calculate all indicators
            analysis.ema_short = self.calculate_ema(df, self.settings['ema_short'])
            analysis.ema_medium = self.calculate_ema(df, self.settings['ema_medium'])
            analysis.ema_long = self.calculate_ema(df, self.settings['ema_long'])
            
            analysis.rsi = self.calculate_rsi(df, self.settings['rsi_period'])
            
            macd_result = self.calculate_macd(df)
            analysis.macd = macd_result['macd']
            analysis.macd_signal = macd_result['signal']
            analysis.macd_histogram = macd_result['histogram']
            
            bb_result = self.calculate_bollinger_bands(df)
            analysis.bb_upper = bb_result['upper']
            analysis.bb_middle = bb_result['middle']
            analysis.bb_lower = bb_result['lower']
            analysis.bb_position = bb_result['position']
            
            analysis.atr = self.calculate_atr(df, self.settings['atr_period'])
            
            stoch_result = self.calculate_stochastic(df)
            analysis.stoch_k = stoch_result['k']
            analysis.stoch_d = stoch_result['d']
            
            # Generate signals
            analysis.rsi_signal = self._rsi_signal(analysis.rsi)
            analysis.bb_signal = self._bb_signal(analysis.bb_position)
            analysis.macd_signal_line = self._macd_signal(analysis.macd_histogram)
            analysis.stoch_signal = self._stoch_signal(analysis.stoch_k, analysis.stoch_d)
            
            # Determine trend
            analysis.trend = self._determine_trend(
                analysis.ema_short,
                analysis.ema_medium,
                analysis.ema_long
            )
            
            # Calculate overall scores
            analysis.momentum_score = self._calculate_momentum_score(analysis)
            analysis.technical_score = self._calculate_technical_score(analysis)
            
            # Store raw indicators
            analysis.indicators = {
                'ema_short': analysis.ema_short,
                'ema_medium': analysis.ema_medium,
                'ema_long': analysis.ema_long,
                'rsi': analysis.rsi,
                'macd': analysis.macd,
                'macd_histogram': analysis.macd_histogram,
                'bb_upper': analysis.bb_upper,
                'bb_middle': analysis.bb_middle,
                'bb_lower': analysis.bb_lower,
                'bb_position': analysis.bb_position,
                'atr': analysis.atr,
                'stoch_k': analysis.stoch_k,
                'stoch_d': analysis.stoch_d,
            }
            
            analysis.signals = {
                'rsi': analysis.rsi_signal,
                'bb': analysis.bb_signal,
                'macd': analysis.macd_signal_line,
                'stoch': analysis.stoch_signal,
                'trend': analysis.trend,
            }
            
        except Exception as e:
            logger.error(f"Error during technical analysis: {e}")
        
        return analysis
    
    # ==================== INDICATOR CALCULATIONS ====================
    
    def calculate_sma(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(df) < period:
            return df['close'].mean() if len(df) > 0 else 0.0
        return df['close'].iloc[-period:].mean()
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(df) < period:
            return df['close'].mean() if len(df) > 0 else 0.0
        return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI ranges from 0-100:
        - Above 70: Overbought
        - Below 30: Oversold
        - Around 50: Neutral
        """
        if len(df) < period + 1:
            return 50.0
        
        delta = df['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Handle edge case where avg_loss is 0
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Returns:
            Dictionary with 'macd', 'signal', 'histogram'
        """
        fast = self.settings['macd_fast']
        slow = self.settings['macd_slow']
        signal_period = self.settings['macd_signal']
        
        if len(df) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        # Calculate MACD line
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Bollinger Bands.
        
        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'position'
        """
        period = self.settings['bb_period']
        std_dev = self.settings['bb_std']
        
        if len(df) < period:
            middle = df['close'].mean() if len(df) > 0 else 0.0
            return {
                'upper': middle * 1.02,
                'middle': middle,
                'lower': middle * 0.98,
                'position': 0.5
            }
        
        # Calculate middle band (SMA)
        middle = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df['close'].rolling(window=period).std()
        
        # Calculate bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Calculate position (0 = at lower band, 1 = at upper band)
        current_price = df['close'].iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        if current_upper == current_lower:
            position = 0.5
        else:
            position = (current_price - current_lower) / (current_upper - current_lower)
        
        return {
            'upper': upper.iloc[-1],
            'middle': middle.iloc[-1],
            'lower': lower.iloc[-1],
            'position': position
        }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) - Volatility indicator.
        """
        if len(df) < period:
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0
    
    def calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Stochastic Oscillator.
        
        Returns:
            Dictionary with 'k' (%K) and 'd' (%D)
        """
        period = self.settings['stoch_period']
        smooth = self.settings['stoch_smooth']
        
        if len(df) < period:
            return {'k': 50.0, 'd': 50.0}
        
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        # Calculate %K
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        # Smooth %K to get %D
        stoch_d = stoch_k.rolling(window=smooth).mean()
        
        return {
            'k': stoch_k.iloc[-1],
            'd': stoch_d.iloc[-1]
        }
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate historical volatility (standard deviation of returns).
        """
        if len(df) < period:
            return 0.0
        
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(365)  # Annualized
        
        return volatility.iloc[-1] if not np.isnan(volatility.iloc[-1]) else 0.0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them as columns to the DataFrame.
        This method is used for ML model training.
        
        Args:
            df: DataFrame with OHLCV data (must have: open, high, low, close, volume)
            
        Returns:
            DataFrame with added indicator columns
        """
        if df is None or df.empty:
            return df
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for calculate_indicators. Got: {df.columns.tolist()}")
            return df
        
        result_df = df.copy()
        
        try:
            # EMA indicators
            result_df['ema_short'] = df['close'].ewm(span=self.settings['ema_short'], adjust=False).mean()
            result_df['ema_medium'] = df['close'].ewm(span=self.settings['ema_medium'], adjust=False).mean()
            result_df['ema_long'] = df['close'].ewm(span=self.settings['ema_long'], adjust=False).mean()
            
            # RSI
            delta = df['close'].diff()
            gains = delta.where(delta > 0, 0)
            losses = (-delta).where(delta < 0, 0)
            avg_gain = gains.rolling(window=self.settings['rsi_period']).mean()
            avg_loss = losses.rolling(window=self.settings['rsi_period']).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            result_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            fast = self.settings['macd_fast']
            slow = self.settings['macd_slow']
            signal_period = self.settings['macd_signal']
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            result_df['macd'] = macd_line
            result_df['macd_signal'] = signal_line
            result_df['macd_histogram'] = macd_line - signal_line
            
            # Bollinger Bands
            period_bb = self.settings['bb_period']
            std_dev = self.settings['bb_std']
            middle = df['close'].rolling(window=period_bb).mean()
            std = df['close'].rolling(window=period_bb).std()
            result_df['bb_upper'] = middle + (std * std_dev)
            result_df['bb_middle'] = middle
            result_df['bb_lower'] = middle - (std * std_dev)
            
            # Calculate BB position
            current_upper = result_df['bb_upper']
            current_lower = result_df['bb_lower']
            result_df['bb_position'] = (df['close'] - current_lower) / (current_upper - current_lower).replace(0, np.nan)
            
            # ATR
            high = df['high']
            low = df['low']
            close = df['close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result_df['atr'] = tr.rolling(window=self.settings['atr_period']).mean()
            
            # Stochastic
            period_stoch = self.settings['stoch_period']
            smooth = self.settings['stoch_smooth']
            low_min = df['low'].rolling(window=period_stoch).min()
            high_max = df['high'].rolling(window=period_stoch).max()
            stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
            result_df['stoch_k'] = stoch_k
            result_df['stoch_d'] = stoch_k.rolling(window=smooth).mean()
            
            # Additional indicators for ML
            result_df['returns'] = df['close'].pct_change()
            result_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            result_df['high_low_ratio'] = df['high'] / df['low']
            result_df['close_open_ratio'] = df['close'] / df['open']
            result_df['volume_price_ratio'] = df['volume'] / df['close']
            
            # Fill NaN values
            result_df = result_df.fillna(0)
            
            logger.debug(f"Calculated indicators for {len(result_df)} rows")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return result_df
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> float:
        """
        Calculate momentum (rate of change).
        """
        if len(df) < period:
            return 0.0
        
        current = df['close'].iloc[-1]
        past = df['close'].iloc[-period]
        
        if past == 0:
            return 0.0
        
        return ((current - past) / past) * 100
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """
        Calculate Average Directional Index (ADX) - Trend strength indicator.
        
        Returns:
            Dictionary with 'adx', 'plus_di', 'minus_di'
        """
        if len(df) < period + 1:
            return {'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0}
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di_smooth = plus_dm.rolling(window=period).mean()
        minus_di_smooth = minus_dm.rolling(window=period).mean()
        
        # Calculate DI
        plus_di = 100 * (plus_di_smooth / atr)
        minus_di = 100 * (minus_di_smooth / atr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0,
            'plus_di': plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0.0,
            'minus_di': minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0.0,
        }
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """
        Calculate Volume Weighted Average Price (VWAP).
        """
        if len(df) < 1 or 'volume' not in df.columns:
            return df['close'].mean() if len(df) > 0 else 0.0
        
        # Typical Price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # VWAP
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        return vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else 0.0
    
    def calculate_obv(self, df: pd.DataFrame) -> float:
        """
        Calculate On-Balance Volume (OBV).
        """
        if 'volume' not in df.columns or len(df) < 2:
            return 0.0
        
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        return obv.iloc[-1] if not pd.isna(obv.iloc[-1]) else 0.0
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate Commodity Channel Index (CCI).
        """
        if len(df) < period:
            return 0.0
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma) / (0.015 * mad)
        return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0.0
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Williams %R.
        """
        if len(df) < period:
            return -50.0
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50.0
    
    def calculate_roc(self, df: pd.DataFrame, period: int = 12) -> float:
        """
        Calculate Rate of Change (ROC).
        """
        if len(df) < period:
            return 0.0
        
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return roc.iloc[-1] if not pd.isna(roc.iloc[-1]) else 0.0
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Ichimoku Cloud components.
        
        Returns: Dictionary with tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        """
        if len(df) < 52:
            return {
                'tenkan_sen': 0.0, 'kijun_sen': 0.0,
                'senkou_span_a': 0.0, 'senkou_span_b': 0.0, 'chikou_span': 0.0
            }
        
        # Tenkan-sen (Conversion Line)
        high9 = df['high'].rolling(window=9).max()
        low9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high9 + low9) / 2
        
        # Kijun-sen (Base Line)
        high26 = df['high'].rolling(window=26).max()
        low26 = df['low'].rolling(window=26).min()
        kijun_sen = (high26 + low26) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high52 = df['high'].rolling(window=52).max()
        low52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high52 + low52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = df['close'].shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen.iloc[-1] if not pd.isna(tenkan_sen.iloc[-1]) else 0.0,
            'kijun_sen': kijun_sen.iloc[-1] if not pd.isna(kijun_sen.iloc[-1]) else 0.0,
            'senkou_span_a': senkou_span_a.iloc[-1] if not pd.isna(senkou_span_a.iloc[-1]) else 0.0,
            'senkou_span_b': senkou_span_b.iloc[-1] if not pd.isna(senkou_span_b.iloc[-1]) else 0.0,
            'chikou_span': chikou_span.iloc[-1] if not pd.isna(chikou_span.iloc[-1]) else 0.0
        }
    
    def calculate_fibonacci_retracement(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Fibonacci Retracement levels.
        """
        if len(df) < 20:
            return {'level_0': 0.0, 'level_236': 0.0, 'level_382': 0.0, 
                    'level_500': 0.0, 'level_618': 0.0, 'level_786': 0.0, 'level_100': 0.0}
        
        # Use last 20 candles for swing
        high = df['high'].iloc[-20:].max()
        low = df['low'].iloc[-20:].min()
        diff = high - low
        
        levels = {
            'level_0': low,
            'level_236': low + diff * 0.236,
            'level_382': low + diff * 0.382,
            'level_500': low + diff * 0.500,
            'level_618': low + diff * 0.618,
            'level_786': low + diff * 0.786,
            'level_100': high
        }
        
        return levels
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Pivot Points and Support/Resistance levels.
        """
        if len(df) < 2:
            return {'pivot': 0.0, 's1': 0.0, 's2': 0.0, 'r1': 0.0, 'r2': 0.0}
        
        # Use previous candle for pivot
        prev = df.iloc[-2]
        pivot = (prev['high'] + prev['low'] + prev['close']) / 3
        
        r1 = 2 * pivot - prev['low']
        s1 = 2 * pivot - prev['high']
        r2 = pivot + (prev['high'] - prev['low'])
        s2 = pivot - (prev['high'] - prev['low'])
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2,
            's1': s1, 's2': s2
        }
    
    def calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candles.
        """
        if len(df) < 1:
            return df
        
        ha = df.copy()
        ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha['ha_open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        
        for i in range(1, len(ha)):
            ha.iloc[i, ha.columns.get_loc('ha_open')] = (ha.iloc[i-1]['ha_open'] + ha.iloc[i-1]['ha_close']) / 2
        
        ha['ha_high'] = ha[['ha_open', 'ha_close', 'high']].max(axis=1)
        ha['ha_low'] = ha[['ha_open', 'ha_close', 'low']].min(axis=1)
        
        return ha
    
    # ==================== SIGNAL GENERATION ====================
    
    def _rsi_signal(self, rsi: float) -> str:
        """Generate signal from RSI value"""
        if rsi > self.settings['rsi_overbought']:
            return 'sell'
        elif rsi < self.settings['rsi_oversold']:
            return 'buy'
        return 'neutral'
    
    def _bb_signal(self, position: float) -> str:
        """Generate signal from Bollinger Band position"""
        if position > 0.9:
            return 'sell'  # Near upper band - overbought
        elif position < 0.1:
            return 'buy'   # Near lower band - oversold
        return 'neutral'
    
    def _macd_signal(self, histogram: float) -> str:
        """Generate signal from MACD histogram"""
        if histogram > 0:
            return 'buy'
        elif histogram < 0:
            return 'sell'
        return 'neutral'
    
    def _stoch_signal(self, k: float, d: float) -> str:
        """Generate signal from Stochastic"""
        if k > 80 and d > 80:
            return 'sell'
        elif k < 20 and d < 20:
            return 'buy'
        # Crossover signals
        elif k > d and k < 50:
            return 'buy'
        elif k < d and k > 50:
            return 'sell'
        return 'neutral'
    
    def _determine_trend(self, ema_short: float, ema_medium: float, 
                        ema_long: float) -> str:
        """Determine trend from EMA values"""
        if ema_short > ema_medium > ema_long:
            return 'bullish'
        elif ema_short < ema_medium < ema_long:
            return 'bearish'
        return 'neutral'
    
    # ==================== SCORING ====================
    
    def _calculate_momentum_score(self, analysis: TechnicalAnalysis) -> float:
        """Calculate momentum score (0-1)"""
        score = 0.5
        
        # RSI contribution (0.25 weight)
        if analysis.rsi < 30:
            score += 0.125  # Oversold - potential buy
        elif analysis.rsi > 70:
            score -= 0.125  # Overbought - potential sell
        elif analysis.rsi < 40:
            score += 0.0625
        elif analysis.rsi > 60:
            score -= 0.0625
        
        # MACD contribution (0.25 weight)
        if analysis.macd_histogram > 0:
            score += 0.125
        elif analysis.macd_histogram < 0:
            score -= 0.125
        
        # Stochastic contribution (0.25 weight)
        if analysis.stoch_k < 20:
            score += 0.125
        elif analysis.stoch_k > 80:
            score -= 0.125
        elif analysis.stoch_k < 30:
            score += 0.0625
        elif analysis.stoch_k > 70:
            score -= 0.0625
        
        # Momentum indicator (0.25 weight)
        momentum = (analysis.current_price - analysis.ema_short) / analysis.ema_short * 100
        if momentum > 2:
            score += 0.125
        elif momentum < -2:
            score -= 0.125
        
        return max(0, min(1, score))
    
    def _calculate_technical_score(self, analysis: TechnicalAnalysis) -> float:
        """Calculate overall technical score (0-1)"""
        score = 0.5
        
        # Trend contribution (0.30 weight)
        if analysis.trend == 'bullish':
            score += 0.15
        elif analysis.trend == 'bearish':
            score -= 0.15
        
        # RSI contribution (0.25 weight)
        if analysis.rsi_signal == 'buy':
            score += 0.125
        elif analysis.rsi_signal == 'sell':
            score -= 0.125
        
        # MACD contribution (0.20 weight)
        if analysis.macd_signal_line == 'buy':
            score += 0.10
        elif analysis.macd_signal_line == 'sell':
            score -= 0.10
        
        # Bollinger Bands contribution (0.15 weight)
        if analysis.bb_signal == 'buy':
            score += 0.075
        elif analysis.bb_signal == 'sell':
            score -= 0.075
        
        # Stochastic contribution (0.10 weight)
        if analysis.stoch_signal == 'buy':
            score += 0.05
        elif analysis.stoch_signal == 'sell':
            score -= 0.05
        
        return max(0, min(1, score))
    
    # ==================== COMPARATIVE ANALYSIS ====================
    
    def compare_assets(self, analyses: List[TechnicalAnalysis]) -> List[Tuple[str, float, str]]:
        """
        Compare multiple assets and rank by technical score.
        
        Args:
            analyses: List of TechnicalAnalysis objects
            
        Returns:
            List of (symbol, score, recommendation) tuples
        """
        rankings = []
        
        for analysis in analyses:
            score = analysis.technical_score
            
            # Determine recommendation
            if score > 0.7:
                recommendation = 'STRONG_BUY'
            elif score > 0.6:
                recommendation = 'BUY'
            elif score < 0.3:
                recommendation = 'STRONG_SELL'
            elif score < 0.4:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            rankings.append((analysis.symbol, score, recommendation))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    # ==================== UTILITY METHODS ====================
    
    def get_indicator_description(self, indicator: str) -> str:
        """Get description of an indicator"""
        descriptions = {
            'rsi': 'Relative Strength Index - Momentum oscillator (0-100)',
            'macd': 'Moving Average Convergence Divergence - Trend indicator',
            'bb': 'Bollinger Bands - Volatility bands around SMA',
            'ema': 'Exponential Moving Average - Weighted moving average',
            'atr': 'Average True Range - Volatility measure',
            'stoch': 'Stochastic Oscillator - Momentum indicator'
        }
        return descriptions.get(indicator.lower(), 'Unknown indicator')
    
    def export_analysis(self, analysis: TechnicalAnalysis, filepath: str):
        """Export analysis to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2, default=str)
        logger.info(f"Analysis exported to {filepath}")


# ==================== STANDALONE FUNCTIONS ====================

def analyze_crypto(df: pd.DataFrame, symbol: str) -> TechnicalAnalysis:
    """Convenience function for quick analysis"""
    analyzer = TechnicalAnalyzer()
    return analyzer.analyze(df, symbol)


if __name__ == "__main__":
    import random
    from datetime import timedelta
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    print("\n" + "="*60)
    print("TECHNICAL ANALYZER TEST")
    print("="*60)
    
    # Generate simulated OHLCV data
    base_price = 50000.0
    data = []
    
    for i in range(100):
        timestamp = datetime.now() - timedelta(hours=100-i)
        change = random.gauss(0, 0.02)
        base_price *= (1 + change)
        
        high = base_price * random.uniform(1.001, 1.02)
        low = base_price * random.uniform(0.98, 0.999)
        open_price = random.uniform(low, high)
        close = random.uniform(low, high)
        volume = random.uniform(1000, 50000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Run analysis
    analyzer = TechnicalAnalyzer()
    result = analyzer.analyze(df, 'BTC/USDT')
    
    print(f"\n📊 Technical Analysis for {result.symbol}")
    print(f"   Price: ${result.current_price:,.2f}")
    print(f"   Trend: {result.trend}")
    print(f"   RSI: {result.rsi:.2f} ({result.rsi_signal})")
    print(f"   MACD: {result.macd:.4f} (histogram: {result.macd_histogram:.4f})")
    print(f"   Bollinger Position: {result.bb_position:.2%}")
    print(f"   Technical Score: {result.technical_score:.2%}")
    print(f"   Momentum Score: {result.momentum_score:.2%}")
    
    print("\n✅ Test complete!")

