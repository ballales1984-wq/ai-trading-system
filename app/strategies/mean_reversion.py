"""
Mean Reversion Strategy
====================
Mean reversion trading strategy based on Bollinger Bands.
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd

from app.strategies.base_strategy import (
    BaseStrategy, StrategyConfig, TradingSignal, 
    SignalDirection
)


logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    Generates signals based on:
    - Bollinger Bands
    - Z-score
    - RSI
    
    Entry:
    - Price touches lower BB -> LONG
    - Price touches upper BB -> SHORT
    
    Exit:
    - Price returns to middle BB
    - Stop loss or take profit hit
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize mean reversion strategy."""
        super().__init__(config)
        
        # Parameters
        self.lookback_period = self.parameters.get("lookback_period", 50)
        self.bb_std = self.parameters.get("bb_std", 2.0)  # Standard deviations
        self.zscore_threshold = self.parameters.get("zscore_threshold", 2.0)
        self.stop_loss_pct = self.parameters.get("stop_loss_pct", 0.03)
        self.take_profit_pct = self.parameters.get("take_profit_pct", 0.02)
        
        # State
        self.position = SignalDirection.FLAT
    
    def calculate_zscore(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate z-score."""
        mean = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        zscore = (data - mean) / std
        return zscore
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate mean reversion trading signal.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            TradingSignal or None
        """
        if len(data) < self.lookback_period + 1:
            return None
        
        # Get indicators
        df = self.get_indicators(data)
        
        # Get latest values
        latest = df.iloc[-1]
        close = latest['close']
        
        # Bollinger Bands
        bb_upper = latest['bb_upper']
        bb_middle = latest['bb_middle']
        bb_lower = latest['bb_lower']
        
        # Skip if BB not available
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            return None
        
        # Calculate z-score
        df['zscore'] = self.calculate_zscore(df['close'], self.lookback_period)
        zscore = df['zscore'].iloc[-1]
        
        if pd.isna(zscore):
            return None
        
        # Mean reversion signal logic
        signal = None
        confidence = 0.5
        
        # Long signal: Price at lower BB or oversold z-score
        if (close <= bb_lower or zscore < -self.zscore_threshold):
            
            direction = SignalDirection.LONG
            
            # Higher confidence for extreme z-score
            if zscore < -self.zscore_threshold:
                confidence = min(0.95, abs(zscore) / self.zscore_threshold * 0.7 + 0.3)
            else:
                confidence = 0.65
            
            entry_price = close
            stop_loss = close * (1 - self.stop_loss_pct)
            take_profit = bb_middle  # Target: middle BB
            
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                metadata={
                    'zscore': zscore,
                    'bb_position': 'lower' if close <= bb_lower else 'normal',
                    'distance_from_mean': (close - bb_middle) / bb_middle
                }
            )
        
        # Short signal: Price at upper BB or overbought z-score
        elif (close >= bb_upper or zscore > self.zscore_threshold):
            
            direction = SignalDirection.SHORT
            
            # Higher confidence for extreme z-score
            if zscore > self.zscore_threshold:
                confidence = min(0.95, abs(zscore) / self.zscore_threshold * 0.7 + 0.3)
            else:
                confidence = 0.65
            
            entry_price = close
            stop_loss = close * (1 + self.stop_loss_pct)
            take_profit = bb_middle  # Target: middle BB
            
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                metadata={
                    'zscore': zscore,
                    'bb_position': 'upper' if close >= bb_upper else 'normal',
                    'distance_from_mean': (close - bb_middle) / bb_middle
                }
            )
        
        # Exit signal: Price returns to mean
        elif (self.position == SignalDirection.LONG and 
              close >= bb_middle):
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=SignalDirection.FLAT,
                confidence=0.7,
                entry_price=close,
                strategy_name=self.name,
                metadata={'reason': 'mean_reversion_complete'}
            )
            self.position = SignalDirection.FLAT
            
        elif (self.position == SignalDirection.SHORT and 
              close <= bb_middle):
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=SignalDirection.FLAT,
                confidence=0.7,
                entry_price=close,
                strategy_name=self.name,
                metadata={'reason': 'mean_reversion_complete'}
            )
            self.position = SignalDirection.FLAT
        
        if signal:
            self.signals_generated += 1
            self.last_signal = signal
            
            if signal.direction != SignalDirection.FLAT:
                self.position = signal.direction
        
        return signal
    
    def get_parameters(self) -> dict:
        """Get strategy parameters."""
        return {
            "lookback_period": self.lookback_period,
            "bb_std": self.bb_std,
            "zscore_threshold": self.zscore_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct
        }

