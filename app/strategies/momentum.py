"""
Momentum Strategy
================
Trend-following strategy based on momentum indicators.
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


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Generates signals based on:
    - RSI overbought/oversold
    - MACD crossover
    - Price momentum
    
    Entry:
    - RSI < oversold_threshold (30) -> LONG
    - RSI > overbought_threshold (70) -> SHORT
    
    Exit:
    - RSI returns to neutral (50)
    - Stop loss or take profit hit
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize momentum strategy."""
        super().__init__(config)
        
        # Parameters
        self.lookback_period = self.parameters.get("lookback_period", 20)
        self.threshold = self.parameters.get("threshold", 0.02)
        self.rsi_oversold = self.parameters.get("rsi_oversold", 30)
        self.rsi_overbought = self.parameters.get("rsi_overbought", 70)
        self.stop_loss_pct = self.parameters.get("stop_loss_pct", 0.02)
        self.take_profit_pct = self.parameters.get("take_profit_pct", 0.05)
        
        # State
        self.position = SignalDirection.FLAT
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate momentum-based trading signal.
        
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
        previous = df.iloc[-2]
        
        # Current values
        rsi = latest['rsi']
        macd = latest['macd']
        macd_signal = latest['macd_signal']
        macd_hist = latest['macd_hist']
        prev_macd_hist = previous['macd_hist']
        close = latest['close']
        sma_20 = latest['sma_20']
        sma_50 = latest['sma_50']
        
        # Skip if RSI not available
        if pd.isna(rsi) or pd.isna(macd):
            return None
        
        # Momentum signal logic
        signal = None
        confidence = 0.5
        
        # Long signal: RSI oversold + MACD bullish crossover
        if (rsi < self.rsi_oversold and 
            prev_macd_hist < 0 and macd_hist > 0):
            
            direction = SignalDirection.LONG
            confidence = min(0.95, (30 - rsi) / 30 + 0.5)
            
            entry_price = close
            stop_loss = close * (1 - self.stop_loss_pct)
            take_profit = close * (1 + self.take_profit_pct)
            
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                metadata={
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'trend': 'bullish' if sma_20 > sma_50 else 'neutral'
                }
            )
        
        # Short signal: RSI overbought + MACD bearish crossover
        elif (rsi > self.rsi_overbought and 
              prev_macd_hist > 0 and macd_hist < 0):
            
            direction = SignalDirection.SHORT
            confidence = min(0.95, (rsi - 70) / 30 + 0.5)
            
            entry_price = close
            stop_loss = close * (1 + self.stop_loss_pct)
            take_profit = close * (1 - self.take_profit_pct)
            
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                metadata={
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'trend': 'bearish' if sma_20 < sma_50 else 'neutral'
                }
            )
        
        # Exit signal: RSI returns to neutral
        elif (self.position == SignalDirection.LONG and rsi > 50):
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=SignalDirection.FLAT,
                confidence=0.7,
                entry_price=close,
                strategy_name=self.name,
                metadata={'reason': 'rsi_neutral_long_exit'}
            )
            self.position = SignalDirection.FLAT
            
        elif (self.position == SignalDirection.SHORT and rsi < 50):
            signal = TradingSignal(
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                direction=SignalDirection.FLAT,
                confidence=0.7,
                entry_price=close,
                strategy_name=self.name,
                metadata={'reason': 'rsi_neutral_short_exit'}
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
            "threshold": self.threshold,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct
        }

