"""
Base Strategy
============
Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    """Signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class TradingSignal:
    """Trading signal."""
    symbol: str
    direction: SignalDirection
    confidence: float  # 0-1
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "metadata": self.metadata
        }


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement the generate_signal method.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.parameters = config.parameters
        
        # Performance tracking
        self.signals_generated = 0
        self.last_signal: Optional[TradingSignal] = None
        
        logger.info(f"Strategy initialized: {self.name}")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal from data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            TradingSignal or None if no signal
        """
        pass
    
    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_balance: float,
        risk_pct: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            signal: Trading signal
            account_balance: Account balance
            risk_pct: Risk percentage per trade
            
        Returns:
            Position size
        """
        if signal.stop_loss is None:
            return 0.0
        
        # Calculate risk amount
        risk_amount = account_balance * risk_pct
        
        # Calculate position size
        entry = signal.entry_price or 0
        stop = signal.stop_loss
        
        if entry == 0 or entry == stop:
            return 0.0
        
        price_risk = abs(entry - stop) / entry
        position_size = risk_amount / price_risk
        
        return position_size
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate signal parameters.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if valid
        """
        # Check confidence
        if signal.confidence < 0 or signal.confidence > 1:
            logger.warning(f"Invalid confidence: {signal.confidence}")
            return False
        
        # Check direction
        if signal.direction not in SignalDirection:
            logger.warning(f"Invalid direction: {signal.direction}")
            return False
        
        # Check stop loss for non-FLAT signals
        if signal.direction != SignalDirection.FLAT:
            if signal.stop_loss is None:
                logger.warning("No stop loss specified")
                return False
        
        return True
    
    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common indicators.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with indicators
        """
        df = data.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Simple moving averages
        for period in [9, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential moving averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
    
    def get_performance(self) -> Dict:
        """Get strategy performance metrics."""
        return {
            "name": self.name,
            "signals_generated": self.signals_generated,
            "enabled": self.enabled
        }
    
    def reset(self):
        """Reset strategy state."""
        self.signals_generated = 0
        self.last_signal = None
        logger.info(f"Strategy reset: {self.name}")

