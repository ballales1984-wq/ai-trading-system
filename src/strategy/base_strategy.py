# src/strategy/base_strategy.py
"""
Base Strategy Module
====================
Abstract base class for all trading strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class TradingSignal:
    """Trading signal container."""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    price: float
    timestamp: datetime
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Backward compatibility: support both 'action' and 'signal_type'
    def __post_init__(self):
        """Handle backward compatibility for action parameter."""
        # If signal_type is None but we have action metadata, use that
        if self.signal_type is None and 'action' in self.metadata:
            self.signal_type = self.metadata['action']
    
    @property
    def action(self) -> SignalType:
        """Backward compatible action property."""
        return self.signal_type
    
    @action.setter
    def action(self, value: SignalType):
        """Backward compatible action setter."""
        self.signal_type = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value if self.signal_type else None,
            "action": self.signal_type.value if self.signal_type else None,  # Backward compatibility
            "strength": self.strength.value,
            "confidence": self.confidence,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "metadata": self.metadata,
        }


@dataclass
class StrategyContext:
    """
    Context data passed to strategies for signal generation.
    
    Contains all relevant market data, indicators, and state.
    """
    symbol: str
    prices: np.ndarray
    volumes: np.ndarray
    timestamps: List[datetime]
    
    # Technical indicators
    indicators: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Monte Carlo data
    mc_probability_up: Optional[float] = None
    mc_var: Optional[float] = None
    mc_cvar: Optional[float] = None
    
    # Risk metrics
    risk_level: Optional[str] = None
    volatility: Optional[float] = None
    
    # Sentiment
    sentiment: Optional[float] = None
    
    # Position info
    current_position: Optional[Dict] = None
    
    # Additional data
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides:
    - Signal generation interface
    - Configuration management
    - Performance tracking
    - Risk integration
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize strategy.
        
        Args:
            name: Strategy identifier
            config: Strategy configuration dictionary
        """
        self.name = name
        self.config = config
        
        # Parameters
        self._params = config.get("params", {})
        
        # Risk limits
        self._max_position_size = config.get("max_position_size", 1.0)
        self._max_risk_per_trade = config.get("max_risk_per_trade", 0.02)
        self._min_confidence = config.get("min_confidence", 0.5)
        
        # Performance tracking
        self._signals_generated = 0
        self._signals_executed = 0
        self._signals_profitable = 0
        
        # State
        self._enabled = True
        self._last_signal: Optional[TradingSignal] = None
        
        logger.info(f"Strategy '{name}' initialized")
    
    @property
    def enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self._enabled
    
    def enable(self):
        """Enable the strategy."""
        self._enabled = True
        logger.info(f"Strategy '{self.name}' enabled")
    
    def disable(self):
        """Disable the strategy."""
        self._enabled = False
        logger.info(f"Strategy '{self.name}' disabled")
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get strategy parameter."""
        return self._params.get(key, default)
    
    def set_param(self, key: str, value: Any):
        """Set strategy parameter."""
        self._params[key] = value
        logger.debug(f"Strategy '{self.name}' param {key} = {value}")
    
    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> Optional[TradingSignal]:
        """
        Generate trading signal based on context.
        
        Args:
            context: Strategy context with market data and indicators
            
        Returns:
            TradingSignal or None if no signal
        """
        pass
    
    def validate_signal(self, signal: TradingSignal, context: StrategyContext) -> bool:
        """
        Validate a signal before execution.
        
        Args:
            signal: Signal to validate
            context: Strategy context
            
        Returns:
            True if signal is valid
        """
        # Check confidence threshold
        if signal.confidence < self._min_confidence:
            logger.debug(
                f"Signal rejected: confidence {signal.confidence} < "
                f"{self._min_confidence}"
            )
            return False
        
        # Check risk level
        if context.risk_level == "critical":
            logger.debug("Signal rejected: critical risk level")
            return False
        
        # Check position size
        if context.current_position:
            position_size = context.current_position.get("size", 0)
            if abs(position_size) > self._max_position_size:
                logger.debug("Signal rejected: position size limit reached")
                return False
        
        return True
    
    def classify_strength(self, confidence: float) -> SignalStrength:
        """
        Classify signal strength based on confidence.
        
        Args:
            confidence: Signal confidence (0-1)
            
        Returns:
            SignalStrength enum value
        """
        if confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.6:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def create_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        confidence: float,
        price: float,
        metadata: Dict = None
    ) -> TradingSignal:
        """
        Create a trading signal.
        
        Args:
            symbol: Trading pair
            signal_type: Type of signal
            confidence: Signal confidence (0-1)
            price: Current price
            metadata: Additional metadata
            
        Returns:
            TradingSignal instance
        """
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=self.classify_strength(confidence),
            confidence=confidence,
            price=price,
            timestamp=datetime.now(),
            strategy=self.name,
            metadata=metadata or {},
        )
        
        self._signals_generated += 1
        self._last_signal = signal
        
        return signal
    
    def on_signal_executed(self, signal: TradingSignal, result: Dict):
        """
        Called when a signal is executed.
        
        Args:
            signal: The executed signal
            result: Execution result
        """
        self._signals_executed += 1
    
    def on_signal_closed(self, signal: TradingSignal, pnl: float):
        """
        Called when a signal position is closed.
        
        Args:
            signal: The original signal
            pnl: Profit/loss from the trade
        """
        if pnl > 0:
            self._signals_profitable += 1
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Performance dictionary
        """
        win_rate = (
            self._signals_profitable / self._signals_executed
            if self._signals_executed > 0 else 0
        )
        
        return {
            "name": self.name,
            "enabled": self._enabled,
            "signals_generated": self._signals_generated,
            "signals_executed": self._signals_executed,
            "signals_profitable": self._signals_profitable,
            "win_rate": win_rate,
        }
    
    def reset_performance(self):
        """Reset performance tracking."""
        self._signals_generated = 0
        self._signals_executed = 0
        self._signals_profitable = 0
        self._last_signal = None
    
    def get_params(self) -> Dict[str, Any]:
        """Get all strategy parameters."""
        return self._params.copy()
    
    def update_params(self, params: Dict[str, Any]):
        """Update multiple parameters."""
        self._params.update(params)
        logger.info(f"Strategy '{self.name}' params updated: {params}")
    
    # Additional properties and methods for test compatibility
    @property
    def max_position_size(self) -> float:
        """Get max position size."""
        return self._max_position_size
    
    @property
    def stop_loss_pct(self) -> float:
        """Get stop loss percentage."""
        return self._params.get("stop_loss_pct", 0.02)
    
    @property
    def take_profit_pct(self) -> float:
        """Get take profit percentage."""
        return self._params.get("take_profit_pct", 0.04)
    
    @property
    def is_active(self) -> bool:
        """Check if strategy is active (alias for enabled)."""
        return self._enabled
    
    def start(self):
        """Start the strategy (alias for enable)."""
        self.enable()
    
    def stop(self):
        """Stop the strategy (alias for disable)."""
        self.disable()
    
    def calculate_position_size(
        self,
        price: float,
        portfolio_value: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            price: Current price
            portfolio_value: Total portfolio value
            risk_per_trade: Risk per trade as fraction
            
        Returns:
            Position size in units
        """
        risk_amount = portfolio_value * risk_per_trade
        max_position_value = portfolio_value * self._max_position_size
        position_size = min(risk_amount / price, max_position_value / price)
        return position_size
    
    def calculate_stop_loss(self, price: float, signal_type: SignalType) -> float:
        """
        Calculate stop loss price.
        
        Args:
            price: Entry price
            signal_type: Type of signal
            
        Returns:
            Stop loss price
        """
        if signal_type in [SignalType.BUY, SignalType.CLOSE_SHORT]:
            return price * (1 - self.stop_loss_pct)
        else:
            return price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, price: float, signal_type: SignalType) -> float:
        """
        Calculate take profit price.
        
        Args:
            price: Entry price
            signal_type: Type of signal
            
        Returns:
            Take profit price
        """
        if signal_type in [SignalType.BUY, SignalType.CLOSE_SHORT]:
            return price * (1 + self.take_profit_pct)
        else:
            return price * (1 - self.take_profit_pct)
    
    def determine_strength(self, confidence: float) -> SignalStrength:
        """
        Determine signal strength based on confidence (alias for classify_strength).
        
        Args:
            confidence: Signal confidence (0-1)
            
        Returns:
            SignalStrength enum value
        """
        return self.classify_strength(confidence)
    
    def update_metrics(self, pnl: float, is_win: bool = None):
        """
        Update strategy metrics.
        
        Args:
            pnl: Profit/loss from trade
            is_win: Whether trade was profitable (auto-detected if None)
        """
        if is_win is None:
            is_win = pnl > 0
        
        if is_win:
            self._signals_profitable += 1
        
        logger.debug(f"Strategy '{self.name}' metrics updated: pnl={pnl}, win={is_win}")
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get strategy metrics."""
        return self.get_performance()
