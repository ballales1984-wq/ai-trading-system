"""
Trailing Stop and Dynamic Stop Loss Module
Provides advanced risk management features
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop"""
    activation_profit_percent: float = 1.0  # Activate after 1% profit
    trailing_distance_percent: float = 0.5   # Trail 0.5% behind
    min_trailing_distance: float = 0.002    # Minimum 0.2%
    max_trailing_distance: float = 0.02     # Maximum 2%


class TrailingStopManager:
    """
    Manages trailing stops for positions
    """
    
    def __init__(self, config: TrailingStopConfig = None):
        self.config = config or TrailingStopConfig()
        self._positions = {}  # symbol -> position data
    
    def update_position(self, symbol: str, entry_price: float, 
                       current_price: float, position_type: str = "LONG"):
        """
        Update position with new price
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price of position
            current_price: Current market price
            position_type: LONG or SHORT
        """
        profit_percent = ((current_price - entry_price) / entry_price) * 100
        
        if position_type == "SHORT":
            profit_percent = -profit_percent
        
        # Initialize position if new
        if symbol not in self._positions:
            self._positions[symbol] = {
                'entry_price': entry_price,
                'highest_price': current_price if position_type == "LONG" else entry_price,
                'lowest_price': current_price if position_type == "SHORT" else entry_price,
                'position_type': position_type,
                'trailing_stop': None,
                'initial_stop': entry_price * (1 - self.config.trailing_distance_percent),
                'profit_percent': profit_percent,
                'stop_triggered': False
            }
        
        pos = self._positions[symbol]
        pos['profit_percent'] = profit_percent
        
        # Update highest/lowest price
        if position_type == "LONG":
            if current_price > pos['highest_price']:
                pos['highest_price'] = current_price
        else:
            if current_price < pos['lowest_price']:
                pos['lowest_price'] = current_price
        
        # Check if trailing stop should activate
        if profit_percent >= self.config.activation_profit_percent:
            # Calculate trailing stop
            if position_type == "LONG":
                new_stop = pos['highest_price'] * (1 - self.config.trailing_distance_percent)
            else:
                new_stop = pos['lowest_price'] * (1 + self.config.trailing_distance_percent)
            
            # Apply min/max constraints
            new_stop = max(new_stop, entry_price * (1 + self.config.min_trailing_distance))
            if self.config.max_trailing_distance > 0:
                new_stop = min(new_stop, entry_price * (1 + self.config.max_trailing_distance))
            
            # Only update if new stop is better
            if pos['trailing_stop'] is None or (
                position_type == "LONG" and new_stop > pos['trailing_stop']
            ) or (
                position_type == "SHORT" and new_stop < pos['trailing_stop']
            ):
                pos['trailing_stop'] = new_stop
                logger.info(f"Trailing stop updated for {symbol}: {new_stop:.4f}")
    
    def get_stop_loss(self, symbol: str) -> Optional[float]:
        """
        Get current stop loss for position
        
        Returns:
            Stop loss price or None if no position
        """
        if symbol not in self._positions:
            return None
        
        pos = self._positions[symbol]
        
        # Return trailing stop if active, otherwise initial stop
        if pos['trailing_stop'] is not None:
            return pos['trailing_stop']
        return pos['initial_stop']
    
    def check_stop_triggered(self, symbol: str, current_price: float) -> bool:
        """
        Check if stop loss is triggered
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            True if stop is triggered
        """
        if symbol not in self._positions:
            return False
        
        pos = self._positions[symbol]
        stop_price = self.get_stop_loss(symbol)
        
        if stop_price is None:
            return False
        
        if pos['position_type'] == "LONG":
            triggered = current_price <= stop_price
        else:
            triggered = current_price >= stop_price
        
        if triggered:
            pos['stop_triggered'] = True
            logger.info(f"Stop triggered for {symbol} at {current_price}, stop was {stop_price}")
        
        return triggered
    
    def close_position(self, symbol: str):
        """Remove position from tracking"""
        if symbol in self._positions:
            del self._positions[symbol]
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get position information"""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict:
        """Get all tracked positions"""
        return self._positions.copy()


class DynamicStopLoss:
    """
    Dynamic stop loss based on ATR and volatility
    """
    
    def __init__(self, atr_multiplier: float = 2.0, 
                 min_stop_percent: float = 0.5,
                 max_stop_percent: float = 5.0):
        self.atr_multiplier = atr_multiplier
        self.min_stop_percent = min_stop_percent
        self.max_stop_percent = max_stop_percent
    
    def calculate_stop_loss(self, entry_price: float, atr: float,
                          current_price: float, volatility: float = None) -> Dict[str, float]:
        """
        Calculate dynamic stop loss levels
        
        Args:
            entry_price: Position entry price
            atr: Average True Range
            current_price: Current market price
            volatility: Optional volatility measure
            
        Returns:
            Dict with 'stop_loss', 'take_profit', 'risk_percent'
        """
        # Calculate ATR-based stop
        atr_stop = atr
        
        # Calculate percentage stop
        base_stop_percent = (atr_stop / current_price) * self.atr_multiplier * 100
        
        # Apply constraints
        stop_percent = np.clip(
            base_stop_percent,
            self.min_stop_percent,
            self.max_stop_percent
        )
        
        # Calculate stop loss price
        stop_loss = entry_price * (1 - stop_percent / 100)
        
        # Calculate take profit (1.5:1 risk-reward)
        risk = entry_price - stop_loss
        take_profit = entry_price + (risk * 1.5)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_percent': stop_percent,
            'atr_stop': atr_stop
        }


class HedgingManager:
    """
    Manages hedging strategies based on correlations
    """
    
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
        self.hedges = {}  # symbol -> hedge_symbol
    
    def calculate_hedge_ratio(self, symbol: str, hedge_symbol: str,
                            correlation: float) -> Optional[float]:
        """
        Calculate hedge ratio based on correlation
        
        Args:
            symbol: Primary symbol
            hedge_symbol: Hedge symbol
            correlation: Correlation coefficient
            
        Returns:
            Hedge ratio or None if correlation too low
        """
        if abs(correlation) < self.correlation_threshold:
            return None
        
        # Higher correlation = higher hedge ratio
        hedge_ratio = abs(correlation)
        
        # Store hedge relationship
        self.hedges[symbol] = hedge_symbol
        
        return hedge_ratio
    
    def get_hedge_position_size(self, position_size: float, 
                               hedge_ratio: float) -> float:
        """
        Calculate hedge position size
        
        Args:
            position_size: Size of primary position
            hedge_ratio: Ratio from correlation
            
        Returns:
            Hedge position size
        """
        return position_size * hedge_ratio
    
    def should_hedge(self, symbol: str, correlation: float) -> bool:
        """Check if should establish hedge"""
        return abs(correlation) >= self.correlation_threshold


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes for better signals
    """
    
    SUPPORTED_TIMEFRAMES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ['1h', '4h', '1d']
    
    def get_signal_alignment(self, signals: Dict[str, int]) -> str:
        """
        Get alignment of signals across timeframes
        
        Args:
            signals: Dict of timeframe -> signal (-1, 0, 1)
            
        Returns:
            Alignment: 'BULLISH', 'BEARISH', 'NEUTRAL', 'MIXED'
        """
        if not signals:
            return 'NEUTRAL'
        
        # Weight by timeframe (longer = more important)
        weights = {
            '1m': 1, '5m': 2, '15m': 3,
            '1h': 4, '4h': 5, '1d': 6
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for tf, signal in signals.items():
            weight = weights.get(tf, 1)
            weighted_sum += signal * weight
            total_weight += weight
        
        if total_weight == 0:
            return 'NEUTRAL'
        
        avg_signal = weighted_sum / total_weight
        
        if avg_signal > 0.3:
            return 'BULLISH'
        elif avg_signal < -0.3:
            return 'BEARISH'
        elif -0.3 <= avg_signal <= 0.3:
            return 'MIXED'
        
        return 'NEUTRAL'
    
    def calculate_timeframe_confidence(self, signals: Dict[str, int]) -> float:
        """Calculate confidence based on timeframe alignment"""
        if not signals:
            return 0.0
        
        alignment = self.get_signal_alignment(signals)
        
        # Base confidence on alignment
        confidence_map = {
            'BULLISH': 0.8,
            'BEARISH': 0.8,
            'MIXED': 0.4,
            'NEUTRAL': 0.3
        }
        
        return confidence_map.get(alignment, 0.3)


if __name__ == "__main__":
    # Test trailing stop
    tsm = TrailingStopManager()
    
    # Simulate LONG position
    entry = 45000
    tsm.update_position('BTC/USDT', entry, entry, 'LONG')
    
    # Price goes up
    tsm.update_position('BTC/USDT', entry, 46000, 'LONG')
    print(f"Stop after 2% gain: {tsm.get_stop_loss('BTC/USDT')}")
    
    # Price goes up more
    tsm.update_position('BTC/USDT', entry, 47000, 'LONG')
    print(f"Stop after 4.4% gain: {tsm.get_stop_loss('BTC/USDT')}")
    
    # Test multi-timeframe
    mta = MultiTimeframeAnalyzer()
    signals = {'1h': 1, '4h': 1, '1d': 1}
    print(f"Alignment: {mta.get_signal_alignment(signals)}")
    print(f"Confidence: {mta.calculate_timeframe_confidence(signals)}")
