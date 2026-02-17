"""
Position Sizing Module
Dynamic position sizing for multi-asset portfolios
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_volatility(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate the historical volatility of returns.
    
    Args:
        df: DataFrame with 'close' column
        period: Period for volatility calculation
        
    Returns:
        Volatility (standard deviation of returns)
    """
    if df is None or len(df) < period + 1:
        return 0.0
    
    returns = df['close'].pct_change().dropna()
    
    if len(returns) < period:
        return returns.std() if len(returns) > 1 else 0.0
    
    return returns.tail(period).std()


def calculate_position_size(
    capital: float,
    price: float,
    volatility: float,
    target_volatility: float = 0.02,
    max_position_pct: float = 0.1
) -> float:
    """
    Calculate position size based on volatility targeting.
    
    Args:
        capital: Available capital
        price: Current asset price
        volatility: Historical volatility of the asset
        target_volatility: Target portfolio volatility (default 2%)
        max_position_pct: Maximum position as percentage of capital
        
    Returns:
        Position size (number of units)
    """
    if volatility <= 0 or np.isnan(volatility):
        logger.warning("Invalid volatility, using equal weight")
        return (capital * max_position_pct) / price
    
    # Calculate position size based on volatility targeting
    # size = (capital * target_vol) / volatility
    raw_size = capital * (target_volatility / volatility)
    
    # Apply max position constraint
    max_size = capital * max_position_pct
    position_value = min(raw_size, max_size)
    
    # Convert to number of units
    units = position_value / price
    
    return max(0, units)


def calculate_kelly_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    fraction: float = 0.25
) -> float:
    """
    Calculate position size using Kelly Criterion.
    
    Args:
        win_rate: Win rate (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount
        capital: Available capital
        fraction: Kelly fraction to use (default 0.25 for half-Kelly)
        
    Returns:
        Position size as fraction of capital
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    
    # Calculate Kelly percentage
    win_loss_ratio = avg_win / avg_loss
    kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # Apply fractional Kelly
    kelly_pct *= fraction
    
    # Constrain to reasonable values
    kelly_pct = max(0, min(kelly_pct, 0.25))
    
    return kelly_pct


class VolatilityPositionSizer:
    """
    Position sizer that targets a specific volatility level.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.02,
        max_position_pct: float = 0.1,
        min_position_pct: float = 0.01,
        lookback_period: int = 14
    ):
        """
        Initialize the volatility position sizer.
        
        Args:
            target_volatility: Target portfolio volatility (default 2%)
            max_position_pct: Maximum position as % of capital
            min_position_pct: Minimum position as % of capital
            lookback_period: Period for volatility calculation
        """
        self.target_volatility = target_volatility
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.lookback_period = lookback_period
        
        logger.info(f"Initialized VolatilityPositionSizer: target_vol={target_volatility}")
    
    def calculate(
        self,
        df: pd.DataFrame,
        capital: float,
        price: float
    ) -> float:
        """
        Calculate position size for an asset.
        
        Args:
            df: DataFrame with price data
            capital: Available capital
            price: Current asset price
            
        Returns:
            Position size in units
        """
        volatility = calculate_volatility(df, self.lookback_period)
        
        return calculate_position_size(
            capital=capital,
            price=price,
            volatility=volatility,
            target_volatility=self.target_volatility,
            max_position_pct=self.max_position_pct
        )
    
    def calculate_for_portfolio(
        self,
        data_frames: Dict[str, pd.DataFrame],
        prices: Dict[str, float],
        total_capital: float
    ) -> Dict[str, float]:
        """
        Calculate position sizes for a portfolio of assets.
        
        Args:
            data_frames: Dictionary of DataFrames per asset
            prices: Dictionary of current prices per asset
            total_capital: Total available capital
            
        Returns:
            Dictionary of position sizes (in units) per asset
        """
        positions = {}
        
        for asset, df in data_frames.items():
            if asset not in prices:
                continue
            
            price = prices[asset]
            capital_per_asset = total_capital / len(data_frames)
            
            positions[asset] = self.calculate(df, capital_per_asset, price)
        
        return positions


class FixedRatioPositionSizer:
    """
    Fixed ratio position sizer - maintains fixed risk/reward ratio.
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        reward_risk_ratio: float = 2.0
    ):
        """
        Initialize the fixed ratio position sizer.
        
        Args:
            risk_per_trade: Risk per trade as % of capital
            reward_risk_ratio: Target reward to risk ratio
        """
        self.risk_per_trade = risk_per_trade
        self.reward_risk_ratio = reward_risk_ratio
        
        logger.info(f"Initialized FixedRatioPositionSizer: risk={risk_per_trade}")
    
    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        target_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate position size and targets.
        
        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss_price: Stop loss price
            target_price: Optional target price
            
        Returns:
            Dictionary with 'size', 'risk_amount', 'target_price'
        """
        # Calculate risk amount
        risk_amount = capital * self.risk_per_trade
        
        # Calculate stop loss distance
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance <= 0:
            return {'size': 0, 'risk_amount': 0, 'target_price': entry_price}
        
        # Calculate position size
        size = risk_amount / stop_distance
        
        # Calculate target price if not provided
        if target_price is None:
            target_price = entry_price + (stop_distance * self.reward_risk_ratio)
        
        return {
            'size': size,
            'risk_amount': risk_amount,
            'target_price': target_price
        }


class AdaptivePositionSizer:
    """
    Adaptive position sizer that adjusts based on market conditions.
    """
    
    def __init__(
        self,
        base_sizer: VolatilityPositionSizer,
        regime_adjustment: bool = True
    ):
        """
        Initialize the adaptive position sizer.
        
        Args:
            base_sizer: Base position sizer
            regime_adjustment: Whether to adjust for market regime
        """
        self.base_sizer = base_sizer
        self.regime_adjustment = regime_adjustment
        self.current_regime = 'neutral'
        
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Regime: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        if df is None or len(df) < 50:
            return 'neutral'
        
        # Simple regime detection using moving averages
        sma_short = df['close'].rolling(10).mean()
        sma_long = df['close'].rolling(50).mean()
        
        current_price = df['close'].iloc[-1]
        
        # Check trend
        if sma_short.iloc[-1] > sma_long.iloc[-1] * 1.02:
            return 'trending_up'
        elif sma_short.iloc[-1] < sma_long.iloc[-1] * 0.98:
            return 'trending_down'
        
        # Check volatility
        volatility = calculate_volatility(df)
        if volatility > 0.03:
            return 'volatile'
        
        return 'ranging'
    
    def calculate(
        self,
        df: pd.DataFrame,
        capital: float,
        price: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate position size with regime adjustment.
        
        Args:
            df: DataFrame with price data
            capital: Available capital
            price: Current asset price
            confidence: Signal confidence (0-1)
            
        Returns:
            Position size in units
        """
        # Get base size
        size = self.base_sizer.calculate(df, capital, price)
        
        if not self.regime_adjustment:
            return size * confidence
        
        # Detect regime
        regime = self.detect_regime(df)
        self.current_regime = regime
        
        # Apply regime adjustments
        regime_multipliers = {
            'trending_up': 1.2,
            'trending_down': 0.8,
            'ranging': 1.0,
            'volatile': 0.5,
            'neutral': 1.0
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        
        # Apply confidence and regime multiplier
        adjusted_size = size * confidence * multiplier
        
        return max(0, adjusted_size)
