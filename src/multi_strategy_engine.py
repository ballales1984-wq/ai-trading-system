"""
Multi-Strategy Engine
====================
Engine that runs multiple strategies in parallel and aggregates signals.

Features:
- Load multiple strategies
- Run in parallel
- Aggregate signals with dynamic weights
- Track individual strategy performance
- Produce final ensemble signal
"""

import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for all strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_history = []
    
    @abstractmethod
    def generate_signal(self, df) -> int:
        """
        Generate trading signal.
        
        Returns:
            -1 = short
             0 = flat
             1 = long
        """
        pass
    
    def update_performance(self, pnl: float):
        """Track strategy performance."""
        self.performance_history.append(pnl)
    
    def get_avg_performance(self, window: int = 20) -> float:
        """Get average performance over window."""
        if len(self.performance_history) == 0:
            return 0.0
        return np.mean(self.performance_history[-window:])


class TrendStrategy(BaseStrategy):
    """Trend following strategy."""
    
    def __init__(self):
        super().__init__("Trend")
    
    def generate_signal(self, df) -> int:
        if len(df) < 50:
            return 0
        
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma50 = df['close'].rolling(50).mean().iloc[-1]
        price = df['close'].iloc[-1]
        
        if price > ma20 > ma50:
            return 1
        elif price < ma20 < ma50:
            return -1
        return 0


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands."""
    
    def __init__(self):
        super().__init__("MeanReversion")
    
    def generate_signal(self, df) -> int:
        if len(df) < 20:
            return 0
        
        # Calculate Bollinger Bands
        ma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        bb_upper = ma + 2 * std
        bb_lower = ma - 2 * std
        
        price = df['close'].iloc[-1]
        lower = bb_lower.iloc[-1]
        upper = bb_upper.iloc[-1]
        
        if price < lower:
            return 1  # Oversold - buy
        elif price > upper:
            return -1  # Overbought - sell
        return 0


class MLStrategy(BaseStrategy):
    """Machine Learning strategy."""
    
    def __init__(self, model=None):
        super().__init__("ML")
        self.model = model
    
    def generate_signal(self, df) -> int:
        if self.model is None:
            # Default: use RSI-like signal
            if len(df) < 14:
                return 0
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if rsi.iloc[-1] < 30:
                return 1
            elif rsi.iloc[-1] > 70:
                return -1
            return 0
        
        # Use ML model
        try:
            features = df.iloc[-1:][['close', 'volume']]
            prob = self.model.predict_proba(features)[0][1]
            
            if prob > 0.55:
                return 1
            elif prob < 0.45:
                return -1
            return 0
        except:
            return 0


class RLStrategy(BaseStrategy):
    """Reinforcement Learning strategy."""
    
    def __init__(self, agent=None):
        super().__init__("RL")
        self.agent = agent
    
    def generate_signal(self, df) -> int:
        if self.agent is None:
            # Default: momentum
            if len(df) < 10:
                return 0
            
            returns = df['close'].pct_change().tail(10).mean()
            
            if returns > 0.01:
                return 1
            elif returns < -0.01:
                return -1
            return 0
        
        try:
            return self.agent.predict(df.iloc[-1:])
        except:
            return 0


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy."""
    
    def __init__(self):
        super().__init__("Breakout")
    
    def generate_signal(self, df) -> int:
        if len(df) < 20:
            return 0
        
        high = df['high'].rolling(20).max().iloc[-1]
        low = df['low'].rolling(20).min().iloc[-1]
        price = df['close'].iloc[-1]
        
        if price > high * 0.99:  # Breakout up
            return 1
        elif price < low * 1.01:  # Breakout down
            return -1
        return 0


class MultiStrategyEngine:
    """
    Engine that runs multiple strategies in parallel.
    """
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.weights: Dict[str, float] = {}
        self.performance_tracker: Dict[str, List[float]] = {}
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the engine."""
        self.strategies[strategy.name] = strategy
        self.weights[strategy.name] = weight
        self.performance_tracker[strategy.name] = []
        logger.info(f"✅ Strategy added: {strategy.name} (weight: {weight})")
    
    def set_weight(self, strategy_name: str, weight: float):
        """Set weight for a strategy."""
        if strategy_name in self.weights:
            self.weights[strategy_name] = weight
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights (normalized)."""
        total = sum(self.weights.values())
        if total == 0:
            return {k: 1/len(self.weights) for k in self.weights}
        return {k: v/total for k, v in self.weights.items()}
    
    def generate_signals(self, df) -> Dict[str, int]:
        """Generate signals from all strategies."""
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                signals[name] = strategy.generate_signal(df)
            except Exception as e:
                logger.error(f"Error generating signal for {name}: {e}")
                signals[name] = 0
        return signals
    
    def aggregate_signal(self, df) -> tuple:
        """
        Aggregate signals from all strategies.
        
        Returns:
            (final_signal, all_signals, weights)
        """
        signals = self.generate_signals(df)
        weights = self.get_weights()
        
        # Weighted sum
        weighted_sum = sum(
            signals.get(name, 0) * weights.get(name, 0)
            for name in self.strategies
        )
        
        # Final signal
        final_signal = int(np.sign(weighted_sum))
        
        return final_signal, signals, weights
    
    def update_performance(self, strategy_name: str, pnl: float):
        """Update performance for a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_performance(pnl)
            self.performance_tracker[strategy_name].append(pnl)
    
    def get_performance(self, strategy_name: str, window: int = 20) -> float:
        """Get average performance for a strategy."""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].get_avg_performance(window)
        return 0.0
    
    def get_all_performance(self) -> Dict[str, float]:
        """Get performance for all strategies."""
        return {
            name: self.get_performance(name)
            for name in self.strategies
        }


class StrategyWeighting:
    """
    Dynamic strategy weighting based on performance.
    """
    
    def __init__(self, engine: MultiStrategyEngine):
        self.engine = engine
        self.lookback = 20
    
    def compute_weights(self) -> Dict[str, float]:
        """Compute weights based on recent performance."""
        performance = self.engine.get_all_performance()
        
        weights = {}
        for name, perf in performance.items():
            # Use performance as weight (min 0.01)
            weights[name] = max(perf, 0.01)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            # Equal weights if no performance data
            weights = {k: 1/len(weights) for k in weights}
        
        return weights
    
    def apply_weights(self):
        """Apply computed weights to engine."""
        weights = self.compute_weights()
        for name, weight in weights.items():
            self.engine.set_weight(name, weight)
        
        logger.info(f"📊 Weights updated: {weights}")


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    # Create engine
    mse = MultiStrategyEngine()
    
    # Add strategies
    mse.add_strategy(TrendStrategy(), weight=0.3)
    mse.add_strategy(MeanReversionStrategy(), weight=0.2)
    mse.add_strategy(MLStrategy(), weight=0.3)
    mse.add_strategy(RLStrategy(), weight=0.2)
    
    # Get signals
    import pandas as pd
    
    # Dummy data
    df = pd.DataFrame({
        'close': [100 + i for i in range(100)],
        'high': [105 + i for i in range(100)],
        'low': [95 + i for i in range(100)],
        'volume': [1000] * 100
    })
    
    final_signal, signals, weights = mse.aggregate_signal(df)
    
    print(f"📊 Signals: {signals}")
    print(f"📊 Weights: {weights}")
    print(f"🎯 Final Signal: {final_signal}")
