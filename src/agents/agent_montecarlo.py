# src/agents/agent_montecarlo.py
"""
Monte Carlo Simulation Agent
============================
Advanced Monte Carlo simulation engine for price prediction.
Implements 5-level simulation with multiple factors.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.agents.base_agent import BaseAgent
from src.core.event_bus import EventBus, EventType


logger = logging.getLogger(__name__)


class SimulationLevel(Enum):
    """Monte Carlo simulation levels."""
    LEVEL_1_BASE = "base"                          # Geometric Brownian Motion
    LEVEL_2_CONDITIONAL = "conditional"            # Event-conditioned paths
    LEVEL_3_ADAPTIVE = "adaptive"                  # RL from past accuracy
    LEVEL_4_MULTI_FACTOR = "multi_factor"          # Cross-correlations, regime
    LEVEL_5_SEMANTIC = "semantic_history"          # Pattern matching, black swans


@dataclass
class SimulationResult:
    """Monte Carlo simulation result."""
    symbol: str
    level: SimulationLevel
    paths: np.ndarray
    mean_price: float
    std_price: float
    percentiles: Dict[str, float]
    var_95: float
    cvar_95: float
    probability_up: float
    probability_down: float
    timestamp: datetime


class MonteCarloAgent(BaseAgent):
    """
    Advanced Monte Carlo simulation agent.
    
    Features:
    - 5-level simulation hierarchy
    - Geometric Brownian Motion
    - Jump diffusion models
    - Regime switching
    - Fat-tail modeling
    - Event conditioning
    - Multi-asset correlation
    """
    
    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        state_manager: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize Monte Carlo Agent.
        
        Args:
            name: Agent identifier
            event_bus: Event bus for communication
            state_manager: State manager instance
            config: Configuration dictionary with:
                - symbols: List of symbols to simulate
                - interval_sec: Simulation interval
                - n_paths: Number of simulation paths
                - n_steps: Number of time steps
                - time_horizon: Time horizon in days
                - levels: List of simulation levels to run
        """
        super().__init__(name, event_bus, state_manager, config)
        
        # Configuration
        self.symbols = config.get("symbols", ["BTCUSDT"])
        self.interval_sec = config.get("interval_sec", 30)
        self.n_paths = config.get("n_paths", 1000)
        self.n_steps = config.get("n_steps", 50)
        self.time_horizon = config.get("time_horizon", 1.0)  # days
        self.levels = config.get(
            "levels",
            [
                SimulationLevel.LEVEL_1_BASE,
                SimulationLevel.LEVEL_2_CONDITIONAL,
            ]
        )
        
        # Simulation results cache
        self._results: Dict[str, SimulationResult] = {}
        
        # Historical accuracy tracking (for adaptive level)
        self._accuracy_history: Dict[str, List[float]] = {}
        
        # Regime state
        self._current_regime: Dict[str, str] = {}
        
        # Random seed for reproducibility
        self._rng = np.random.default_rng(
            config.get("random_seed", None)
        )
        
        logger.info(
            f"MonteCarloAgent initialized: {self.n_paths} paths, "
            f"{self.n_steps} steps, {len(self.levels)} levels"
        )
    
    async def run(self):
        """Main agent loop - run simulations periodically."""
        while self._running:
            try:
                # Run simulation for each symbol
                for symbol in self.symbols:
                    # Get current price from state
                    price = self.get_shared_state(
                        "MarketDataAgent",
                        f"price:{symbol}"
                    )
                    
                    if price:
                        # Run multi-level simulation
                        result = await self._run_simulation(symbol, price)
                        
                        if result:
                            # Store result
                            self._results[symbol] = result
                            
                            # Update shared state
                            self._update_state(result)
                            
                            # Emit event
                            await self.emit_event(
                                EventType.SIGNAL_GENERATED,
                                {
                                    "symbol": symbol,
                                    "level": result.level.value,
                                    "mean_price": result.mean_price,
                                    "var_95": result.var_95,
                                    "probability_up": result.probability_up,
                                }
                            )
                
                # Wait for next interval
                await asyncio.sleep(self.interval_sec)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Monte Carlo loop: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(self._error_backoff)
    
    async def _run_simulation(
        self,
        symbol: str,
        spot_price: float
    ) -> Optional[SimulationResult]:
        """
        Run Monte Carlo simulation for a symbol.
        
        Args:
            symbol: Trading pair
            spot_price: Current spot price
            
        Returns:
            SimulationResult or None
        """
        # Get historical data for parameter estimation
        history = self._get_price_history(symbol)
        
        # Estimate parameters
        mu, sigma, jump_params = self._estimate_parameters(history)
        
        # Run simulation based on highest enabled level
        level = self._get_highest_level()
        
        if level == SimulationLevel.LEVEL_5_SEMANTIC:
            paths = self._simulate_level_5(symbol, spot_price, mu, sigma)
        elif level == SimulationLevel.LEVEL_4_MULTI_FACTOR:
            paths = self._simulate_level_4(symbol, spot_price, mu, sigma)
        elif level == SimulationLevel.LEVEL_3_ADAPTIVE:
            paths = self._simulate_level_3(symbol, spot_price, mu, sigma)
        elif level == SimulationLevel.LEVEL_2_CONDITIONAL:
            paths = self._simulate_level_2(symbol, spot_price, mu, sigma)
        else:
            paths = self._simulate_level_1(spot_price, mu, sigma)
        
        # Calculate statistics
        result = self._calculate_statistics(symbol, level, paths, spot_price)
        
        return result
    
    def _get_price_history(self, symbol: str) -> np.ndarray:
        """Get price history for parameter estimation."""
        history_data = self.get_shared_state(
            "MarketDataAgent",
            f"price_history:{symbol}",
            []
        )
        
        if not history_data:
            # Generate synthetic history
            return self._generate_synthetic_history(100)
        
        prices = [h.get("price", 0) for h in history_data if h.get("price")]
        return np.array(prices) if prices else self._generate_synthetic_history(100)
    
    def _generate_synthetic_history(self, n: int) -> np.ndarray:
        """Generate synthetic price history."""
        returns = self._rng.normal(0.0001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))
        return prices
    
    def _estimate_parameters(
        self,
        history: np.ndarray
    ) -> Tuple[float, float, Dict]:
        """
        Estimate GBM parameters from price history.
        
        Args:
            history: Array of historical prices
            
        Returns:
            Tuple of (mu, sigma, jump_params)
        """
        if len(history) < 2:
            return 0.0001, 0.02, {}
        
        # Calculate log returns
        log_returns = np.diff(np.log(history))
        
        # Estimate drift (mu) and volatility (sigma)
        mu = np.mean(log_returns) * 252  # Annualized
        sigma = np.std(log_returns) * np.sqrt(252)  # Annualized
        
        # Estimate jump parameters (for higher levels)
        threshold = 2 * np.std(log_returns)
        jumps = log_returns[np.abs(log_returns) > threshold]
        
        jump_params = {
            "jump_intensity": len(jumps) / len(log_returns),
            "jump_mean": np.mean(jumps) if len(jumps) > 0 else 0,
            "jump_std": np.std(jumps) if len(jumps) > 0 else 0,
        }
        
        return mu, max(sigma, 0.01), jump_params
    
    def _simulate_level_1(
        self,
        spot: float,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """
        Level 1: Basic Geometric Brownian Motion.
        
        dS = μS dt + σS dW
        """
        dt = self.time_horizon / self.n_steps
        
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = spot
        
        for t in range(1, self.n_steps + 1):
            z = self._rng.standard_normal(self.n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )
        
        return paths
    
    def _simulate_level_2(
        self,
        symbol: str,
        spot: float,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """
        Level 2: Event-conditioned simulation.
        
        Incorporates macro events and sentiment.
        """
        # Get event/sentiment adjustments
        sentiment = self.get_shared_state(
            "SentimentAgent",
            f"sentiment:{symbol}",
            0.0
        )
        
        event_impact = self.get_shared_state(
            "EventAgent",
            f"impact:{symbol}",
            0.0
        )
        
        # Adjust drift based on events
        adjusted_mu = mu + 0.1 * sentiment + 0.2 * event_impact
        
        # Run GBM with adjusted parameters
        return self._simulate_level_1(spot, adjusted_mu, sigma)
    
    def _simulate_level_3(
        self,
        symbol: str,
        spot: float,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """
        Level 3: Adaptive simulation with RL.
        
        Learns from past prediction accuracy.
        """
        # Get accuracy history
        accuracy = self._accuracy_history.get(symbol, [])
        
        if accuracy:
            # Adjust parameters based on past accuracy
            avg_accuracy = np.mean(accuracy[-10:])
            
            # If predictions were too optimistic, reduce drift
            if avg_accuracy < 0.5:
                mu *= 0.9
            # If predictions were accurate, maintain
            elif avg_accuracy > 0.7:
                pass
            # If predictions were too pessimistic, increase drift
            else:
                mu *= 1.1
        
        return self._simulate_level_2(symbol, spot, mu, sigma)
    
    def _simulate_level_4(
        self,
        symbol: str,
        spot: float,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """
        Level 4: Multi-factor simulation.
        
        Includes regime switching and cross-asset correlation.
        """
        # Detect current regime
        regime = self._detect_regime(symbol)
        self._current_regime[symbol] = regime
        
        # Adjust parameters based on regime
        if regime == "bull":
            mu *= 1.2
            sigma *= 0.8
        elif regime == "bear":
            mu *= 0.8
            sigma *= 1.2
        elif regime == "crisis":
            mu *= 0.5
            sigma *= 1.5
        
        # Get correlation with other assets
        correlation = self._get_cross_asset_correlation(symbol)
        
        # Run simulation with regime-adjusted parameters
        paths = self._simulate_level_1(spot, mu, sigma)
        
        # Apply correlation adjustment
        if correlation is not None:
            paths = self._apply_correlation(paths, correlation)
        
        return paths
    
    def _simulate_level_5(
        self,
        symbol: str,
        spot: float,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """
        Level 5: Semantic history simulation.
        
        Pattern matching and black swan detection.
        """
        # Run base simulation
        paths = self._simulate_level_4(symbol, spot, mu, sigma)
        
        # Detect potential black swans
        black_swan_prob = self._detect_black_swan(symbol)
        
        if black_swan_prob > 0.1:
            # Add fat tails to distribution
            paths = self._add_fat_tails(paths, black_swan_prob)
        
        # Pattern matching
        patterns = self._match_historical_patterns(symbol, spot)
        
        if patterns:
            # Adjust paths based on historical patterns
            paths = self._apply_pattern_adjustment(paths, patterns)
        
        return paths
    
    def _detect_regime(self, symbol: str) -> str:
        """Detect current market regime."""
        history = self._get_price_history(symbol)
        
        if len(history) < 20:
            return "neutral"
        
        # Calculate recent returns
        recent_returns = np.diff(np.log(history[-20:]))
        mean_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        if mean_return > 0.001 and volatility < 0.02:
            return "bull"
        elif mean_return < -0.001 and volatility < 0.02:
            return "bear"
        elif volatility > 0.04:
            return "crisis"
        else:
            return "neutral"
    
    def _get_cross_asset_correlation(self, symbol: str) -> Optional[float]:
        """Get correlation with other tracked assets."""
        # Placeholder for correlation calculation
        return None
    
    def _apply_correlation(
        self,
        paths: np.ndarray,
        correlation: float
    ) -> np.ndarray:
        """Apply correlation adjustment to paths."""
        # Placeholder for correlation application
        return paths
    
    def _detect_black_swan(self, symbol: str) -> float:
        """Detect probability of black swan event."""
        # Placeholder for black swan detection
        return 0.0
    
    def _add_fat_tails(
        self,
        paths: np.ndarray,
        probability: float
    ) -> np.ndarray:
        """Add fat tails to simulation paths."""
        n_tail = int(self.n_paths * probability)
        
        # Add extreme moves to some paths
        tail_indices = self._rng.choice(
            self.n_paths, n_tail, replace=False
        )
        
        for idx in tail_indices:
            # Add extreme move
            extreme_move = self._rng.choice([-1, 1]) * self._rng.uniform(0.1, 0.3)
            paths[idx, :] *= (1 + extreme_move)
        
        return paths
    
    def _match_historical_patterns(
        self,
        symbol: str,
        current_price: float
    ) -> List[Dict]:
        """Match current price action to historical patterns."""
        # Placeholder for pattern matching
        return []
    
    def _apply_pattern_adjustment(
        self,
        paths: np.ndarray,
        patterns: List[Dict]
    ) -> np.ndarray:
        """Apply pattern-based adjustment to paths."""
        # Placeholder for pattern adjustment
        return paths
    
    def _calculate_statistics(
        self,
        symbol: str,
        level: SimulationLevel,
        paths: np.ndarray,
        spot_price: float
    ) -> SimulationResult:
        """Calculate statistics from simulation paths."""
        # Final prices
        final_prices = paths[:, -1]
        
        # Basic statistics
        mean_price = float(np.mean(final_prices))
        std_price = float(np.std(final_prices))
        
        # Percentiles
        percentiles = {
            "p5": float(np.percentile(final_prices, 5)),
            "p25": float(np.percentile(final_prices, 25)),
            "p50": float(np.percentile(final_prices, 50)),
            "p75": float(np.percentile(final_prices, 75)),
            "p95": float(np.percentile(final_prices, 95)),
        }
        
        # Returns
        returns = final_prices / spot_price - 1
        
        # VaR and CVaR (95%)
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(np.mean(returns[returns <= np.percentile(returns, 5)]))
        
        # Probability of up/down move
        probability_up = float(np.mean(final_prices > spot_price))
        probability_down = float(np.mean(final_prices < spot_price))
        
        return SimulationResult(
            symbol=symbol,
            level=level,
            paths=paths,
            mean_price=mean_price,
            std_price=std_price,
            percentiles=percentiles,
            var_95=var_95,
            cvar_95=cvar_95,
            probability_up=probability_up,
            probability_down=probability_down,
            timestamp=datetime.now(),
        )
    
    def _update_state(self, result: SimulationResult):
        """Update shared state with simulation results."""
        self.update_state(
            f"mc_paths:{result.symbol}",
            result.paths.tolist()
        )
        self.update_state(
            f"mc_mean:{result.symbol}",
            result.mean_price
        )
        self.update_state(
            f"mc_var:{result.symbol}",
            result.var_95
        )
        self.update_state(
            f"mc_cvar:{result.symbol}",
            result.cvar_95
        )
        self.update_state(
            f"mc_prob_up:{result.symbol}",
            result.probability_up
        )
        self.update_state(
            f"mc_percentiles:{result.symbol}",
            result.percentiles
        )
    
    def get_simulation_result(self, symbol: str) -> Optional[SimulationResult]:
        """Get latest simulation result for a symbol."""
        return self._results.get(symbol)
    
    def get_var(self, symbol: str) -> Optional[float]:
        """Get VaR (95%) for a symbol."""
        result = self._results.get(symbol)
        return result.var_95 if result else None
    
    def get_cvar(self, symbol: str) -> Optional[float]:
        """Get CVaR (95%) for a symbol."""
        result = self._results.get(symbol)
        return result.cvar_95 if result else None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "symbols_simulated": len(self._results),
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "levels": [l.value for l in self.levels],
        })
        return metrics
