"""
AI Trading System - Backtest Engine

A professional-grade backtesting engine for testing trading strategies
against historical data with realistic simulation.

Features:
- Historical data simulation
- Multiple order types (market, limit, stop)
- Transaction costs (commission, slippage)
- Performance metrics calculation
- Walk-forward validation
- Monte Carlo simulation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from copy import deepcopy
import random
import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders supported in backtest"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class BacktestStatus(Enum):
    """Backtest run status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    
@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None       # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: str = "pending"            # pending, filled, cancelled
    filled_price: Optional[float] = None
    filled_quantity: float = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    filled_timestamp: Optional[datetime] = None
    
    
@dataclass
class Trade:
    """Represents a completed trade"""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    pnl: float = 0                     # Realized P&L
    
    
@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float = 0
    unrealized_pnl: float = 0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        else:
            self.unrealized_pnl = 0


@dataclass 
class BacktestConfig:
    """Configuration for backtest run"""
    initial_capital: float = 100000
    commission_rate: float = 0.001        # 0.1%
    slippage_model: str = "fixed"          # fixed, volume_weighted, random
    slippage_pct: float = 0.0005           # 0.05% slippage
    maker_fee_rate: float = 0.001
    taker_fee_rate: float = 0.001
    min_order_size: float = 0.0001
    max_position_size: float = 1.0         # 100% of capital
    latency_ms: int = 100                   # Simulated latency
    
    
@dataclass
class BacktestResult:
    """Results of a backtest run"""
    status: BacktestStatus
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0
    
    # Risk Metrics
    max_drawdown: float = 0
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    
    # Trade Metrics
    avg_win: float = 0
    avg_loss: float = 0
    largest_win: float = 0
    largest_loss: float = 0
    avg_trade_duration: float = 0
    
    # Additional Metrics
    profit_factor: float = 0
    expectancy: float = 0
    recovery_factor: float = 0
    
    # Equity Curve
    equity_curve: List[Dict] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    # Metadata
    execution_time_seconds: float = 0
    error_message: Optional[str] = None


class BacktestEngine:
    """
    Professional backtesting engine for algorithmic trading strategies.
    
    Features:
    - Realistic order execution simulation
    - Multiple slippage models
    - Transaction cost modeling
    - Comprehensive performance analytics
    - Walk-forward validation support
    - Monte Carlo simulation
    """
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        data_provider: Optional[Callable] = None
    ):
        self.config = config or BacktestConfig()
        self.data_provider = data_provider
        
        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Statistics
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        
        # Trade tracking
        self.trade_id_counter = 0
        
    def reset(self):
        """Reset the backtest engine state"""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.start_date = None
        self.end_date = None
        self.start_time = None
        
    async def run(
        self,
        strategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> BacktestResult:
        """
        Run a backtest for a strategy.
        
        Args:
            strategy: Trading strategy instance with generate_signal method
            symbols: List of trading symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 1h, 1d)
            
        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        self.start_date = start
        self.end_date = end
        self.start_time = datetime.utcnow()
        
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        
        try:
            # Load historical data
            historical_data = await self._load_historical_data(
                symbols, start, end, interval
            )
            
            if not historical_data:
                return BacktestResult(
                    status=BacktestStatus.FAILED,
                    start_date=start,
                    end_date=end,
                    initial_capital=self.config.initial_capital,
                    final_capital=self.config.initial_capital,
                    total_return=0,
                    total_return_pct=0,
                    error_message="No historical data available"
                )
            
            # Run backtest simulation
            await self._run_simulation(strategy, historical_data, symbols)
            
            # Calculate results
            result = self._calculate_results()
            
            logger.info(f"Backtest completed in {result.execution_time_seconds:.2f}s")
            logger.info(f"Final capital: ${result.final_capital:,.2f}")
            logger.info(f"Return: {result.total_return_pct:.2f}%")
            logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
            logger.info(f"Win Rate: {result.win_rate:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return BacktestResult(
                status=BacktestStatus.FAILED,
                start_date=start,
                end_date=end,
                initial_capital=self.config.initial_capital,
                final_capital=self.capital,
                total_return=0,
                total_return_pct=0,
                error_message=str(e)
            )
    
    async def _load_historical_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval: str
    ) -> Dict[str, List[OHLCV]]:
        """Load historical price data"""
        data = {}
        
        for symbol in symbols:
            if self.data_provider:
                # Use custom data provider
                candles = await self.data_provider(symbol, start, end, interval)
            else:
                # Generate synthetic data for demonstration
                candles = self._generate_synthetic_data(symbol, start, end, interval)
            
            if candles:
                data[symbol] = candles
                
        return data
    
    def _generate_synthetic_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[OHLCV]:
        """Generate synthetic OHLCV data for testing"""
        # Determine interval duration
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1)
        }
        delta = interval_map.get(interval, timedelta(hours=1))
        
        # Generate price series with random walk
        candles = []
        current_time = start
        base_price = 100.0  # Starting price
        
        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, int((end - start) / delta))
        
        for i, ret in enumerate(returns):
            current_time = start + delta * i
            
            # Generate OHLC from return
            open_price = base_price * (1 + ret)
            
            # Add intrabar volatility
            volatility = abs(ret) * 2
            high = open_price * (1 + volatility)
            low = open_price * (1 - volatility)
            close = open_price * (1 + np.random.normal(0, volatility/2))
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.exponential(1000) * (1 + abs(ret) * 10)
            
            candles.append(OHLCV(
                timestamp=current_time,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            ))
            
            base_price = close
            
        return candles
    
    async def _run_simulation(
        self,
        strategy,
        historical_data: Dict[str, List[OHLCV]],
        symbols: List[str]
    ):
        """Run the backtest simulation"""
        # Find the maximum length
        max_length = max(len(data) for data in historical_data.values())
        
        # Create context data for strategy
        context = {
            symbol: {
                "prices": [],
                "volumes": [],
                "highs": [],
                "lows": [],
                "opens": []
            }
            for symbol in symbols
        }
        
        # Iterate through each time step
        for i in range(max_length):
            current_time = None
            
            # Update context with latest candle
            for symbol in symbols:
                data = historical_data.get(symbol, [])
                if i < len(data):
                    candle = data[i]
                    current_time = candle.timestamp
                    
                    ctx = context[symbol]
                    ctx["prices"].append(candle.close)
                    ctx["volumes"].append(candle.volume)
                    ctx["highs"].append(candle.high)
                    ctx["lows"].append(candle.low)
                    ctx["opens"].append(candle.open)
                    
                    # Keep only relevant lookback data
                    max_lookback = 200
                    for key in ctx:
                        if len(ctx[key]) > max_lookback:
                            ctx[key] = ctx[key][-max_lookback:]
            
            if current_time is None:
                continue
                
            # Get current prices
            current_prices = {
                symbol: historical_data[symbol][i].close
                for symbol in symbols
                if i < len(historical_data.get(symbol, []))
            }
            
            # Update positions with current prices
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position.update(current_prices[symbol])
            
            # Generate signals from strategy
            for symbol in symbols:
                if symbol not in current_prices:
                    continue
                    
                ctx = context[symbol]
                if len(ctx["prices"]) < 10:
                    continue
                
                # Get signal from strategy
                signal = strategy.generate_signal(symbol, ctx)
                
                if signal and signal.action in ["buy", "sell"]:
                    # Execute signal as market order
                    await self._execute_signal(symbol, signal, current_prices[symbol])
            
            # Process pending orders
            await self._process_orders(current_prices)
            
            # Record equity
            total_equity = self._calculate_equity(current_prices)
            self.equity_curve.append({
                "timestamp": current_time,
                "equity": total_equity,
                "capital": self.capital,
                "positions_value": total_equity - self.capital
            })
    
    async def _execute_signal(self, symbol: str, signal, current_price: float):
        """Execute a trading signal"""
        # Calculate position size
        position_size = self._calculate_position_size(
            signal.action,
            current_price,
            signal.confidence if hasattr(signal, 'confidence') else 1.0
        )
        
        if position_size <= 0:
            return
            
        # Check if we have position
        has_position = symbol in self.positions and self.positions[symbol].quantity > 0
        position = self.positions.get(symbol)
        
        if signal.action == "buy":
            if has_position and position.side == PositionSide.LONG:
                # Already long, could add to position
                pass
            else:
                # Open long position
                await self._open_position(symbol, OrderSide.BUY, position_size, current_price)
                
        elif signal.action == "sell":
            if has_position and position.side == PositionSide.SHORT:
                # Already short
                pass
            elif has_position:
                # Close long position
                await self._close_position(symbol, OrderSide.SELL, position.quantity, current_price)
            else:
                # Open short position
                await self._open_position(symbol, OrderSide.SELL, position_size, current_price)
    
    def _calculate_position_size(
        self,
        action: str,
        price: float,
        confidence: float = 1.0
    ) -> float:
        """Calculate position size based on risk management"""
        # Use fixed position sizing (2% of capital)
        position_value = self.capital * 0.02 * confidence
        
        # Apply max position limit
        max_position_value = self.capital * self.config.max_position_size
        position_value = min(position_value, max_position_value)
        
        # Convert to quantity
        quantity = position_value / price
        
        # Apply minimum order size
        quantity = max(quantity, self.config.min_order_size)
        
        return quantity
    
    async def _open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float
    ):
        """Open a new position"""
        # Calculate costs
        fill_price = self._apply_slippage(current_price, side)
        commission = fill_price * quantity * self.config.commission_rate
        
        total_cost = fill_price * quantity + commission
        
        if total_cost > self.capital:
            logger.warning(f"Insufficient capital for {symbol} position")
            return
            
        # Deduct cost from capital
        self.capital -= total_cost
        
        # Create position
        position_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
        self.positions[symbol] = Position(
            symbol=symbol,
            side=position_side,
            quantity=quantity,
            entry_price=fill_price,
            current_price=fill_price
        )
        
        # Record trade
        self.trade_id_counter += 1
        trade = Trade(
            trade_id=str(self.trade_id_counter),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            slippage=abs(fill_price - current_price) * quantity,
            timestamp=datetime.utcnow()
        )
        self.trades.append(trade)
        
        logger.debug(f"Opened {side.value} position: {symbol} {quantity} @ {fill_price}")
    
    async def _close_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float
    ):
        """Close an existing position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        quantity = min(quantity, position.quantity)
        
        # Calculate proceeds
        fill_price = self._apply_slippage(current_price, side)
        commission = fill_price * quantity * self.config.commission_rate
        
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (fill_price - position.entry_price) * quantity
        else:
            pnl = (position.entry_price - fill_price) * quantity
            
        pnl -= commission
        
        # Add proceeds to capital
        proceeds = fill_price * quantity - commission
        self.capital += proceeds + pnl
        
        # Update position
        position.quantity -= quantity
        if position.quantity <= 0:
            del self.positions[symbol]
        
        # Record trade
        self.trade_id_counter += 1
        trade = Trade(
            trade_id=str(self.trade_id_counter),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            slippage=abs(fill_price - current_price) * quantity,
            timestamp=datetime.utcnow(),
            pnl=pnl
        )
        self.trades.append(trade)
        
        logger.debug(f"Closed {side.value} position: {symbol} {quantity} @ {fill_price} (PnL: {pnl:.2f})")
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to price based on configured model"""
        if self.config.slippage_model == "fixed":
            slippage = price * self.config.slippage_pct
        elif self.config.slippage_model == "random":
            slippage = price * self.config.slippage_pct * random.uniform(0.5, 1.5)
        elif self.config.slippage_model == "volume_weighted":
            # Simplified volume-weighted slippage
            slippage = price * self.config.slippage_pct * 1.2
        else:
            slippage = 0
            
        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage
    
    async def _process_orders(self, prices: Dict[str, float]):
        """Process pending orders"""
        # For market orders, fill immediately
        for order in self.orders[:]:
            if order.status != "pending":
                continue
                
            if order.symbol not in prices:
                continue
                
            current_price = prices[order.symbol]
            
            if order.order_type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    await self._open_position(
                        order.symbol, order.side, order.quantity, current_price
                    )
                else:
                    await self._close_position(
                        order.symbol, order.side, order.quantity, current_price
                    )
                order.status = "filled"
                order.filled_price = self._apply_slippage(current_price, order.side)
                order.filled_timestamp = datetime.utcnow()
                
        # Remove filled orders
        self.orders = [o for o in self.orders if o.status == "pending"]
    
    def _calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total equity"""
        positions_value = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.current_price)
            for pos in self.positions.values()
        )
        return self.capital + positions_value
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        end_time = datetime.utcnow()
        execution_time = (end_time - self.start_time).total_seconds()
        
        # Calculate final equity
        if self.positions:
            # Use last known prices from equity curve
            if self.equity_curve:
                final_equity = self.equity_curve[-1]["equity"]
            else:
                final_equity = self.capital
        else:
            final_equity = self.capital
        
        total_return = final_equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100
        
        # Trade statistics
        completed_trades = [t for t in self.trades if t.pnl != 0]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        total_trades = len(completed_trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0
        
        # Win/Loss metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades], default=0)
        largest_loss = min([t.pnl for t in losing_trades], default=0)
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * abs(avg_loss)) if total_trades > 0 else 0
        
        # Calculate drawdown from equity curve
        max_equity = self.config.initial_capital
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for point in self.equity_curve:
            equity = point["equity"]
            if equity > max_equity:
                max_equity = equity
            
            drawdown = max_equity - equity
            drawdown_pct = (drawdown / max_equity * 100) if max_equity > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Sharpe Ratio (annualized)
        if len(self.equity_curve) > 1:
            returns = [
                (self.equity_curve[i]["equity"] - self.equity_curve[i-1]["equity"]) 
                / self.equity_curve[i-1]["equity"]
                for i in range(1, len(self.equity_curve))
            ]
            if returns and np.std(returns) > 0:
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24)  # Hourly
                sortino_ratio = (np.mean(returns) / np.std([r for r in returns if r < 0])) * np.sqrt(252 * 24) if any(r < 0 for r in returns) else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calmar Ratio
        annual_return = total_return_pct * 365 / max(1, (self.end_date - self.start_date).days)
        calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Recovery Factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return BacktestResult(
            status=BacktestStatus.COMPLETED,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            equity_curve=self.equity_curve,
            trades=self.trades,
            execution_time_seconds=execution_time
        )


# =============================================================================
# Walk-Forward Analysis
# =============================================================================

class WalkForwardAnalyzer:
    """
    Walk-forward analysis for robust strategy validation.
    
    Splits historical data into training and testing windows,
    rolling forward to validate strategy performance.
    """
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
        
    async def run_analysis(
        self,
        strategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        train_period_days: int = 180,
        test_period_days: int = 30,
        step_days: int = 7
    ) -> List[BacktestResult]:
        """
        Run walk-forward analysis.
        
        Args:
            strategy: Trading strategy
            symbols: Trading symbols
            start_date: Analysis start
            end_date: Analysis end
            train_period_days: Training window (days)
            test_period_days: Testing window (days)
            step_days: Step size between windows
            
        Returns:
            List of BacktestResult for each test period
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = []
        current_train_start = start
        current_train_end = current_train_start + timedelta(days=train_period_days)
        current_test_start = current_train_end
        current_test_end = current_test_start + timedelta(days=test_period_days)
        
        iteration = 0
        
        while current_test_end <= end:
            iteration += 1
            logger.info(f"Walk-forward iteration {iteration}")
            logger.info(f"  Train: {current_train_start.date()} to {current_train_end.date()}")
            logger.info(f"  Test:  {current_test_start.date()} to {current_test_end.date()}")
            
            # Run backtest on test period
            result = await self.engine.run(
                strategy=strategy,
                symbols=symbols,
                start_date=current_test_start.strftime("%Y-%m-%d"),
                end_date=current_test_end.strftime("%Y-%m-%d")
            )
            
            results.append(result)
            
            # Move windows forward
            current_train_start = current_train_start + timedelta(days=step_days)
            current_train_end = current_train_start + timedelta(days=train_period_days)
            current_test_start = current_train_end
            current_test_end = current_test_start + timedelta(days=test_period_days)
        
        return results


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

class MonteCarloBacktest:
    """
    Monte Carlo simulation for strategy robustness testing.
    
    Runs multiple backtests with randomized parameters to
    assess strategy stability and potential drawdown scenarios.
    """
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
        
    async def run_simulation(
        self,
        strategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        n_simulations: int = 100,
        param_ranges: Optional[Dict[str, Tuple]] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            strategy: Base strategy
            symbols: Trading symbols
            start_date: Backtest start
            end_date: Backtest end
            n_simulations: Number of simulations
            param_ranges: Parameter ranges to randomize
            
        Returns:
            Dictionary with simulation statistics
        """
        results = []
        
        for i in range(n_simulations):
            # Create strategy copy with randomized params
            sim_strategy = deepcopy(strategy)
            
            if param_ranges:
                # Randomize parameters
                for param, (min_val, max_val) in param_ranges.items():
                    if hasattr(sim_strategy, param):
                        setattr(
                            sim_strategy, param,
                            random.uniform(min_val, max_val)
                        )
            
            # Run backtest
            result = await self.engine.run(
                strategy=sim_strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{n_simulations} simulations")
        
        # Calculate statistics
        returns = [r.total_return_pct for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        max_dds = [r.max_drawdown_pct for r in results]
        
        return {
            "n_simulations": n_simulations,
            "return_mean": np.mean(returns),
            "return_median": np.median(returns),
            "return_std": np.std(returns),
            "return_5pct": np.percentile(returns, 5),
            "return_95pct": np.percentile(returns, 95),
            "sharpe_mean": np.mean(sharpes),
            "sharpe_median": np.median(sharpes),
            "max_dd_mean": np.mean(max_dds),
            "max_dd_95pct": np.percentile(max_dds, 95),
            "success_rate": len([r for r in results if r.total_return > 0]) / n_simulations * 100,
            "results": results
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    """
    Example: Running a backtest with the BacktestEngine
    
    This demonstrates how to use the backtest engine with a simple
    moving average crossover strategy.
    """
    import asyncio
    from dataclasses import dataclass
    
    @dataclass
    class Signal:
        """Simple trading signal"""
        action: str  # 'buy', 'sell', 'hold'
        confidence: float = 1.0
    
    class SimpleMAStrategy:
        """
        Simple Moving Average Crossover Strategy
        
        Buy when fast MA crosses above slow MA
        Sell when fast MA crosses below slow MA
        """
        
        def __init__(self, fast_period: int = 10, slow_period: int = 30):
            self.fast_period = fast_period
            self.slow_period = slow_period
            
        def generate_signal(self, symbol: str, context: dict) -> Optional[Signal]:
            """Generate trading signal based on MA crossover"""
            prices = context.get("prices", [])
            
            if len(prices) < self.slow_period + 1:
                return None
            
            # Calculate moving averages
            fast_ma = np.mean(prices[-self.fast_period:])
            slow_ma = np.mean(prices[-self.slow_period:])
            
            # Previous bar values
            prev_fast_ma = np.mean(prices[-self.fast_period-1:-1])
            prev_slow_ma = np.mean(prices[-self.slow_period-1:-1])
            
            # Crossover detection
            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
                return Signal(action="buy", confidence=0.8)
            elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
                return Signal(action="sell", confidence=0.8)
            
            return Signal(action="hold", confidence=0.0)
    
    async def run_example_backtest():
        """Run an example backtest"""
        print("=" * 60)
        print("AI Trading System - Backtest Engine Example")
        print("=" * 60)
        
        # Configure backtest
        config = BacktestConfig(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_model="fixed",
            slippage_pct=0.0005,
            latency_ms=100
        )
        
        # Create engine
        engine = BacktestEngine(config=config)
        
        # Create strategy
        strategy = SimpleMAStrategy(fast_period=10, slow_period=30)
        
        # Run backtest
        print("\nRunning backtest on BTC-USD...")
        result = await engine.run(
            strategy=strategy,
            symbols=["BTC-USD"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            interval="1h"
        )
        
        # Print results
        print(f"\nReturn: {result.total_return_pct:.2f}%")
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Max DD: {result.max_drawdown_pct:.2f}%")
        print(f"Win Rate: {result.win_rate:.1f}%")
        
    if __name__ == "__main__":
        asyncio.run(run_example_backtest())
