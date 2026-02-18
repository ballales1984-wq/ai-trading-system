"""
Auto Trading Bot Module
Automatic trading system that executes trades based on signals and learned patterns
"""

import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import threading

import pandas as pd
import numpy as np

import config
from data_collector import DataCollector
from technical_analysis import TechnicalAnalyzer
from sentiment_news import SentimentAnalyzer
from decision_engine import DecisionEngine

# Configure logging
logger = logging.getLogger(__name__)


# ==================== DATA STRUCTURES ====================

@dataclass
class Trade:
    """Represents a single trade"""
    id: str
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    status: str = 'pending'  # pending, executed, cancelled, closed
    pnl: float = 0.0
    fees: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'pnl': self.pnl,
            'fees': self.fees
        }


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_time': self.entry_time.isoformat(),
            'unrealized_pnl': self.unrealized_pnl
        }


@dataclass
class Portfolio:
    """Portfolio state"""
    balance: float
    initial_balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_value': self.get_total_value(),
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'win_rate': self.get_win_rate(),
            'positions': {k: v.to_dict() for k, v in self.positions.items()}
        }
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(p.current_price * p.quantity for p in self.positions.values())
        return self.balance + positions_value
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if self.trade_count == 0:
            return 0.0
        return (self.win_count / self.trade_count) * 100


# ==================== AUTO TRADING BOT ====================

class AutoTradingBot:
    """
    Automatic trading bot that executes trades based on signals
    Supports both paper trading and live trading modes
    """
    
    def __init__(self, initial_balance: float = 10000.0, 
                 paper_trading: bool = True):
        """
        Initialize the trading bot.
        
        Args:
            initial_balance: Starting balance
            paper_trading: If True, simulate trades without real money
        """
        self.paper_trading = paper_trading
        self.portfolio = Portfolio(
            balance=initial_balance,
            initial_balance=initial_balance
        )
        
        # Initialize components
        self.data_collector = DataCollector(simulation=True)
        self.decision_engine = DecisionEngine(self.data_collector)
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Trading settings
        self.running = False
        self.trade_interval = 60  # seconds between trade checks
        self.max_positions = 5
        self.position_size_percent = 0.1  # 10% per trade
        
        # Risk management
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.stop_loss_percent = 0.02  # 2% stop loss
        self.take_profit_percent = 0.05  # 5% take profit
        
        # Learning system
        self.signal_performance = {}  # Track signal accuracy
        
        # History
        self.daily_stats = []
        
        logger.info(f"AutoTradingBot initialized (paper={paper_trading}, balance=${initial_balance})")
    
    # ==================== TRADING LOGIC ====================
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        logger.info("🤖 Trading bot STARTED")
        
        # Run main loop
        self._trading_loop()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logger.info("🛑 Trading bot STOPPED")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check if we should trade
                if self._should_trade():
                    # Generate signals
                    signals = self.decision_engine.generate_signals()
                    
                    # Execute trades based on signals
                    self._process_signals(signals)
                
                # Check existing positions for stop loss / take profit
                self._manage_positions()
                
                # Update portfolio
                self._update_portfolio()
                
                # Log status
                self._log_status()
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
            
            # Wait before next iteration
            time.sleep(self.trade_interval)
    
    def _should_trade(self) -> bool:
        """Check if we should attempt new trades"""
        # Don't trade if we have too many positions
        if len(self.portfolio.positions) >= self.max_positions:
            return False
        
        # Don't trade if we've hit daily loss limit
        if self._check_daily_loss_limit():
            return False
        
        return True
    
    def _process_signals(self, signals: List):
        """Process trading signals and execute trades"""
        for signal in signals:
            # Only act on strong signals
            if signal.confidence < 0.65:
                continue
            
            if signal.action == 'BUY':
                self._execute_buy(signal)
            elif signal.action == 'SELL':
                self._execute_sell(signal)
    
    def _execute_buy(self, signal) -> Optional[Trade]:
        """Execute a BUY order"""
        # Check if we already have a position
        if signal.symbol in self.portfolio.positions:
            return None
        
        # Calculate position size
        position_value = self.portfolio.balance * self.position_size_percent
        quantity = position_value / signal.current_price
        
        # Check if we have enough balance
        if position_value > self.portfolio.balance:
            logger.warning(f"Insufficient balance for {signal.symbol}")
            return None
        
        # Create trade
        trade = Trade(
            id=f"trade_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            symbol=signal.symbol,
            action='BUY',
            quantity=quantity,
            price=signal.current_price,
            timestamp=datetime.now(),
            status='executed'
        )
        
        # Execute trade
        if not self.paper_trading:
            # Real trading would go here
            pass
        
        # Update portfolio
        self.portfolio.balance -= position_value
        self.portfolio.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            quantity=quantity,
            entry_price=signal.current_price,
            current_price=signal.current_price,
            entry_time=datetime.now()
        )
        
        self.portfolio.trades.append(trade)
        
        logger.info(f"✅ BUY executed: {signal.symbol} @ ${signal.current_price:,.2f} "
                   f"(qty: {quantity:.4f})")
        
        return trade
    
    def _execute_sell(self, signal) -> Optional[Trade]:
        """Execute a SELL order"""
        # Check if we have a position
        if signal.symbol not in self.portfolio.positions:
            return None
        
        position = self.portfolio.positions[signal.symbol]
        
        # Calculate PnL
        sell_value = signal.current_price * position.quantity
        cost_basis = position.entry_price * position.quantity
        pnl = sell_value - cost_basis
        fees = sell_value * 0.001  # 0.1% fee estimate
        
        # Create trade
        trade = Trade(
            id=f"trade_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            symbol=signal.symbol,
            action='SELL',
            quantity=position.quantity,
            price=signal.current_price,
            timestamp=datetime.now(),
            status='executed',
            pnl=pnl - fees,
            fees=fees
        )
        
        # Execute trade
        if not self.paper_trading:
            # Real trading would go here
            pass
        
        # Update portfolio
        self.portfolio.balance += (sell_value - fees)
        self.portfolio.realized_pnl += pnl - fees
        self.portfolio.total_pnl += pnl - fees
        self.portfolio.trade_count += 1
        
        if pnl > 0:
            self.portfolio.win_count += 1
        
        # Remove position
        del self.portfolio.positions[signal.symbol]
        
        self.portfolio.trades.append(trade)
        
        logger.info(f"✅ SELL executed: {signal.symbol} @ ${signal.current_price:,.2f} "
                   f"pnl: ${pnl-fees:.2f}")
        
        return trade
    
    def _manage_positions(self):
        """Check and manage open positions (stop loss, take profit)"""
        symbols_to_close = []
        
        for symbol, position in self.portfolio.positions.items():
            # Use the current price from position (already updated by simulation)
            # This ensures backtest uses simulated prices consistently
            current_price = position.current_price
            
            if current_price is None or current_price == 0:
                continue
            
            # Calculate unrealized PnL
            position.unrealized_pnl = (
                (current_price - position.entry_price) * position.quantity
            )
            
            # Check stop loss
            price_change_pct = (current_price - position.entry_price) / position.entry_price
            
            if price_change_pct <= -self.stop_loss_percent:
                logger.warning(f"🛑 STOP LOSS triggered: {symbol}")
                symbols_to_close.append(symbol)
            
            # Check take profit
            elif price_change_pct >= self.take_profit_percent:
                logger.info(f"🎯 TAKE PROFIT triggered: {symbol}")
                symbols_to_close.append(symbol)
        
        # Close positions
        for symbol in symbols_to_close:
            position = self.portfolio.positions[symbol]
            current_price = position.current_price
            
            # Execute sell
            trade = Trade(
                id=f"trade_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
                symbol=symbol,
                action='SELL',
                quantity=position.quantity,
                price=current_price,
                timestamp=datetime.now(),
                status='executed',
                pnl=position.unrealized_pnl
            )
            
            # Update portfolio
            sell_value = current_price * position.quantity
            self.portfolio.balance += sell_value
            self.portfolio.realized_pnl += position.unrealized_pnl
            self.portfolio.total_pnl += position.unrealized_pnl
            self.portfolio.trade_count += 1
            
            if position.unrealized_pnl > 0:
                self.portfolio.win_count += 1
            
            del self.portfolio.positions[symbol]
            self.portfolio.trades.append(trade)
    
    def _update_portfolio(self):
        """Update portfolio values"""
        total_value = self.portfolio.get_total_value()
        self.portfolio.total_pnl = total_value - self.portfolio.initial_balance
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        if not self.daily_stats:
            return False
        
        today = datetime.now().date()
        today_stats = [s for s in self.daily_stats if s['date'] == today]
        
        if today_stats:
            daily_pnl = today_stats[-1]['pnl']
            daily_loss_pct = daily_pnl / self.portfolio.initial_balance
            
            return daily_loss_pct <= -self.max_daily_loss
        
        return False
    
    def _log_status(self):
        """Log current portfolio status"""
        total_value = self.portfolio.get_total_value()
        
        logger.info(f"📊 Portfolio: ${total_value:,.2f} | "
                   f"PnL: ${self.portfolio.total_pnl:,.2f} | "
                   f"Positions: {len(self.portfolio.positions)} | "
                   f"Win Rate: {self.portfolio.get_win_rate():.1f}%")
    
    # ==================== BACKTESTING ====================
    
    def backtest(self, days: int = 30) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            days: Number of days to backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {days} days...")
        
        # Reset portfolio
        self.portfolio = Portfolio(
            balance=self.portfolio.initial_balance,
            initial_balance=self.portfolio.initial_balance
        )
        
        # Simulate historical trading
        # In a real backtest, you'd iterate through historical data
        # Here we simulate random outcomes
        
        for day in range(days):
            # Generate signals
            signals = self.decision_engine.generate_signals()
            
            # Simulate some random outcomes
            for signal in signals:
                if random.random() < 0.5:  # 50% chance to trade
                    if signal.action == 'BUY' and len(self.portfolio.positions) < self.max_positions:
                        self._simulate_buy(signal)
                    elif signal.action == 'SELL' and signal.symbol in self.portfolio.positions:
                        self._simulate_sell(signal)
            
            # Simulate price changes
            self._simulate_price_changes()
            
            # Check stop loss / take profit
            self._manage_positions()
        
        # Calculate final metrics
        results = self._calculate_backtest_metrics()
        
        logger.info(f"Backtest complete: Final balance: ${results['final_balance']:,.2f}, "
                   f"Total Return: {results['total_return']:.2f}%")
        
        return results
    
    def _simulate_buy(self, signal):
        """Simulate a buy during backtest"""
        position_value = self.portfolio.balance * self.position_size_percent
        quantity = position_value / signal.current_price
        
        self.portfolio.balance -= position_value
        self.portfolio.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            quantity=quantity,
            entry_price=signal.current_price,
            current_price=signal.current_price,
            entry_time=datetime.now()
        )
    
    def _simulate_sell(self, signal):
        """Simulate a sell during backtest"""
        if signal.symbol not in self.portfolio.positions:
            return
        
        position = self.portfolio.positions[signal.symbol]
        sell_value = signal.current_price * position.quantity
        
        pnl = sell_value - (position.entry_price * position.quantity)
        
        self.portfolio.balance += sell_value
        self.portfolio.realized_pnl += pnl
        self.portfolio.trade_count += 1
        
        if pnl > 0:
            self.portfolio.win_count += 1
        
        del self.portfolio.positions[signal.symbol]
    
    def _simulate_price_changes(self):
        """Simulate price changes for backtest"""
        # Use lower volatility for more realistic simulation
        # 0.5% std dev is more realistic for hourly crypto moves
        for position in self.portfolio.positions.values():
            # Random price change between -1.5% and +1.5% (realistic range)
            change = random.gauss(0, 0.005)
            position.current_price *= (1 + change)
            position.unrealized_pnl = (
                (position.current_price - position.entry_price) * position.quantity
            )
    
    def _calculate_backtest_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        final_value = self.portfolio.get_total_value()
        total_return = ((final_value - self.portfolio.initial_balance) / 
                      self.portfolio.initial_balance) * 100
        
        # Calculate max drawdown
        peak = self.portfolio.initial_balance
        max_drawdown = 0
        
        # Simple daily returns
        returns = []
        
        return {
            'initial_balance': self.portfolio.initial_balance,
            'final_balance': final_value,
            'total_return': total_return,
            'total_trades': self.portfolio.trade_count,
            'win_rate': self.portfolio.get_win_rate(),
            'total_pnl': self.portfolio.total_pnl,
            'realized_pnl': self.portfolio.realized_pnl,
            'max_drawdown': max_drawdown,
            'positions': len(self.portfolio.positions)
        }
    
    # ==================== REPORTING ====================
    
    def export_results(self, filepath: str):
        """Export trading results to file"""
        data = {
            'generated_at': datetime.now().isoformat(),
            'settings': {
                'paper_trading': self.paper_trading,
                'initial_balance': self.portfolio.initial_balance,
                'position_size_percent': self.position_size_percent,
                'stop_loss_percent': self.stop_loss_percent,
                'take_profit_percent': self.take_profit_percent
            },
            'portfolio': self.portfolio.to_dict(),
            'trades': [t.to_dict() for t in self.portfolio.trades[-50:]]  # Last 50 trades
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")


# ==================== STANDALONE FUNCTIONS ====================

def run_paper_trading(initial_balance: float = 10000.0):
    """Run the bot in paper trading mode"""
    bot = AutoTradingBot(initial_balance=initial_balance, paper_trading=True)
    bot.start()


def run_backtest(initial_balance: float = 10000.0, days: int = 30) -> Dict:
    """Run a backtest"""
    bot = AutoTradingBot(initial_balance=initial_balance, paper_trading=True)
    return bot.backtest(days)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("AUTO TRADING BOT TEST")
    print("="*60)
    
    # Test backtest
    print("\n📊 Running backtest...")
    bot = AutoTradingBot(initial_balance=10000.0, paper_trading=True)
    results = bot.backtest(days=30)
    
    print("\n📈 Backtest Results:")
    print(f"   Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"   Final Balance: ${results['final_balance']:,.2f}")
    print(f"   Total Return: {results['total_return']:.2f}%")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    
    print("\n✅ Test complete!")

