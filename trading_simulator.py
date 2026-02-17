"""
Trading Simulator Module
Simulates automatic trading based on signals
"""

import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np

import config
from data_collector import DataCollector
from technical_analysis import TechnicalAnalyzer
from sentiment_news import SentimentAnalyzer
from decision_engine import DecisionEngine

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record"""
    id: str
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    fees: float = 0.0


@dataclass
class Position:
    """Open position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0


@dataclass
class Portfolio:
    """Portfolio state"""
    balance: float
    initial_balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    
    def get_total_value(self) -> float:
        positions_value = sum(p.current_price * p.quantity for p in self.positions.values())
        return self.balance + positions_value
    
    def get_win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return (self.win_count / self.trade_count) * 100


class TradingSimulator:
    """
    Automatic trading simulator that executes trades based on signals.
    Uses paper trading (no real money).
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize the simulator.
        
        Args:
            initial_balance: Starting balance in USDT
        """
        self.portfolio = Portfolio(
            balance=initial_balance,
            initial_balance=initial_balance
        )
        
        # Components
        self.data_collector = DataCollector(simulation=True)
        self.decision_engine = DecisionEngine(self.data_collector)
        
        # Settings
        self.trade_interval = 60  # seconds between trades
        self.max_positions = 3
        self.position_size_percent = 0.2  # 20% per trade
        self.stop_loss_percent = 0.02  # 2%
        self.take_profit_percent = 0.05  # 5%
        
        # State
        self.running = False
        self.trade_history = []  # For tracking
        
        logger.info(f"TradingSimulator initialized with ${initial_balance}")
    
    def start(self, duration_seconds: int = 300):
        """
        Start the trading simulation.
        
        Args:
            duration_seconds: How long to run
        """
        self.running = True
        start_time = time.time()
        
        print("\n" + "="*60)
        print("  AUTOMATIC TRADING SIMULATOR")
        print("="*60)
        print(f"  Initial Balance: ${self.portfolio.balance:,.2f}")
        print(f"  Max Positions: {self.max_positions}")
        print(f"  Position Size: {self.position_size_percent*100}%")
        print(f"  Stop Loss: {self.stop_loss_percent*100}%")
        print(f"  Take Profit: {self.take_profit_percent*100}%")
        print("="*60 + "\n")
        
        iteration = 0
        while self.running and (time.time() - start_time) < duration_seconds:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Generate signals
            signals = self.decision_engine.generate_signals()
            
            # Process signals
            trades_made = self._process_signals(signals)
            
            # Check existing positions
            self._manage_positions()
            
            # Update portfolio
            self._update_portfolio()
            
            # Print status
            self._print_status()
            
            # Wait
            if self.running and (time.time() - start_time) < duration_seconds:
                time.sleep(self.trade_interval)
        
        # Final results
        self._print_final_results()
    
    def _process_signals(self, signals: List) -> int:
        """Process trading signals"""
        trades_count = 0
        
        for signal in signals:
            # Only trade on strong signals
            if signal.confidence < 0.60:
                continue
            
            if signal.action == 'BUY' and len(self.portfolio.positions) < self.max_positions:
                if self._execute_buy(signal):
                    trades_count += 1
            elif signal.action == 'SELL' and signal.symbol in self.portfolio.positions:
                if self._execute_sell(signal):
                    trades_count += 1
        
        return trades_count
    
    def _execute_buy(self, signal) -> bool:
        """Execute a BUY order"""
        # Check if we already have this position
        if signal.symbol in self.portfolio.positions:
            return False
        
        # Calculate position size
        position_value = self.portfolio.balance * self.position_size_percent
        quantity = position_value / signal.current_price
        
        # Check balance
        if position_value > self.portfolio.balance:
            logger.warning(f"Insufficient balance for {signal.symbol}")
            return False
        
        # Execute trade
        trade = Trade(
            id=f"trade_{len(self.portfolio.trades)+1}",
            symbol=signal.symbol,
            action='BUY',
            quantity=quantity,
            price=signal.current_price,
            timestamp=datetime.now()
        )
        
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
        
        print(f"  BUY: {signal.symbol} @ ${signal.current_price:,.2f} (qty: {quantity:.4f})")
        
        return True
    
    def _execute_sell(self, signal) -> bool:
        """Execute a SELL order"""
        if signal.symbol not in self.portfolio.positions:
            return False
        
        position = self.portfolio.positions[signal.symbol]
        
        # Calculate PnL
        sell_value = signal.current_price * position.quantity
        cost_basis = position.entry_price * position.quantity
        pnl = sell_value - cost_basis
        fees = sell_value * 0.001  # 0.1% fee
        
        # Update portfolio
        self.portfolio.balance += (sell_value - fees)
        self.portfolio.total_pnl += pnl - fees
        self.portfolio.trade_count += 1
        
        if pnl > 0:
            self.portfolio.win_count += 1
        
        # Record trade
        trade = Trade(
            id=f"trade_{len(self.portfolio.trades)+1}",
            symbol=signal.symbol,
            action='SELL',
            quantity=position.quantity,
            price=signal.current_price,
            timestamp=datetime.now(),
            pnl=pnl - fees,
            fees=fees
        )
        
        self.portfolio.trades.append(trade)
        
        print(f"  SELL: {signal.symbol} @ ${signal.current_price:,.2f} | PnL: ${pnl-fees:.2f}")
        
        # Remove position
        del self.portfolio.positions[signal.symbol]
        
        return True
    
    def _manage_positions(self):
        """Check and manage open positions"""
        symbols_to_close = []
        
        for symbol, position in self.portfolio.positions.items():
            # Get current price
            current_price = self.data_collector.fetch_current_price(symbol)
            
            if current_price is None:
                continue
            
            position.current_price = current_price
            
            # Calculate PnL percentage
            price_change_pct = (current_price - position.entry_price) / position.entry_price
            
            # Check stop loss
            if price_change_pct <= -self.stop_loss_percent:
                print(f"  STOP LOSS: {symbol} (-{abs(price_change_pct)*100:.1f}%)")
                symbols_to_close.append((symbol, 'STOP_LOSS'))
            
            # Check take profit
            elif price_change_pct >= self.take_profit_percent:
                print(f"  TAKE PROFIT: {symbol} (+{price_change_pct*100:.1f}%)")
                symbols_to_close.append((symbol, 'TAKE_PROFIT'))
        
        # Close positions
        for symbol, reason in symbols_to_close:
            position = self.portfolio.positions[symbol]
            current_price = position.current_price
            
            pnl = (current_price - position.entry_price) * position.quantity
            fees = current_price * position.quantity * 0.001
            
            self.portfolio.balance += (current_price * position.quantity - fees)
            self.portfolio.total_pnl += pnl - fees
            self.portfolio.trade_count += 1
            
            if pnl > 0:
                self.portfolio.win_count += 1
            
            del self.portfolio.positions[symbol]
    
    def _update_portfolio(self):
        """Update portfolio values"""
        total_value = self.portfolio.get_total_value()
        self.portfolio.total_pnl = total_value - self.portfolio.initial_balance
    
    def _print_status(self):
        """Print current portfolio status"""
        total_value = self.portfolio.get_total_value()
        pnl = self.portfolio.total_pnl
        pnl_pct = (pnl / self.portfolio.initial_balance) * 100
        
        print(f"  📊 Portfolio: ${total_value:,.2f} | PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"     Positions: {len(self.portfolio.positions)} | Trades: {self.portfolio.trade_count}")
        
        if self.portfolio.positions:
            print("     Open Positions:")
            for symbol, pos in self.portfolio.positions.items():
                pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                print(f"       {symbol}: ${pos.current_price:,.2f} ({pnl_pct:+.2f}%)")
    
    def _print_final_results(self):
        """Print final results"""
        total_value = self.portfolio.get_total_value()
        total_return = ((total_value - self.portfolio.initial_balance) / self.portfolio.initial_balance) * 100
        
        print("\n" + "="*60)
        print("  FINAL RESULTS")
        print("="*60)
        print(f"  Initial Balance:  ${self.portfolio.initial_balance:,.2f}")
        print(f"  Final Balance:    ${total_value:,.2f}")
        print(f"  Total Return:     {total_return:+.2f}%")
        print(f"  Total PnL:       ${self.portfolio.total_pnl:,.2f}")
        print(f"  Total Trades:     {self.portfolio.trade_count}")
        print(f"  Win Rate:        {self.portfolio.get_win_rate():.1f}%")
        print("="*60 + "\n")
    
    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state for dashboard"""
        return {
            'balance': self.portfolio.balance,
            'total_value': self.portfolio.get_total_value(),
            'total_pnl': self.portfolio.total_pnl,
            'positions': {s: asdict(p) for s, p in self.portfolio.positions.items()},
            'trade_count': self.portfolio.trade_count,
            'win_rate': self.portfolio.get_win_rate()
        }


def run_simulation(duration: int = 300):
    """Run a trading simulation"""
    simulator = TradingSimulator(initial_balance=10000.0)
    simulator.start(duration_seconds=duration)


if __name__ == "__main__":
    run_simulation(duration=60)  # Run for 1 minute demo
