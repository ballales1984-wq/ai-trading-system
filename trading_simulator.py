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
        self.equity_history = []  # For portfolio chart
        
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
        # Track equity history
        from datetime import datetime
        self.equity_history.append({
            'timestamp': datetime.now(),
            'value': self.portfolio.get_total_value()
        })
        # Keep only last 100 entries
        if len(self.equity_history) > 100:
            self.equity_history = self.equity_history[-100:]
        
        return {
            'balance': self.portfolio.balance,
            'total_value': self.portfolio.get_total_value(),
            'total_pnl': self.portfolio.total_pnl,
            'positions': {s: asdict(p) for s, p in self.portfolio.positions.items()},
            'trade_count': self.portfolio.trade_count,
            'win_rate': self.portfolio.get_win_rate()
        }
    
    def get_equity_history(self) -> List[Dict]:
        """Get portfolio equity history for charting"""
        return self.equity_history
    
    # ==================== PORTFOLIO CONTROL FUNCTIONS ====================
    
    def check_portfolio(self) -> Dict:
        """
        Check portfolio status - returns detailed portfolio information.
        
        Returns:
            Dict with portfolio status details
        """
        state = self.get_portfolio_state()
        
        # Calculate additional metrics
        initial = self.portfolio.initial_balance
        current = state['total_value']
        pnl_percent = ((current - initial) / initial) * 100 if initial > 0 else 0
        
        return {
            'status': 'OK',
            'balance': state['balance'],
            'total_value': current,
            'initial_balance': initial,
            'total_pnl': state['total_pnl'],
            'pnl_percent': pnl_percent,
            'open_positions': len(state['positions']),
            'max_positions': self.max_positions,
            'trade_count': state['trade_count'],
            'win_rate': state['win_rate'],
            'positions_detail': state['positions'],
            'settings': {
                'position_size_percent': self.position_size_percent * 100,
                'stop_loss_percent': self.stop_loss_percent * 100,
                'take_profit_percent': self.take_profit_percent * 100,
            }
        }
    
    def close_position(self, symbol: str) -> Dict:
        """
        Manually close an open position.
        
        Args:
            symbol: Trading symbol to close
            
        Returns:
            Dict with close result
        """
        if symbol not in self.portfolio.positions:
            return {'status': 'ERROR', 'message': f'No open position for {symbol}'}
        
        position = self.portfolio.positions[symbol]
        
        # Get current price
        try:
            df = self.data_collector.fetch_ohlcv(symbol, '1h', 1)
            current_price = df['close'].iloc[-1] if df is not None else position.current_price
        except:
            current_price = position.current_price
        
        # Calculate P&L
        pnl = (current_price - position.entry_price) * position.quantity
        self.portfolio.balance += (current_price * position.quantity)
        self.portfolio.total_pnl += pnl
        
        # Record trade
        trade = Trade(
            id=f"{datetime.now().timestamp()}",
            symbol=symbol,
            action='SELL',
            quantity=position.quantity,
            price=current_price,
            timestamp=datetime.now(),
            pnl=pnl
        )
        self.portfolio.trades.append(trade)
        self.portfolio.trade_count += 1
        if pnl > 0:
            self.portfolio.win_count += 1
        
        # Remove position
        del self.portfolio.positions[symbol]
        
        return {
            'status': 'SUCCESS',
            'message': f'Closed position for {symbol}',
            'price': current_price,
            'quantity': position.quantity,
            'pnl': pnl
        }
    
    def close_all_positions(self) -> Dict:
        """
        Close all open positions.
        
        Returns:
            Dict with close results
        """
        closed = []
        for symbol in list(self.portfolio.positions.keys()):
            result = self.close_position(symbol)
            if result['status'] == 'SUCCESS':
                closed.append(result)
        
        return {
            'status': 'SUCCESS',
            'closed_positions': closed,
            'total_closed': len(closed)
        }
    
    def set_stop_loss(self, percent: float) -> Dict:
        """
        Set stop-loss percentage.
        
        Args:
            percent: Stop-loss percentage (e.g., 2.0 for 2%)
            
        Returns:
            Dict with setting result
        """
        if percent < 0 or percent > 50:
            return {'status': 'ERROR', 'message': 'Stop-loss must be between 0% and 50%'}
        
        self.stop_loss_percent = percent / 100
        
        return {
            'status': 'SUCCESS',
            'message': f'Stop-loss set to {percent}%',
            'stop_loss_percent': percent
        }
    
    def set_take_profit(self, percent: float) -> Dict:
        """
        Set take-profit percentage.
        
        Args:
            percent: Take-profit percentage (e.g., 5.0 for 5%)
            
        Returns:
            Dict with setting result
        """
        if percent < 0 or percent > 100:
            return {'status': 'ERROR', 'message': 'Take-profit must be between 0% and 100%'}
        
        self.take_profit_percent = percent / 100
        
        return {
            'status': 'SUCCESS',
            'message': f'Take-profit set to {percent}%',
            'take_profit_percent': percent
        }
    
    def set_position_size(self, percent: float) -> Dict:
        """
        Set position size percentage.
        
        Args:
            percent: Position size percentage (e.g., 20.0 for 20%)
            
        Returns:
            Dict with setting result
        """
        if percent < 1 or percent > 50:
            return {'status': 'ERROR', 'message': 'Position size must be between 1% and 50%'}
        
        self.position_size_percent = percent / 100
        
        return {
            'status': 'SUCCESS',
            'message': f'Position size set to {percent}%',
            'position_size_percent': percent
        }
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        trades = self.portfolio.trades[-limit:]
        return [asdict(t) for t in trades]
    
    def reset_portfolio(self) -> Dict:
        """
        Reset portfolio to initial state.
        
        Returns:
            Dict with reset result
        """
        initial = self.portfolio.initial_balance
        self.portfolio = Portfolio(
            balance=initial,
            initial_balance=initial
        )
        
        return {
            'status': 'SUCCESS',
            'message': 'Portfolio reset to initial state',
            'new_balance': initial
        }
    
    def get_portfolio_analysis(self) -> Dict:
        """
        Get detailed portfolio analysis.
        
        Returns:
            Dict with analysis metrics
        """
        if not self.portfolio.trades:
            return {'status': 'NO_DATA', 'message': 'No trades to analyze'}
        
        trades = self.portfolio.trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        return {
            'status': 'OK',
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': self.portfolio.get_win_rate(),
            'total_pnl': self.portfolio.total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'best_trade': max((t.pnl for t in trades), default=0),
            'worst_trade': min((t.pnl for t in trades), default=0),
        }


def run_simulation(duration: int = 300):
    """Run a trading simulation"""
    simulator = TradingSimulator(initial_balance=10000.0)
    simulator.start(duration_seconds=duration)


if __name__ == "__main__":
    run_simulation(duration=60)  # Run for 1 minute demo
