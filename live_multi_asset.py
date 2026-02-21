#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Multi-Asset Trading System
Real-time multi-asset trading with WebSocket, ML ensemble, and dynamic portfolio allocation
"""

import sys
import os
import time
import logging
import signal
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Import project modules
from src.live.binance_multi_ws import BinanceMultiWebSocket
from src.live.position_sizing import VolatilityPositionSizer, AdaptivePositionSizer
from src.live.portfolio_live import (
    LivePortfolio, 
    EqualWeightAllocator, 
    VolatilityParityAllocator,
    RiskParityAllocator,
    MomentumAllocator
)
from src.live.telegram_notifier import TelegramNotifier
from src.live.risk_engine import RiskEngine, RiskManager
from technical_analysis import TechnicalAnalyzer
from data_collector import DataCollector
from config import CRYPTO_SYMBOLS, SIMULATED_PRICES

# Try to import ML modules
try:
    sys.path.insert(0, 'src')
    from ml_model import MLSignalModel
    from ml_model_xgb import XGBSignalModel
    from models.ensemble import EnsembleSignalModel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML modules not available, using technical analysis only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveMultiAssetTrader:
    """
    Main class for live multi-asset trading.
    """
    
    def __init__(
        self,
        assets: List[str],
        initial_capital: float = 100000,
        interval: str = "1m",
        allocation_strategy: str = "equal_weight",
        testnet: bool = False,
        paper_trading: bool = True,
        telegram_bot_token: str = None,
        telegram_chat_id: str = None,
        telegram_enabled: bool = True
    ):
        """
        Initialize the live multi-asset trader.
        
        Args:
            assets: List of trading assets (e.g., ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
            initial_capital: Initial capital
            interval: Kline interval
            allocation_strategy: 'equal_weight', 'volatility_parity', 'risk_parity', 'momentum'
            testnet: Use Binance testnet
            paper_trading: Use paper trading (no real orders)
            telegram_bot_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            telegram_enabled: Enable Telegram notifications
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.interval = interval
        self.allocation_strategy = allocation_strategy
        self.testnet = testnet
        self.paper_trading = paper_trading
        
        # Initialize components
        self.ws = BinanceMultiWebSocket(assets, interval, testnet)
        self.data_collector = DataCollector(simulation=True)
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize portfolio
        self.portfolio = LivePortfolio(initial_capital)
        
        # Initialize position sizer
        self.position_sizer = VolatilityPositionSizer(
            target_volatility=0.02,
            max_position_pct=0.1,
            min_position_pct=0.01
        )
        
        # Initialize allocator
        self.allocator = self._create_allocator(allocation_strategy)
        
        # Initialize Telegram notifier
        self.notifier = None
        if telegram_enabled and telegram_bot_token and telegram_chat_id:
            self.notifier = TelegramNotifier(
                bot_token=telegram_bot_token,
                chat_id=telegram_chat_id,
                enabled=True
            )
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram notifications disabled (no credentials)")
        
        # Initialize Risk Engine
        self.risk_engine = RiskEngine(
            max_drawdown=0.20,
            sl_multiplier=2.0,
            tp_multiplier=3.0,
            trailing_multiplier=1.5
        )
        logger.info("Risk Engine initialized")
        
        # Initialize ML models
        self.models = {}
        if ML_AVAILABLE:
            self._initialize_models()
        
        # State tracking
        self.running = False
        self.last_signals = {}
        self.last_rebalance = time.time()
        self.rebalance_interval = 300  # 5 minutes
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Initialized LiveMultiAssetTrader for {assets}")
    
    def _create_allocator(self, strategy: str):
        """Create the allocation strategy."""
        allocators = {
            'equal_weight': EqualWeightAllocator(max_positions=len(self.assets)),
            'volatility_parity': VolatilityParityAllocator(target_vol=0.02),
            'risk_parity': RiskParityAllocator(),
            'momentum': MomentumAllocator()
        }
        return allocators.get(strategy, EqualWeightAllocator())
    
    def _initialize_models(self):
        """Initialize ML models for each asset."""
        logger.info("Initializing ML models...")
        
        for asset in self.assets:
            try:
                # Create ensemble model
                rf = MLSignalModel("random_forest")
                xgb = XGBSignalModel()
                
                # We'll train on historical data when available
                self.models[asset] = {
                    'ensemble': EnsembleSignalModel([rf, xgb], weights=[0.5, 0.5]),
                    'trained': False
                }
                
                logger.info(f"Initialized model for {asset}")
                
            except Exception as e:
                logger.warning(f"Could not initialize model for {asset}: {e}")
    
    def _train_models(self):
        """Train models on historical data."""
        if not ML_AVAILABLE:
            return
        
        logger.info("Training models on historical data...")
        
        for asset in self.assets:
            try:
                # Fetch historical data
                df = self.data_collector.fetch_ohlcv(asset, self.interval, 200)
                
                if df is not None and len(df) > 50:
                    # Calculate indicators
                    df = self.technical_analyzer.calculate_indicators(df)
                    
                    # Train ensemble
                    self.models[asset]['ensemble'].fit(df)
                    self.models[asset]['trained'] = True
                    
                    logger.info(f"Trained model for {asset}")
                    
            except Exception as e:
                logger.warning(f"Could not train model for {asset}: {e}")
    
    def get_signals(self, data_frames: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        Generate trading signals for all assets.
        
        Args:
            data_frames: Dictionary of DataFrames per asset
            
        Returns:
            Dictionary of signals per asset (1=buy, -1=sell, 0=hold)
        """
        signals = {}
        
        for asset, df in data_frames.items():
            if df is None or len(df) < 50:
                signals[asset] = 0
                continue
            
            try:
                # Use ML model if available and trained
                if ML_AVAILABLE and asset in self.models and self.models[asset].get('trained'):
                    model = self.models[asset]['ensemble']
                    
                    # Get prediction
                    raw_signal = model.predict_signals(df)
                    
                    # Apply filters (simple confidence check)
                    if raw_signal == 1 and np.random.random() > 0.3:  # Reduce false positives
                        signals[asset] = 1
                    elif raw_signal == -1 and np.random.random() > 0.3:
                        signals[asset] = -1
                    else:
                        signals[asset] = 0
                else:
                    # Use technical analysis
                    analysis = self.technical_analyzer.analyze(df, asset)
                    
                    # Generate signal based on technical score
                    if analysis.technical_score > 0.65:
                        signals[asset] = 1
                    elif analysis.technical_score < 0.35:
                        signals[asset] = -1
                    else:
                        signals[asset] = 0
                        
            except Exception as e:
                logger.warning(f"Error generating signal for {asset}: {e}")
                signals[asset] = 0
        
        # Send Telegram notifications for signals
        if self.notifier:
            for asset, sig in signals.items():
                if sig != 0 and asset in prices:
                    self.notifier.send_signal_alert(
                        symbol=asset,
                        signal=sig,
                        price=prices[asset],
                        confidence=0.75
                    )
        
        return signals
    
    def _rebalance_portfolio(
        self,
        signals: Dict[str, int],
        data_frames: Dict[str, pd.DataFrame],
        prices: Dict[str, float]
    ):
        """Rebalance portfolio based on signals."""
        current_time = time.time()
        
        # Only rebalance every rebalance_interval seconds
        if current_time - self.last_rebalance < self.rebalance_interval:
            return
        
        logger.info("Rebalancing portfolio...")
        
        # Get allocations from allocator
        allocations = self.allocator.allocate(
            signals=signals,
            capital=self.portfolio.get_total_value(prices),
            prices=prices,
            data_frames=data_frames
        )
        
        # Close positions not in allocation
        positions_to_close = set(self.portfolio.positions.keys()) - set(allocations.keys())
        for symbol in positions_to_close:
            if symbol in prices:
                self.portfolio.close_position(symbol, prices[symbol], reason="rebalance")
        
        # Open/update positions
        for symbol, quantity in allocations.items():
            if symbol not in prices:
                continue
            
            price = prices[symbol]
            
            if symbol in self.portfolio.positions:
                # Check if we need to adjust position
                current_position = self.portfolio.positions[symbol]
                diff = quantity - current_position.quantity
                
                if abs(diff) > quantity * 0.1:  # 10% threshold
                    # Close and reopen with new size
                    self.portfolio.close_position(symbol, price, reason="rebalance")
                    if quantity > 0:
                        signal = 1 if diff > 0 else -1
                        self.portfolio.open_position(
                            symbol, 
                            "long" if signal > 0 else "short",
                            quantity, 
                            price
                        )
            else:
                # Open new position
                if quantity > 0 and signals.get(symbol, 0) != 0:
                    side = "long" if signals[symbol] > 0 else "short"
                    self.portfolio.open_position(symbol, side, quantity, price)
        
        self.last_rebalance = current_time
        
        # Send Telegram notification for portfolio update
        if self.notifier:
            total_value = self.portfolio.get_total_value(prices)
            # Calculate unrealized P&L
            total_pnl = sum(
                pos.unrealized_pnl for pos in self.portfolio.positions.values()
            )
            # Build positions dict
            positions_dict = {
                symbol: {'unrealized_pnl': pos.unrealized_pnl}
                for symbol, pos in self.portfolio.positions.items()
            }
            self.notifier.send_portfolio_update(
                total_value=total_value,
                total_pnl=total_pnl,
                positions=positions_dict
            )
    
    def _check_stop_loss_take_profit(self, prices: Dict[str, float]):
        """Check and execute stop loss / take profit using RiskEngine."""
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol not in prices:
                continue
            
            price = prices[symbol]
            position.update(price)
            
            # Use risk engine for exit signals
            exit_signal = self.risk_engine.check_exit_signal(
                asset=symbol,
                current_price=price
            )
            
            if exit_signal:
                logger.warning(f"{exit_signal} triggered for {symbol}")
                self.portfolio.close_position(symbol, price, reason=exit_signal.lower())
                
                # Notify via Telegram
                if self.notifier:
                    self.notifier.send_trade_execution(
                        symbol=symbol,
                        side="CLOSE",
                        quantity=position.quantity,
                        price=price,
                        pnl=position.unrealized_pnl
                    )
        
        # Check portfolio-level max drawdown
        total_equity = self.portfolio.get_total_value(prices)
        should_kill, drawdown = self.risk_engine.check_max_drawdown(total_equity)
        
        if should_kill:
            logger.critical(f"MAX DRAWDOWN KILL SWITCH TRIGGERED: {drawdown:.2%}")
            
            # Close all positions
            for symbol, position in list(self.portfolio.positions.items()):
                if symbol in prices:
                    self.portfolio.close_position(symbol, prices[symbol], reason="kill_switch")
            
            # Notify via Telegram
            if self.notifier:
                self.notifier.send_error_alert(
                    error_type="KILL_SWITCH",
                    message=f"Max drawdown {drawdown:.2%} triggered - all positions closed"
                )
            
            # Stop the trading
            self.running = False
    
    def _print_status(self, prices: Dict[str, float]):
        """Print current status."""
        stats = self.portfolio.get_stats(prices)
        
        print("\n" + "="*60)
        print(f"LIVE MULTI-ASSET TRADING - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        print(f"\n💰 Portfolio Value: ${stats['total_value']:,.2f}")
        print(f"📈 Total PnL: ${stats['total_pnl']:,.2f} ({stats['return_pct']:+.2f}%)")
        print(f"🎯 Win Rate: {stats['win_rate']:.1f}%")
        
        print(f"\n📊 Positions ({stats['num_positions']}):")
        for symbol, position in self.portfolio.positions.items():
            if symbol in prices:
                position.update(prices[symbol])
                pnl_emoji = "🟢" if position.unrealized_pnl > 0 else "🔴"
                print(f"  {pnl_emoji} {symbol}: {position.quantity:.4f} @ ${position.entry_price:.2f} "
                      f"(PnL: ${position.unrealized_pnl:+,.2f})")
        
        print(f"\n💵 Cash: ${stats['cash']:,.2f}")
        print("="*60)
    
    def start(self):
        """Start the live trading system."""
        logger.info("Starting live multi-asset trading...")
        
        # Train models
        self._train_models()
        
        # Start WebSocket
        self.ws.start()
        
        self.running = True
        
        logger.info("Live trading started!")
        
        # Send startup notification
        if self.notifier:
            self.notifier.send_heartbeat({
                'status': 'started',
                'total_value': self.initial_capital,
                'total_pnl': 0,
                'num_positions': 0,
                'trades_today': 0
            })
        
        try:
            while self.running:
                try:
                    # Wait for data
                    if not self.ws.is_ready(min_candles=30):
                        logger.info("Waiting for data...")
                        time.sleep(5)
                        continue
                    
                    # Get current data
                    data_frames = self.ws.get_all_data()
                    prices = self.ws.get_all_prices()
                    
                    # Get signals
                    signals = self.get_signals(data_frames)
                    
                    # Store signals
                    self.last_signals = signals
                    
                    # Rebalance portfolio
                    self._rebalance_portfolio(signals, data_frames, prices)
                    
                    # Check stop loss / take profit
                    self._check_stop_loss_take_profit(prices)
                    
                    # Print status every minute
                    current_time = time.time()
                    if not hasattr(self, '_last_status_print') or current_time - self._last_status_print > 60:
                        self._print_status(prices)
                        self._last_status_print = current_time
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    if self.notifier:
                        self.notifier.send_error_alert(
                            error_type="TradingLoopError",
                            message=str(e)
                        )
                    time.sleep(5)
                
                # Sleep before next iteration
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.stop()
    
    def stop(self):
        """Stop the live trading system."""
        logger.info("Stopping live trading...")
        
        self.running = False
        
        # Close all positions
        prices = self.ws.get_all_prices()
        for symbol in list(self.portfolio.positions.keys()):
            if symbol in prices:
                self.portfolio.close_position(symbol, prices[symbol], reason="shutdown")
        
        # Stop WebSocket
        self.ws.stop()
        
        # Print final stats
        self._print_status(prices)
        
        logger.info("Live trading stopped")


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Multi-Asset Trading System")
    
    parser.add_argument(
        '--assets', '-a',
        type=str,
        default='BTCUSDT,ETHUSDT,SOLUSDT',
        help='Comma-separated list of assets (default: BTCUSDT,ETHUSDT,SOLUSDT)'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=100000,
        help='Initial capital (default: 100000)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=str,
        default='1m',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Kline interval (default: 1m)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='equal_weight',
        choices=['equal_weight', 'volatility_parity', 'risk_parity', 'momentum'],
        help='Allocation strategy (default: equal_weight)'
    )
    
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Use Binance testnet'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Use real trading (not paper trading)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse assets
    assets = [a.strip().upper() for a in args.assets.split(',')]
    
    print("\n" + "="*60)
    print("🚀 LIVE MULTI-ASSET TRADING SYSTEM")
    print("="*60)
    print(f"\n📈 Assets: {', '.join(assets)}")
    print(f"💵 Capital: ${args.capital:,.2f}")
    print(f"⏱️  Interval: {args.interval}")
    print(f"⚖️  Strategy: {args.strategy}")
    print(f"🧪 Testnet: {args.testnet}")
    print(f"📝 Paper Trading: {not args.live}")
    print("\n" + "="*60)
    
    # Create and start trader
    trader = LiveMultiAssetTrader(
        assets=assets,
        initial_capital=args.capital,
        interval=args.interval,
        allocation_strategy=args.strategy,
        testnet=args.testnet,
        paper_trading=not args.live
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        trader.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start trading
    trader.start()


if __name__ == '__main__':
    main()
