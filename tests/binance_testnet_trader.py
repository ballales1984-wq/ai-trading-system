"""
Binance Testnet Trader
=====================
Paper trading using Binance Testnet API.

Usage:
    python tests/binance_testnet_trader.py
    
Environment variables needed:
    BINANCE_TESTNET_API_KEY
    BINANCE_TESTNET_API_SECRET
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
import time
from pathlib import Path

def load_env_paper_trading():
    """Load environment variables from .env.paper_trading if exists."""
    env_path = Path(__file__).parent.parent / ".env.paper_trading"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Skip placeholder values
                    if value and not value.startswith('your_') and not value.endswith('_here'):
                        os.environ.setdefault(key, value)
                        
# Load env on import
load_env_paper_trading()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base prices for simulation
BASE_PRICES = {
    'BTCUSDT': 67000,
    'ETHUSDT': 3500,
    'SOLUSDT': 149,
    'BNBUSDT': 580,
    'XRPUSDT': 0.62,
    'ADAUSDT': 0.45
}


class BinanceTestnetTrader:
    """
    Paper trading using Binance Testnet.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        initial_balance: float = 100000.0
    ):
        # Try different env var names
        self.api_key = api_key or os.getenv('BINANCE_API_KEY') or os.getenv('BINANCE_TESTNET_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_TESTNET_API_SECRET')
        self.initial_balance = initial_balance
        
        self.client = None
        self.connected = False
        self.trades: List[Dict] = []
        self.balance = initial_balance
        self.symbol_filters = {}  # Store min quantity per symbol
        
        self._connect()
        if self.connected:
            self._load_symbol_filters()
    
    def _connect(self):
        """Connect to Binance Testnet."""
        try:
            from binance.client import Client
            if self.api_key and self.api_secret:
                self.client = Client(self.api_key, self.api_secret, testnet=True)
                self.client.ping()
                self.connected = True
                logger.info("Connected to Binance Testnet")
            else:
                logger.warning("No API keys - running in simulation mode")
        except ImportError:
            logger.warning("python-binance not installed - simulation mode")
        except Exception as e:
            logger.warning(f"Connection failed: {e}")
    
    def _load_symbol_filters(self):
        """Load symbol filters to get min quantity per symbol."""
        try:
            info = self.client.get_exchange_info()
            for symbol_info in info['symbols']:
                symbol = symbol_info['symbol']
                for f in symbol_info.get('filters', []):
                    if f['filterType'] == 'LOT_SIZE':
                        self.symbol_filters[symbol] = {
                            'minQty': float(f['minQty']),
                            'maxQty': float(f['maxQty']),
                            'stepSize': float(f['stepSize'])
                        }
                        break
            logger.info(f"Loaded filters for {len(self.symbol_filters)} symbols")
        except Exception as e:
            logger.warning(f"Could not load symbol filters: {e}")
    
    def get_balance(self) -> float:
        """Get current USDT balance."""
        if self.connected and self.client:
            try:
                account = self.client.get_account()
                for balance in account['balances']:
                    if balance['asset'] == 'USDT':
                        return float(balance['free'])
            except Exception as e:
                logger.error(f"Error: {e}")
        return self.balance
    
    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Get current price - uses real API if connected, otherwise simulation prices."""
        if self.connected and self.client:
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            except Exception as e:
                logger.error(f"Price error: {e}")
        return BASE_PRICES.get(symbol, 50000)
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET"
    ) -> Optional[Dict]:
        """Place an order."""
        if not self.connected:
            return self._simulate_order(symbol, side, quantity)
        
        try:
            # Apply symbol filters (LOT_SIZE)
            if symbol in self.symbol_filters:
                filters = self.symbol_filters[symbol]
                min_qty = filters['minQty']
                step_size = filters['stepSize']
                # Round to step size and ensure min quantity
                quantity = max(min_qty, quantity)
                quantity = round(quantity / step_size) * step_size
                # Format to remove floating point errors
                quantity = float(f'{quantity:.8f}'.rstrip('0').rstrip('.'))
            else:
                # Fallback: round to 8 decimals
                quantity = round(quantity, 8)
            
            order = self.client.create_order(
                symbol=symbol, side=side, type=order_type, quantity=quantity
            )
            # Get price from order response (for MARKET orders)
            price = float(order.get('price', order.get('cummulativeQuoteAssetQty', 0)))
            self.trades.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol, 'side': side, 'quantity': quantity,
                'price': price, 'order_id': order.get('orderId'), 
                'status': order.get('status'), 'real': True
            })
            logger.info(f"Order: {side} {quantity} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Order error: {e}")
            return self._simulate_order(symbol, side, quantity)
    
    def _simulate_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Simulate an order."""
        price = self.get_current_price(symbol)
        
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol, 'side': side, 'quantity': quantity,
            'price': price, 'simulated': True
        }
        self.trades.append(trade)
        
        cost = quantity * price
        if side == "BUY":
            self.balance -= cost
        else:
            self.balance += cost
        
        logger.info(f"Simulated: {side} {quantity} {symbol} @ ${price}")
        return trade
    
    def get_trades(self) -> List[Dict]:
        return self.trades
    
    def get_performance_summary(self) -> Dict:
        if not self.trades:
            return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                    'win_rate': 0.0, 'total_pnl': 0.0, 'final_balance': self.balance}
        
        buys = [t for t in self.trades if t.get('side') == 'BUY']
        sells = [t for t in self.trades if t.get('side') == 'SELL']
        
        total_buy = sum(t['quantity'] * t['price'] for t in buys)
        total_sell = sum(t['quantity'] * t['price'] for t in sells)
        
        total_pnl = total_sell - total_buy
        winning = 1 if total_pnl > 0 else 0
        losing = 1 if total_pnl < 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': winning / (winning + losing) if (winning + losing) > 0 else 0,
            'total_pnl': total_pnl,
            'final_balance': self.get_balance(),
            'initial_balance': self.initial_balance
        }


def run_testnet_demo():
    """Run a demo."""
    print("\n" + "="*60)
    print("BINANCE TESTNET PAPER TRADING")
    print("="*60 + "\n")
    
    trader = BinanceTestnetTrader(initial_balance=100000)
    print(f"Initial Balance: ${trader.get_balance():,.2f}\n")
    
    # Demo trades
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        price = trader.get_current_price(symbol)
        if price > 0:
            quantity = 1000 / price
            trader.place_order(symbol, "BUY", quantity)
            time.sleep(0.2)
    
    summary = trader.get_performance_summary()
    print("\n--- SUMMARY ---")
    print(f"Trades: {summary['total_trades']}")
    print(f"P&L: ${summary['total_pnl']:,.2f}")
    print(f"Balance: ${summary['final_balance']:,.2f}")
    print("="*60)
    
    return trader


if __name__ == "__main__":
    run_testnet_demo()
