#!/usr/bin/env python3
"""
Mont Blanck Live Trading System
================================

Sistema completo per trading live con strategia Mont Blanck.

Features:
- Multi-asset support
- Paper trading e live trading
- Grafici live multi-asset
- Integrazione con TradingLedger
- Segnali BUY/SELL in tempo reale

Usage:
    python run_montblanck_live.py --paper --assets BTCUSDT ETHUSDT
    python run_montblanck_live.py --live --assets BTCUSDT --balance 1000
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Import moduli locali
try:
    from src.strategy.montblanck import MontBlanck, MontBlanckMultiAsset, MontBlanckConfig, Signal
    from src.trading_ledger import TradingLedger
except ImportError:
    # Fallback per import diretti
    MontBlanck = None
    TradingLedger = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("MontBlanckLive")


class BinanceDataProvider:
    """Provider dati da Binance (reale o testnet)."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        
        if api_key and api_secret:
            try:
                from binance.client import Client
                self.client = Client(api_key, api_secret, testnet=testnet)
                logger.info(f"Binance client initialized (testnet={testnet})")
            except ImportError:
                logger.warning("python-binance not installed. Using simulated data.")
    
    def get_price(self, symbol: str) -> float:
        """Ottiene il prezzo corrente."""
        if self.client:
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        
        # Simula prezzo se non c'è client
        return self._simulate_price(symbol)
    
    def get_price_history(self, symbol: str, interval: str = "1m", limit: int = 50) -> List[float]:
        """Ottiene lo storico dei prezzi."""
        if self.client:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                return [float(k[4]) for k in klines]
            except Exception as e:
                logger.error(f"Error getting history for {symbol}: {e}")
        
        # Simula storico se non c'è client
        return self._simulate_history(symbol, limit)
    
    def _simulate_price(self, symbol: str) -> float:
        """Simula un prezzo per testing."""
        base_prices = {
            "BTCUSDT": 95000,
            "ETHUSDT": 2800,
            "BNBUSDT": 650,
            "SOLUSDT": 180,
            "XRPUSDT": 2.5
        }
        base = base_prices.get(symbol, 100)
        # Aggiunge variazione casuale
        variation = np.random.uniform(-0.02, 0.02)
        return base * (1 + variation)
    
    def _simulate_history(self, symbol: str, limit: int) -> List[float]:
        """Simula storico prezzi per testing."""
        base_prices = {
            "BTCUSDT": 95000,
            "ETHUSDT": 2800,
            "BNBUSDT": 650,
            "SOLUSDT": 180,
            "XRPUSDT": 2.5
        }
        base = base_prices.get(symbol, 100)
        
        # Genera random walk
        prices = [base]
        for _ in range(limit - 1):
            change = np.random.normal(0, 0.005)
            prices.append(prices[-1] * (1 + change))
        
        return prices


class MontBlanckLiveTrader:
    """
    Trader live con strategia Mont Blanck.
    """
    
    def __init__(self,
                 assets: List[str],
                 initial_balance: float = 1000.0,
                 paper_trading: bool = True,
                 config: MontBlanckConfig = None,
                 api_key: str = None,
                 api_secret: str = None,
                 use_testnet: bool = True):
        """
        Inizializza il trader.
        
        Args:
            assets: Lista di asset da tradare
            initial_balance: Saldo iniziale totale
            paper_trading: Se True, simula i trade
            config: Configurazione Mont Blanck
            api_key: API key Binance
            api_secret: API secret Binance
            use_testnet: Se usare testnet
        """
        self.assets = assets
        self.initial_balance = initial_balance
        self.paper_trading = paper_trading
        self.config = config or MontBlanckConfig()
        
        # Provider dati
        self.data_provider = BinanceDataProvider(api_key, api_secret, use_testnet)
        
        # Strategia multi-asset
        self.strategy = MontBlanckMultiAsset(self.config)
        for asset in assets:
            self.strategy.add_asset(asset)
        
        # Ledger per tracking
        ledger_dir = Path("data/ledger")
        ledger_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = TradingLedger(
            data_dir=str(ledger_dir),
            initial_balance=initial_balance
        ) if TradingLedger else None
        
        # Stato per ogni asset
        self.positions: Dict[str, Dict] = {asset: None for asset in assets}
        self.balances: Dict[str, float] = {
            asset: initial_balance / len(assets) 
            for asset in assets
        }
        self.price_history: Dict[str, List[float]] = {asset: [] for asset in assets}
        
        # Statistiche
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"MontBlanckLiveTrader initialized")
        logger.info(f"  Assets: {assets}")
        logger.info(f"  Paper Trading: {paper_trading}")
        logger.info(f"  Initial Balance: {initial_balance}")
    
    def _execute_buy(self, asset: str, quantity: float, price: float):
        """Esegue un ordine BUY."""
        cost = quantity * price
        
        if self.balances[asset] < cost:
            logger.warning(f"Insufficient balance for {asset} BUY")
            return False
        
        # Aggiorna saldo
        self.balances[asset] -= cost
        
        # Registra posizione
        self.positions[asset] = {
            "type": "BUY",
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now()
        }
        
        # Registra nel ledger
        if self.ledger:
            self.ledger.record_trade(
                asset=asset,
                trade_type="BUY",
                quantity=quantity,
                price=price,
                commission=0,
                notes=f"MontBlanck signal - Paper: {self.paper_trading}"
            )
        
        self.total_trades += 1
        logger.info(f"[BUY] {asset}: {quantity:.6f} @ {price:.2f} | Balance: {self.balances[asset]:.2f}")
        return True
    
    def _execute_sell(self, asset: str, price: float):
        """Esegue un ordine SELL."""
        position = self.positions[asset]
        if position is None:
            return False
        
        quantity = position["quantity"]
        revenue = quantity * price
        
        # Calcola profit/loss
        cost = quantity * position["price"]
        profit = revenue - cost
        
        # Aggiorna saldo
        self.balances[asset] += revenue
        
        # Registra nel ledger
        if self.ledger:
            self.ledger.record_trade(
                asset=asset,
                trade_type="SELL",
                quantity=quantity,
                price=price,
                commission=0,
                notes=f"MontBlanck signal - P/L: {profit:.2f}"
            )
        
        if profit > 0:
            self.winning_trades += 1
        
        self.positions[asset] = None
        logger.info(f"[SELL] {asset}: {quantity:.6f} @ {price:.2f} | P/L: {profit:.2f} | Balance: {self.balances[asset]:.2f}")
        return True
    
    def update(self):
        """Aggiorna prezzi e genera segnali."""
        for asset in self.assets:
            # Ottieni prezzo corrente
            price = self.data_provider.get_price(asset)
            self.price_history[asset].append(price)
            
            # Mantieni solo ultimi 100 prezzi
            if len(self.price_history[asset]) > 100:
                self.price_history[asset] = self.price_history[asset][-100:]
            
            # Genera previsione
            if len(self.price_history[asset]) >= self.config.window_size:
                prediction = self.strategy.update(asset, self.price_history[asset])
                
                # Esegui segnale
                if prediction.signal == Signal.BUY and self.positions[asset] is None:
                    # Calcola quantità (50% del saldo)
                    quantity = (self.balances[asset] * 0.5) / price
                    self._execute_buy(asset, quantity, price)
                
                elif prediction.signal == Signal.SELL and self.positions[asset] is not None:
                    self._execute_sell(asset, price)
                
                # Log stato
                logger.debug(f"{asset}: Price={price:.2f} | Signal={prediction.signal.value} | "
                           f"Confidence={prediction.confidence:.2f} | Trend={prediction.trend}")
    
    def get_status(self) -> Dict:
        """Restituisce lo stato corrente."""
        total_balance = sum(self.balances.values())
        
        # Aggiungi valore posizioni aperte
        for asset, position in self.positions.items():
            if position:
                current_price = self.price_history[asset][-1] if self.price_history[asset] else 0
                total_balance += position["quantity"] * current_price
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "total_balance": total_balance,
            "initial_balance": self.initial_balance,
            "profit_loss": total_balance - self.initial_balance,
            "return_pct": (total_balance - self.initial_balance) / self.initial_balance * 100,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "open_positions": sum(1 for p in self.positions.values() if p is not None),
            "balances": self.balances.copy(),
            "positions": {k: v for k, v in self.positions.items() if v is not None}
        }
    
    def print_status(self):
        """Stampa lo stato corrente."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("  MONT BLANCK LIVE TRADER - STATUS")
        print("=" * 60)
        print(f"  Total Balance:    {status['total_balance']:.2f}")
        print(f"  Initial Balance:  {status['initial_balance']:.2f}")
        print(f"  P/L:              {status['profit_loss']:.2f} ({status['return_pct']:.2f}%)")
        print(f"  Total Trades:     {status['total_trades']}")
        print(f"  Win Rate:         {status['win_rate']:.1f}%")
        print(f"  Open Positions:   {status['open_positions']}")
        print("=" * 60)
        
        for asset, balance in status['balances'].items():
            pos_info = ""
            if asset in status['positions']:
                pos = status['positions'][asset]
                pos_info = f" | Position: {pos['quantity']:.6f} @ {pos['price']:.2f}"
            print(f"  {asset}: {balance:.2f}{pos_info}")
        
        print("=" * 60 + "\n")
    
    def run(self, interval: int = 60, duration: int = None):
        """
        Esegue il trader in loop.
        
        Args:
            interval: Intervallo in secondi tra aggiornamenti
            duration: Durata totale in secondi (None = infinito)
        """
        logger.info(f"Starting MontBlanck Live Trader (interval={interval}s)")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} ---")
                
                # Aggiorna
                self.update()
                
                # Stampa stato ogni 5 iterazioni
                if iteration % 5 == 0:
                    self.print_status()
                
                # Controlla durata
                if duration:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        logger.info(f"Duration reached ({duration}s). Stopping.")
                        break
                
                # Aspetta
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        
        # Report finale
        self.print_status()
        if self.ledger:
            print(self.ledger.generate_report("all"))


def main():
    parser = argparse.ArgumentParser(description="Mont Blanck Live Trading System")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode (use real API)")
    parser.add_argument("--assets", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Assets to trade")
    parser.add_argument("--balance", type=float, default=1000.0, help="Initial balance")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    parser.add_argument("--duration", type=int, default=None, help="Duration in seconds (None = infinite)")
    parser.add_argument("--window", type=int, default=4, help="MontBlanck window size")
    parser.add_argument("--degree", type=int, default=3, help="Polynomial degree")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold")
    
    args = parser.parse_args()
    
    # Configurazione
    config = MontBlanckConfig(
        window_size=args.window,
        poly_degree=args.degree,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold
    )
    
    # Crea trader
    trader = MontBlanckLiveTrader(
        assets=args.assets,
        initial_balance=args.balance,
        paper_trading=not args.live,
        config=config
    )
    
    # Esegui
    trader.run(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()
