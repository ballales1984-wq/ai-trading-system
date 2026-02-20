#!/usr/bin/env python3
"""
Trading Ledger Module - Registro Performance Trading
====================================================

Modulo completo per:
- Registrazione automatica di ogni trade
- Calcolo profit/loss reale
- Aggiornamento saldo totale
- Report giornaliero/settimanale/mensile
- Grafici automatici delle performance
- Statistiche avanzate (win rate, drawdown, Sharpe ratio)

Usage:
    from src.trading_ledger import TradingLedger
    
    ledger = TradingLedger()
    ledger.record_trade({...})
    ledger.generate_report()
    ledger.plot_performance()
"""

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingLedger")


class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class Trade:
    """Rappresenta un singolo trade."""
    id: str
    timestamp: datetime
    asset: str
    trade_type: str  # BUY or SELL
    quantity: float
    price: float
    commission: float
    profit_loss: float = 0.0
    balance_after: float = 0.0
    status: str = "CLOSED"
    notes: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "trade_type": self.trade_type,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "profit_loss": self.profit_loss,
            "balance_after": self.balance_after,
            "status": self.status,
            "notes": self.notes,
            "tags": self.tags
        }


@dataclass
class Position:
    """Rappresenta una posizione aperta."""
    asset: str
    quantity: float
    avg_price: float
    total_cost: float
    opened_at: datetime
    trades: List[str] = None
    
    def __post_init__(self):
        if self.trades is None:
            self.trades = []


class TradingLedger:
    """
    Registro completo per il tracking delle performance di trading.
    
    Features:
    - Registra ogni trade con tutti i dettagli
    - Calcola profit/loss reale (FIFO)
    - Mantiene posizioni aperte
    - Genera report e statistiche
    - Crea grafici automatici
    """
    
    def __init__(self, 
                 data_dir: str = "data/ledger",
                 initial_balance: float = 10000.0,
                 currency: str = "USD"):
        """
        Inizializza il ledger.
        
        Args:
            data_dir: Directory per salvare i dati
            initial_balance: Saldo iniziale
            currency: Valuta di riferimento
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades_file = self.data_dir / "trades.csv"
        self.positions_file = self.data_dir / "positions.json"
        self.balance_file = self.data_dir / "balance.json"
        self.config_file = self.data_dir / "config.json"
        
        self.currency = currency
        self.initial_balance = initial_balance
        
        # Stato interno
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}  # asset -> Position
        self.balance = initial_balance
        self.trade_counter = 0
        
        # Carica dati esistenti
        self._load_data()
        
        logger.info(f"TradingLedger initialized. Balance: {self.balance:.2f} {currency}")
    
    def _load_data(self):
        """Carica i dati salvati."""
        # Carica configurazione
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.trade_counter = config.get('trade_counter', 0)
                self.initial_balance = config.get('initial_balance', self.initial_balance)
        
        # Carica saldo
        if self.balance_file.exists():
            with open(self.balance_file, 'r') as f:
                data = json.load(f)
                self.balance = data.get('balance', self.initial_balance)
        
        # Carica trades
        if self.trades_file.exists():
            with open(self.trades_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trade = Trade(
                        id=row['id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        asset=row['asset'],
                        trade_type=row['trade_type'],
                        quantity=float(row['quantity']),
                        price=float(row['price']),
                        commission=float(row['commission']),
                        profit_loss=float(row['profit_loss']),
                        balance_after=float(row['balance_after']),
                        status=row['status'],
                        notes=row.get('notes', ''),
                        tags=json.loads(row.get('tags', '[]'))
                    )
                    self.trades.append(trade)
        
        # Carica posizioni
        if self.positions_file.exists():
            with open(self.positions_file, 'r') as f:
                data = json.load(f)
                for asset, pos_data in data.items():
                    self.positions[asset] = Position(
                        asset=asset,
                        quantity=pos_data['quantity'],
                        avg_price=pos_data['avg_price'],
                        total_cost=pos_data['total_cost'],
                        opened_at=datetime.fromisoformat(pos_data['opened_at']),
                        trades=pos_data.get('trades', [])
                    )
    
    def _save_data(self):
        """Salva tutti i dati."""
        # Salva configurazione
        with open(self.config_file, 'w') as f:
            json.dump({
                'trade_counter': self.trade_counter,
                'initial_balance': self.initial_balance,
                'currency': self.currency
            }, f, indent=2)
        
        # Salva saldo
        with open(self.balance_file, 'w') as f:
            json.dump({
                'balance': self.balance,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        
        # Salva trades
        with open(self.trades_file, 'w', newline='') as f:
            fieldnames = ['id', 'timestamp', 'asset', 'trade_type', 'quantity', 
                         'price', 'commission', 'profit_loss', 'balance_after',
                         'status', 'notes', 'tags']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade.to_dict())
        
        # Salva posizioni
        with open(self.positions_file, 'w') as f:
            data = {}
            for asset, pos in self.positions.items():
                data[asset] = {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'total_cost': pos.total_cost,
                    'opened_at': pos.opened_at.isoformat(),
                    'trades': pos.trades
                }
            json.dump(data, f, indent=2)
    
    def _generate_trade_id(self) -> str:
        """Genera un ID univoco per il trade."""
        self.trade_counter += 1
        return f"TRD-{datetime.now().strftime('%Y%m%d')}-{self.trade_counter:05d}"
    
    def record_trade(self, 
                     asset: str,
                     trade_type: str,
                     quantity: float,
                     price: float,
                     commission: float = 0.0,
                     notes: str = "",
                     tags: List[str] = None) -> Trade:
        """
        Registra un nuovo trade.
        
        Args:
            asset: Simbolo dell'asset (es. BTC, ETH)
            trade_type: 'BUY' o 'SELL'
            quantity: Quantità scambiata
            price: Prezzo di esecuzione
            commission: Commissioni pagate
            notes: Note opzionali
            tags: Tag per categorizzazione
            
        Returns:
            Trade registrato
        """
        trade_type = trade_type.upper()
        if trade_type not in ['BUY', 'SELL']:
            raise ValueError(f"trade_type must be 'BUY' or 'SELL', got {trade_type}")
        
        profit_loss = 0.0
        
        if trade_type == 'BUY':
            # Acquisto: aggiungi alla posizione
            total_cost = price * quantity + commission
            self.balance -= total_cost
            
            if asset in self.positions:
                pos = self.positions[asset]
                new_quantity = pos.quantity + quantity
                new_total_cost = pos.total_cost + total_cost
                pos.quantity = new_quantity
                pos.total_cost = new_total_cost
                pos.avg_price = new_total_cost / new_quantity if new_quantity > 0 else 0
                pos.trades.append(self._generate_trade_id())
            else:
                self.positions[asset] = Position(
                    asset=asset,
                    quantity=quantity,
                    avg_price=price,
                    total_cost=total_cost,
                    opened_at=datetime.now(),
                    trades=[self._generate_trade_id()]
                )
        
        elif trade_type == 'SELL':
            # Vendita: calcola profit/loss
            if asset not in self.positions:
                logger.warning(f"Selling {asset} without open position - treating as short")
                profit_loss = -commission
            else:
                pos = self.positions[asset]
                if pos.quantity < quantity:
                    logger.warning(f"Selling more {asset} than held ({quantity} > {pos.quantity})")
                
                # Calcola profit/loss usando prezzo medio
                cost_basis = pos.avg_price * quantity
                revenue = price * quantity
                profit_loss = revenue - cost_basis - commission
                
                # Aggiorna posizione
                pos.quantity -= quantity
                pos.total_cost = pos.avg_price * pos.quantity
                
                if pos.quantity <= 0:
                    del self.positions[asset]
            
            self.balance += price * quantity - commission
        
        # Crea il trade
        trade = Trade(
            id=self._generate_trade_id(),
            timestamp=datetime.now(),
            asset=asset,
            trade_type=trade_type,
            quantity=quantity,
            price=price,
            commission=commission,
            profit_loss=profit_loss,
            balance_after=self.balance,
            status="CLOSED",
            notes=notes,
            tags=tags or []
        )
        
        self.trades.append(trade)
        self._save_data()
        
        logger.info(f"Trade recorded: {trade_type} {quantity} {asset} @ {price} | P/L: {profit_loss:.2f} | Balance: {self.balance:.2f}")
        
        return trade
    
    def get_statistics(self, period: str = "all") -> Dict:
        """
        Calcola statistiche di performance.
        
        Args:
            period: 'day', 'week', 'month', 'year', 'all'
            
        Returns:
            Dizionario con statistiche
        """
        # Filtra trades per periodo
        now = datetime.now()
        if period == "day":
            start = now - timedelta(days=1)
        elif period == "week":
            start = now - timedelta(weeks=1)
        elif period == "month":
            start = now - timedelta(days=30)
        elif period == "year":
            start = now - timedelta(days=365)
        else:
            start = datetime.min
        
        filtered_trades = [t for t in self.trades if t.timestamp >= start]
        
        if not filtered_trades:
            return {
                "period": period,
                "total_trades": 0,
                "total_profit_loss": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        
        # Calcola metriche
        profits = [t.profit_loss for t in filtered_trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in filtered_trades if t.profit_loss < 0]
        
        total_pl = sum(t.profit_loss for t in filtered_trades)
        total_trades = len(filtered_trades)
        winning_trades = len(profits)
        losing_trades = len(losses)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Calcola drawdown
        balance_history = [self.initial_balance]
        for t in sorted(filtered_trades, key=lambda x: x.timestamp):
            balance_history.append(t.balance_after)
        
        peak = balance_history[0]
        max_drawdown = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio semplificato
        returns = [t.profit_loss / t.balance_after for t in filtered_trades if t.balance_after > 0]
        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            sharpe_ratio = (avg_return / std_dev * (252 ** 0.5)) if std_dev > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            "period": period,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_profit_loss": total_pl,
            "total_profit_loss_pct": (total_pl / self.initial_balance * 100),
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "current_balance": self.balance,
            "return_pct": ((self.balance - self.initial_balance) / self.initial_balance * 100)
        }
    
    def get_trade_history(self, 
                          asset: str = None,
                          trade_type: str = None,
                          limit: int = None) -> List[Trade]:
        """
        Ottiene lo storico dei trade.
        
        Args:
            asset: Filtra per asset
            trade_type: Filtra per tipo (BUY/SELL)
            limit: Numero massimo di trade da restituire
            
        Returns:
            Lista di trade
        """
        trades = self.trades
        
        if asset:
            trades = [t for t in trades if t.asset == asset]
        if trade_type:
            trades = [t for t in trades if t.trade_type == trade_type.upper()]
        
        trades = sorted(trades, key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            trades = trades[:limit]
        
        return trades
    
    def get_open_positions(self) -> Dict[str, Position]:
        """Restituisce le posizioni attualmente aperte."""
        return self.positions.copy()
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calcola il valore totale del portafoglio.
        
        Args:
            prices: Dizionario asset -> prezzo attuale
            
        Returns:
            Valore totale in valuta base
        """
        total = self.balance
        
        for asset, pos in self.positions.items():
            if asset in prices:
                total += pos.quantity * prices[asset]
        
        return total
    
    def generate_report(self, period: str = "all") -> str:
        """
        Genera un report testuale delle performance.
        
        Args:
            period: Periodo del report
            
        Returns:
            Report formattato
        """
        stats = self.get_statistics(period)
        
        report = f"""
{'='*60}
  TRADING PERFORMANCE REPORT
  Period: {period.upper()}
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

📊 OVERVIEW
  Initial Balance:    {self.initial_balance:,.2f} {self.currency}
  Current Balance:    {stats['current_balance']:,.2f} {self.currency}
  Total P/L:          {stats['total_profit_loss']:,.2f} {self.currency} ({stats['total_profit_loss_pct']:.2f}%)
  Return:             {stats['return_pct']:.2f}%

📈 TRADE STATISTICS
  Total Trades:       {stats['total_trades']}
  Winning Trades:     {stats['winning_trades']}
  Losing Trades:      {stats['losing_trades']}
  Win Rate:           {stats['win_rate']:.1f}%

💰 PROFIT/LOSS
  Average Profit:     {stats['avg_profit']:,.2f} {self.currency}
  Average Loss:       {stats['avg_loss']:,.2f} {self.currency}
  Profit Factor:      {stats['profit_factor']:.2f}

⚠️ RISK METRICS
  Max Drawdown:       {stats['max_drawdown']:.2f}%
  Sharpe Ratio:       {stats['sharpe_ratio']:.2f}

📂 OPEN POSITIONS
"""
        
        if self.positions:
            for asset, pos in self.positions.items():
                report += f"  {asset}: {pos.quantity:.6f} @ {pos.avg_price:.2f} (Cost: {pos.total_cost:.2f})\n"
        else:
            report += "  No open positions\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def export_to_csv(self, filepath: str = None) -> str:
        """
        Esporta i trade in formato CSV.
        
        Args:
            filepath: Percorso del file (opzionale)
            
        Returns:
            Percorso del file creato
        """
        if filepath is None:
            filepath = self.data_dir / f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['id', 'timestamp', 'asset', 'trade_type', 'quantity', 
                         'price', 'commission', 'profit_loss', 'balance_after',
                         'status', 'notes', 'tags']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade.to_dict())
        
        logger.info(f"Trades exported to {filepath}")
        return str(filepath)
    
    def plot_performance(self, save_path: str = None, show: bool = True):
        """
        Genera grafici delle performance.
        
        Args:
            save_path: Percorso per salvare il grafico
            show: Se mostrare il grafico
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            logger.error("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if not self.trades:
            logger.warning("No trades to plot")
            return
        
        # Prepara dati
        trades_sorted = sorted(self.trades, key=lambda x: x.timestamp)
        dates = [t.timestamp for t in trades_sorted]
        balances = [t.balance_after for t in trades_sorted]
        profits = [t.profit_loss for t in trades_sorted]
        
        # Crea figura con 4 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Trading Performance Dashboard', fontsize=14, fontweight='bold')
        
        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(dates, balances, 'b-', linewidth=2, label='Balance')
        ax1.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial')
        ax1.fill_between(dates, self.initial_balance, balances, alpha=0.3, 
                        where=[b >= self.initial_balance for b in balances], color='green')
        ax1.fill_between(dates, self.initial_balance, balances, alpha=0.3,
                        where=[b < self.initial_balance for b in balances], color='red')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'Balance ({self.currency})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # 2. Profit/Loss per Trade
        ax2 = axes[0, 1]
        colors = ['green' if p > 0 else 'red' for p in profits]
        ax2.bar(range(len(profits)), profits, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_title('Profit/Loss per Trade')
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel(f'P/L ({self.currency})')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative P/L
        ax3 = axes[1, 0]
        cumulative_pl = []
        running = 0
        for p in profits:
            running += p
            cumulative_pl.append(running)
        ax3.plot(dates, cumulative_pl, 'g-', linewidth=2)
        ax3.fill_between(dates, 0, cumulative_pl, alpha=0.3, color='green')
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_title('Cumulative Profit/Loss')
        ax3.set_xlabel('Date')
        ax3.set_ylabel(f'Cumulative P/L ({self.currency})')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # 4. Win/Loss Distribution
        ax4 = axes[1, 1]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        ax4.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})')
        ax4.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})')
        ax4.axvline(x=0, color='black', linewidth=1)
        ax4.set_title('Win/Loss Distribution')
        ax4.set_xlabel(f'P/L ({self.currency})')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def reset(self, confirm: bool = False):
        """
        Reset completo del ledger.
        
        Args:
            confirm: Deve essere True per confermare il reset
        """
        if not confirm:
            logger.warning("Reset not confirmed. Pass confirm=True to reset.")
            return
        
        self.trades = []
        self.positions = {}
        self.balance = self.initial_balance
        self.trade_counter = 0
        
        # Rimuovi file
        for f in [self.trades_file, self.positions_file, self.balance_file]:
            if f.exists():
                f.unlink()
        
        self._save_data()
        logger.info("Ledger reset completed")


# Funzioni di convenienza per integrazione rapida
_ledger_instance = None

def get_ledger(initial_balance: float = 10000.0) -> TradingLedger:
    """Ottiene l'istanza singleton del ledger."""
    global _ledger_instance
    if _ledger_instance is None:
        _ledger_instance = TradingLedger(initial_balance=initial_balance)
    return _ledger_instance


def quick_trade(asset: str, trade_type: str, quantity: float, 
                price: float, commission: float = 0.0) -> Trade:
    """Registra rapidamente un trade usando il ledger singleton."""
    ledger = get_ledger()
    return ledger.record_trade(asset, trade_type, quantity, price, commission)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Ledger CLI")
    parser.add_argument("--report", "-r", choices=["day", "week", "month", "year", "all"],
                       default="all", help="Generate report for period")
    parser.add_argument("--plot", "-p", action="store_true", help="Generate performance plots")
    parser.add_argument("--export", "-e", type=str, help="Export trades to CSV file")
    parser.add_argument("--balance", "-b", type=float, default=10000.0, 
                       help="Initial balance for new ledger")
    
    args = parser.parse_args()
    
    ledger = TradingLedger(initial_balance=args.balance)
    
    if args.report:
        print(ledger.generate_report(args.report))
    
    if args.plot:
        ledger.plot_performance(show=True)
    
    if args.export:
        ledger.export_to_csv(args.export)
