"""
Backtest Engine - Realistic Trading Simulation
==============================================
Simulates trading with realistic conditions:
- Commissioni Binance spot: 0.04-0.1%
- Slippage: 0.05-0.3% in base alla volatilità
- Spread realistico

Author: AI Trading System
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configurazione del backtest"""
    initial_balance: float = 100000.0
    commission: float = 0.001  # 0.1% (commissione Binance spot)
    slippage_base: float = 0.001  # 0.1% base
    slippage_volatility_multiplier: float = 0.002  # +0.2% per unità di volatilità
    max_slippage: float = 0.003  # Max 0.3%
    
    # Parametri di trading
    min_trade_value: float = 10.0
    max_trade_value: float = 10000.0
    
    # Risk parameters
    max_drawdown: float = -0.15  # -15%
    stop_loss: float = 0.04  # 4%
    take_profit: float = 0.08  # 8%


@dataclass
class Trade:
    """Registro di un trade"""
    timestamp: datetime
    asset: str
    action: str  # BUY/SELL
    entry_price: float
    exit_price: float = 0.0
    quantity: float = 0.0
    value: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    holding_period: int = 0  # cicli
    reason: str = ""


class BacktestEngine:
    """
    Motore di backtest con condizioni realistiche.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        
        # Posizioni aperte
        self.positions: Dict[str, Dict] = {}  # asset -> {entry_price, quantity, entry_time}
        
        # Storico trades
        self.trades: List[Trade] = []
        
        # Statistiche
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "max_drawdown": 0.0,
            "peak_balance": self.config.initial_balance,
        }
        
        self.cycle_count = 0
    
    def reset(self):
        """Reset del backtest"""
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        self.positions = {}
        self.trades = []
        self.cycle_count = 0
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "max_drawdown": 0.0,
            "peak_balance": self.config.initial_balance,
        }
    
    def calculate_slippage(self, volatility: float = 0.5) -> float:
        """
        Calcola slippage realistico basato sulla volatilità.
        
        Args:
            volatilità: Volatilità dell'asset (0-1)
            
        Returns:
            Slippage come percentuale
        """
        slippage = self.config.slippage_base + (volatility * self.config.slippage_volatility_multiplier)
        return min(slippage, self.config.max_slippage)
    
    def apply_slippage(self, price: float, action: str, volatility: float = 0.5) -> float:
        """
        Applica slippage al prezzo.
        
        Args:
            price: Prezzo originale
            action: BUY o SELL
            volatility: Volatilità dell'asset
            
        Returns:
            Prezzo con slippage
        """
        slippage = self.calculate_slippage(volatility)
        
        if action == "BUY":
            # Slippage negativo (prezzo più alto)
            return price * (1 + slippage)
        else:
            # Slippage positivo (prezzo più basso)
            return price * (1 - slippage)
    
    def execute_order(
        self,
        asset: str,
        action: str,
        price: float,
        quantity: float,
        confidence: float = 0.6,
        volatility: float = 0.5,
        timestamp: datetime = None
    ) -> Tuple[bool, Trade]:
        """
        Esegue un ordine con condizioni realistiche.
        
        Args:
            asset: Simbolo asset
            action: BUY o SELL
            price: Prezzo corrente
            quantity: Quantità in USDT
            confidence: Confidence del segnale
            volatility: Volatilità dell'asset
            timestamp: Timestamp dell'ordine
            
        Returns:
            (success, trade)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Valida quantità
        if quantity < self.config.min_trade_value:
            return False, None
        
        if quantity > self.config.max_trade_value:
            quantity = self.config.max_trade_value
        
        # Calcolo slippage
        slippage_pct = self.calculate_slippage(volatility)
        executed_price = self.apply_slippage(price, action, volatility)
        
        # Calcolo commissione
        trade_value = quantity
        commission = trade_value * self.config.commission
        
        # Verifica balance sufficiente per BUY
        if action == "BUY" and (trade_value + commission) > self.balance:
            return False, None
        
        # Esegui ordine
        if action == "BUY":
            # Apri posizione long
            self.balance -= (trade_value + commission)
            
            self.positions[asset] = {
                "entry_price": executed_price,
                "quantity": quantity / executed_price,  # Converti in quantità crypto
                "entry_time": timestamp,
                "value": trade_value
            }
            
            trade = Trade(
                timestamp=timestamp,
                asset=asset,
                action=action,
                entry_price=executed_price,
                quantity=quantity / executed_price,
                value=trade_value,
                commission=commission,
                slippage=slippage_pct,
                reason=f"confidence={confidence:.2f}"
            )
            
        elif action == "SELL" and asset in self.positions:
            # Chiudi posizione long
            position = self.positions[asset]
            exit_value = position["quantity"] * executed_price
            exit_commission = exit_value * self.config.commission
            
            pnl = exit_value - position["value"] - commission - exit_commission
            pnl_percent = pnl / position["value"]
            
            self.balance += (exit_value - commission - exit_commission)
            
            holding_period = (timestamp - position["entry_time"]).total_seconds() / 60  # Minuti
            
            trade = Trade(
                timestamp=timestamp,
                asset=asset,
                action=action,
                entry_price=position["entry_price"],
                exit_price=executed_price,
                quantity=position["quantity"],
                value=exit_value,
                commission=commission + exit_commission,
                slippage=slippage_pct,
                pnl=pnl,
                pnl_percent=pnl_percent,
                holding_period=int(holding_period),
                reason=f"confidence={confidence:.2f}"
            )
            
            # Aggiorna statistiche
            self.stats["total_trades"] += 1
            if pnl > 0:
                self.stats["winning_trades"] += 1
            else:
                self.stats["losing_trades"] += 1
            
            self.trades.append(trade)
            
            # Rimuovi posizione
            del self.positions[asset]
        else:
            return False, None
        
        # Aggiorna statistiche commissioni
        self.stats["total_commission"] += commission
        self.stats["total_slippage"] += trade_value * slippage_pct
        
        return True, trade
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[Trade]:
        """
        Verifica se stop-loss o take-profit sono raggiunti.
        
        Args:
            current_prices: Prezzi correnti degli asset
            
        Returns:
            Lista di trade eseguiti
        """
        closed_trades = []
        
        for asset, position in list(self.positions.items()):
            if asset not in current_prices:
                continue
            
            current_price = current_prices[asset]
            entry_price = position["entry_price"]
            value = position["value"]
            
            # Calcolo P&L
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Stop Loss
            if pnl_pct <= -self.config.stop_loss:
                success, trade = self.execute_order(
                    asset=asset,
                    action="SELL",
                    price=current_price,
                    quantity=value,
                    volatility=0.5,
                    timestamp=datetime.now()
                )
                if success:
                    trade.reason = "STOP_LOSS"
                    closed_trades.append(trade)
            
            # Take Profit
            elif pnl_pct >= self.config.take_profit:
                success, trade = self.execute_order(
                    asset=asset,
                    action="SELL",
                    price=current_price,
                    quantity=value,
                    volatility=0.5,
                    timestamp=datetime.now()
                )
                if success:
                    trade.reason = "TAKE_PROFIT"
                    closed_trades.append(trade)
        
        return closed_trades
    
    def update_equity(self):
        """Aggiorna curva equity e drawdown"""
        # Valore totale (balance + posizioni aperte)
        total_value = self.balance
        for asset, position in self.positions.items():
            total_value += position["value"]
        
        # Aggiorna peak
        if total_value > self.stats["peak_balance"]:
            self.stats["peak_balance"] = total_value
        
        # Calcola drawdown
        drawdown = (total_value - self.stats["peak_balance"]) / self.stats["peak_balance"]
        
        # Aggiorna max drawdown
        if drawdown < self.stats["max_drawdown"]:
            self.stats["max_drawdown"] = drawdown
        
        # Aggiungi a curve
        self.stats["equity_curve"].append(total_value)
        self.stats["drawdown_curve"].append(drawdown)
        
        return total_value, drawdown
    
    def run_cycle(
        self,
        signals: Dict[str, Dict],
        prices: Dict[str, float],
        volatilities: Dict[str, float] = None
    ) -> Dict:
        """
        Esegue un ciclo di backtest.
        
        Args:
            signals: {asset: {action, confidence, score}}
            prices: {asset: price}
            volatilities: {asset: volatility}
            
        Returns:
            Risultati del ciclo
        """
        self.cycle_count += 1
        
        if volatilities is None:
            volatilities = {asset: 0.5 for asset in prices.keys()}
        
        # Check stop loss / take profit
        self.check_stop_loss_take_profit(prices)
        
        # Esegui segnali
        executed_trades = []
        
        for asset, signal in signals.items():
            if asset not in prices:
                continue
            
            action = signal.get("action", "HOLD")
            if action == "HOLD":
                continue
            
            confidence = signal.get("confidence", 0.6)
            score = signal.get("score", 0.5)
            
            # Usa position size dal segnale o default
            position_size = signal.get("amount", self.balance * 0.05)
            
            success, trade = self.execute_order(
                asset=asset,
                action=action,
                price=prices[asset],
                quantity=position_size,
                confidence=confidence,
                volatility=volatilities.get(asset, 0.5),
                timestamp=datetime.now()
            )
            
            if success:
                executed_trades.append(trade)
                logger.info(f"Trade executed: {trade.action} {trade.asset} @ {trade.entry_price}, value={trade.value}")
        
        # Aggiorna equity
        total_value, drawdown = self.update_equity()
        
        # Check kill switch
        if drawdown <= self.config.max_drawdown:
            # Registra tutti i trades eseguiti (BUY e SELL)
            for trade in executed_trades:
                self.trades.append(trade)
            return {
                "cycle": self.cycle_count,
                "status": "KILL_SWITCH",
                "drawdown": drawdown,
                "total_value": total_value,
                "trades": len(executed_trades)
            }
        
        # Registra tutti i trades eseguiti
        for trade in executed_trades:
            self.trades.append(trade)
        
        return {
            "cycle": self.cycle_count,
            "status": "OK",
            "total_value": total_value,
            "balance": self.balance,
            "drawdown": drawdown,
            "positions": len(self.positions),
            "trades": len(executed_trades),
            "executed_trades": executed_trades
        }
    
    def get_summary(self) -> Dict:
        """
        Calcola statistiche finali del backtest.
        
        Returns:
            Dizionario con statistiche
        """
        if not self.trades:
            return {
                "message": "No trades executed",
                "initial_balance": self.initial_balance,
                "final_balance": self.balance,
                "total_return": 0.0,
                "total_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": self.stats.get("max_drawdown", 0.0),
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
                "total_costs": 0.0,
                "avg_holding_period": 0.0,
                "cycles": self.cycle_count
            }
        
        # Calcoli base
        total_return = (self.balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        total_pnl = sum(t.pnl for t in self.trades) if self.trades else 0.0
        
        # Win rate
        win_rate = self.stats["winning_trades"] / self.stats["total_trades"] if self.stats["total_trades"] > 0 else 0
        
        # Profit Factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe Ratio (semplificato)
        if len(self.stats["equity_curve"]) > 1:
            returns = np.diff(self.stats["equity_curve"]) / self.stats["equity_curve"][:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 2) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Sortino Ratio
        if len(self.stats["equity_curve"]) > 1:
            returns = np.diff(self.stats["equity_curve"]) / self.stats["equity_curve"][:-1]
            downside_returns = returns[returns < 0]
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(365 * 24 * 2) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        else:
            sortino = 0
        
        # Average holding period
        avg_holding = np.mean([t.holding_period for t in self.trades]) if self.trades else 0
        
        return {
            # Performance
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_return": total_return,
            "total_pnl": total_pnl if 'total_pnl' in locals() else 0.0,
            
            # Trades
            "total_trades": self.stats["total_trades"],
            "winning_trades": self.stats["winning_trades"],
            "losing_trades": self.stats["losing_trades"],
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            
            # Risk
            "max_drawdown": self.stats["max_drawdown"],
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            
            # Costs
            "total_commission": self.stats["total_commission"],
            "total_slippage": self.stats["total_slippage"],
            "total_costs": self.stats["total_commission"] + self.stats["total_slippage"],
            
            # Timing
            "avg_holding_period": avg_holding,
            "cycles": self.cycle_count,
            
            # Equity curve
            "equity_curve": self.stats["equity_curve"],
            "drawdown_curve": self.stats["drawdown_curve"]
        }
    
    def print_summary(self):
        """Stampa un sommario leggibile"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\n[PERFORMANCE]")
        print(f"   Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"   Final Balance:   ${summary['final_balance']:,.2f}")
        print(f"   Total Return:   {summary['total_return']:.2%}")
        print(f"   Total P&L:      ${summary['total_pnl']:,.2f}")
        
        print(f"\n[TRADES]")
        print(f"   Total Trades:    {summary['total_trades']}")
        print(f"   Winning Trades:   {summary['winning_trades']}")
        print(f"   Losing Trades:   {summary['losing_trades']}")
        print(f"   Win Rate:        {summary['win_rate']:.2%}")
        print(f"   Profit Factor:   {summary['profit_factor']:.2f}")
        
        print(f"\n[RISK]")
        print(f"   Max Drawdown:    {summary['max_drawdown']:.2%}")
        print(f"   Sharpe Ratio:     {summary['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio:   {summary['sortino_ratio']:.2f}")
        
        print(f"\n[COSTS]")
        print(f"   Total Commission: ${summary['total_commission']:,.2f}")
        print(f"   Total Slippage:  ${summary['total_slippage']:,.2f}")
        print(f"   Total Costs:     ${summary['total_costs']:,.2f}")
        
        print(f"\n[TIMING]")
        print(f"   Cycles:           {summary['cycles']}")
        print(f"   Avg Holding:     {summary['avg_holding_period']:.1f} min")
        
        print("=" * 60)
        
        # Valutazione
        print("\n[EVALUATION]")
        
        if summary['sharpe_ratio'] >= 1.5:
            print("   [OK] Sharpe Ratio: EXCELLENT (>1.5)")
        elif summary['sharpe_ratio'] >= 1.2:
            print("   [OK] Sharpe Ratio: GOOD (1.2-1.5)")
        elif summary['sharpe_ratio'] >= 0.8:
            print("   [!] Sharpe Ratio: ACCEPTABLE (0.8-1.2)")
        else:
            print("   [X] Sharpe Ratio: POOR (<0.8)")
        
        if summary['max_drawdown'] >= -0.15:
            print("   [OK] Max Drawdown: EXCELLENT (> -15%)")
        elif summary['max_drawdown'] >= -0.20:
            print("   [!] Max Drawdown: ACCEPTABLE (-15% to -20%)")
        else:
            print("   [X] Max Drawdown: TOO HIGH (< -20%)")
        
        if summary['win_rate'] >= 0.60:
            print("   [OK] Win Rate: EXCELLENT (>60%)")
        elif summary['win_rate'] >= 0.50:
            print("   [!] Win Rate: ACCEPTABLE (50-60%)")
        else:
            print("   [X] Win Rate: POOR (<50%)")
        
        if summary['profit_factor'] >= 1.5:
            print("   [OK] Profit Factor: EXCELLENT (>1.5)")
        elif summary['profit_factor'] >= 1.2:
            print("   [!] Profit Factor: ACCEPTABLE (1.2-1.5)")
        else:
            print("   [X] Profit Factor: POOR (<1.2)")
        
        print("=" * 60)


# ==============================================
# ESEMPIO DI UTILIZZO
# ==============================================
if __name__ == "__main__":
    # Configura backtest
    config = BacktestConfig(
        initial_balance=100000,
        commission=0.001,  # 0.1%
        slippage_base=0.001,
        max_drawdown=-0.15
    )
    
    engine = BacktestEngine(config)
    
    # Simula 100 cicli
    print("Running backtest simulation...")
    
    for i in range(100):
        # Genera prezzi casuali
        prices = {
            "BTCUSDT": 68000 + np.random.randn() * 1000,
            "ETHUSDT": 3500 + np.random.randn() * 100,
            "SOLUSDT": 180 + np.random.randn() * 10
        }
        
        # Genera segnali casuali (simulazione)
        signals = {}
        for asset in prices:
            if np.random.random() < 0.3:  # 30% probabilità di segnale
                action = np.random.choice(["BUY", "SELL", "HOLD"], p=[0.4, 0.2, 0.4])
                if action != "HOLD":
                    signals[asset] = {
                        "action": action,
                        "confidence": np.random.uniform(0.6, 0.9),
                        "score": np.random.uniform(0.4, 0.8),
                        "amount": 5000
                    }
        
        # Esegui ciclo
        result = engine.run_cycle(signals, prices)
        
        if result.get("status") == "KILL_SWITCH":
            print(f"Kill switch activated at cycle {i}")
            break
    
    # Stampa risultati
    engine.print_summary()
