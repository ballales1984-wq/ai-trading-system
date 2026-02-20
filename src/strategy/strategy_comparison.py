#!/usr/bin/env python3
"""
Strategy Comparison Engine
==========================

Sistema per confrontare performance di Monte Carlo vs Mont Blanck.
Entrambe le strategie girano in parallelo sullo stesso set di dati.

Features:
- Confronto in tempo reale
- Tracking separato di saldi, trade, drawdown
- Metriche di performance
- Grafici comparativi
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from collections import defaultdict

# Import strategie
from src.strategy.montblanck import MontBlanck, Signal as MontBlanckSignal

logger = logging.getLogger("StrategyComparison")


class StrategyType(Enum):
    MONTE_CARLO = "MonteCarlo"
    MONT_BLANCK = "MontBlanck"


@dataclass
class Trade:
    """Rappresenta un singolo trade."""
    timestamp: datetime
    strategy: StrategyType
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    quantity: float
    pnl: float = 0.0
    cumulative_pnl: float = 0.0


@dataclass
class StrategyState:
    """Stato di una strategia."""
    initial_balance: float = 10000.0
    position: float = 0.0  # Quantità detenuta
    entry_price: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    balance_history: List[float] = field(default_factory=list)
    drawdown_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Inizializza i valori dipendenti."""
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance


@dataclass
class ComparisonResult:
    """Risultato del confronto tra strategie."""
    timestamp: datetime
    symbol: str
    
    # Monte Carlo
    mc_signal: str
    mc_balance: float
    mc_pnl: float
    mc_drawdown: float
    
    # Mont Blanck
    mb_signal: str
    mb_balance: float
    mb_pnl: float
    mb_drawdown: float
    
    # Prezzo corrente
    current_price: float


class MonteCarloStrategy:
    """
    Implementazione semplificata della strategia Monte Carlo.
    Usa simulazione Monte Carlo per prevedere movimenti di prezzo.
    """
    
    def __init__(self, 
                 n_simulations: int = 1000,
                 confidence_level: float = 0.95,
                 buy_threshold: float = 0.02,
                 sell_threshold: float = -0.02):
        """
        Inizializza la strategia Monte Carlo.
        
        Args:
            n_simulations: Numero di simulazioni
            confidence_level: Livello di confidenza
            buy_threshold: Soglia per BUY
            sell_threshold: Soglia per SELL
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.price_history: List[float] = []
        
    def _calculate_returns(self, prices: List[float]) -> np.ndarray:
        """Calcola i rendimenti."""
        if len(prices) < 2:
            return np.array([])
        return np.diff(np.log(prices))
    
    def _simulate_paths(self, returns: np.ndarray, n_steps: int = 5) -> np.ndarray:
        """
        Simula percorsi di prezzo usando Monte Carlo.
        
        Args:
            returns: Rendimenti storici
            n_steps: Numero di step da simulare
            
        Returns:
            Matrice di percorsi simulati (n_simulations x n_steps)
        """
        if len(returns) == 0:
            return np.array([])
        
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Genera percorsi casuali
        dt = 1
        paths = np.zeros((self.n_simulations, n_steps))
        
        for i in range(self.n_simulations):
            random_shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_steps)
            paths[i] = np.cumsum(random_shocks)
        
        return paths
    
    def predict(self, prices: List[float]) -> Dict:
        """
        Genera previsione usando Monte Carlo.
        
        Args:
            prices: Lista di prezzi
            
        Returns:
            Dizionario con segnale e metriche
        """
        if len(prices) < 10:
            return {
                "signal": "HOLD",
                "expected_change": 0.0,
                "confidence": 0.0,
                "var_95": 0.0,
                "var_5": 0.0
            }
        
        self.price_history = prices
        current_price = prices[-1]
        
        # Calcola rendimenti
        returns = self._calculate_returns(prices)
        
        # Simula percorsi
        paths = self._simulate_paths(returns, n_steps=5)
        
        if paths.size == 0:
            return {
                "signal": "HOLD",
                "expected_change": 0.0,
                "confidence": 0.0,
                "var_95": 0.0,
                "var_5": 0.0
            }
        
        # Calcola prezzi finali simulati
        final_returns = paths[:, -1]
        simulated_prices = current_price * np.exp(final_returns)
        
        # Calcola metriche
        expected_price = np.mean(simulated_prices)
        expected_change = (expected_price - current_price) / current_price
        
        var_95 = np.percentile(simulated_prices, 95)
        var_5 = np.percentile(simulated_prices, 5)
        
        # Calcola probabilità di movimento positivo
        prob_up = np.sum(simulated_prices > current_price) / len(simulated_prices)
        
        # Genera segnale
        signal = "HOLD"
        if expected_change >= self.buy_threshold and prob_up > 0.6:
            signal = "BUY"
        elif expected_change <= self.sell_threshold and prob_up < 0.4:
            signal = "SELL"
        
        return {
            "signal": signal,
            "expected_change": expected_change,
            "confidence": prob_up if signal == "BUY" else (1 - prob_up) if signal == "SELL" else 0.5,
            "var_95": var_95,
            "var_5": var_5,
            "expected_price": expected_price
        }


class StrategyComparisonEngine:
    """
    Engine per confrontare Monte Carlo vs Mont Blanck.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 trade_size_pct: float = 0.1):
        """
        Inizializza l'engine di confronto.
        
        Args:
            initial_balance: Saldo iniziale per entrambe le strategie
            trade_size_pct: Percentuale del saldo da usare per trade
        """
        self.initial_balance = initial_balance
        self.trade_size_pct = trade_size_pct
        
        # Inizializza strategie
        self.monte_carlo = MonteCarloStrategy()
        self.mont_blanck = MontBlanck(
            window_size=5,
            poly_degree=3,
            buy_threshold=0.02,
            sell_threshold=-0.02
        )
        
        # Stati delle strategie
        self.states = {
            StrategyType.MONTE_CARLO: StrategyState(initial_balance=initial_balance),
            StrategyType.MONT_BLANCK: StrategyState(initial_balance=initial_balance)
        }
        
        # Storico confronti
        self.comparison_history: List[ComparisonResult] = []
        
        # Prezzi per simbolo
        self.price_data: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"StrategyComparisonEngine initialized with balance: {initial_balance}")
    
    def update(self, symbol: str, price: float, timestamp: datetime = None) -> ComparisonResult:
        """
        Aggiorna entrambe le strategie con nuovo prezzo.
        
        Args:
            symbol: Simbolo dell'asset
            price: Nuovo prezzo
            timestamp: Timestamp (opzionale)
            
        Returns:
            ComparisonResult con stato attuale
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Aggiungi prezzo allo storico
        self.price_data[symbol].append(price)
        prices = self.price_data[symbol]
        
        # Ottieni segnali da entrambe le strategie
        mc_result = self.monte_carlo.predict(prices)
        mb_prediction = self.mont_blanck.predict(prices)
        
        # Esegui trade per Monte Carlo
        mc_signal = mc_result["signal"]
        self._execute_trade(StrategyType.MONTE_CARLO, symbol, price, mc_signal, timestamp)
        
        # Esegui trade per Mont Blanck
        mb_signal = mb_prediction.signal.value
        self._execute_trade(StrategyType.MONT_BLANCK, symbol, price, mb_signal, timestamp)
        
        # Aggiorna drawdown
        self._update_drawdown(StrategyType.MONTE_CARLO)
        self._update_drawdown(StrategyType.MONT_BLANCK)
        
        # Crea risultato confronto
        mc_state = self.states[StrategyType.MONTE_CARLO]
        mb_state = self.states[StrategyType.MONT_BLANCK]
        
        result = ComparisonResult(
            timestamp=timestamp,
            symbol=symbol,
            mc_signal=mc_signal,
            mc_balance=mc_state.current_balance,
            mc_pnl=mc_state.total_pnl,
            mc_drawdown=mc_state.max_drawdown,
            mb_signal=mb_signal,
            mb_balance=mb_state.current_balance,
            mb_pnl=mb_state.total_pnl,
            mb_drawdown=mb_state.max_drawdown,
            current_price=price
        )
        
        self.comparison_history.append(result)
        
        return result
    
    def _execute_trade(self, strategy: StrategyType, symbol: str, price: float, signal: str, timestamp: datetime):
        """Esegue un trade per la strategia specificata."""
        state = self.states[strategy]
        
        if signal == "BUY" and state.position == 0:
            # Compra
            trade_value = state.current_balance * self.trade_size_pct
            quantity = trade_value / price
            
            state.position = quantity
            state.entry_price = price
            state.current_balance -= trade_value
            
            trade = Trade(
                timestamp=timestamp,
                strategy=strategy,
                symbol=symbol,
                side="BUY",
                price=price,
                quantity=quantity
            )
            state.trades.append(trade)
            state.total_trades += 1
            
            logger.debug(f"{strategy.value} BUY {symbol}: {quantity:.4f} @ {price:.2f}")
            
        elif signal == "SELL" and state.position > 0:
            # Vendi
            sell_value = state.position * price
            pnl = (price - state.entry_price) * state.position
            
            state.current_balance += sell_value
            state.total_pnl += pnl
            state.position = 0
            state.entry_price = 0
            
            if pnl > 0:
                state.winning_trades += 1
            else:
                state.losing_trades += 1
            
            trade = Trade(
                timestamp=timestamp,
                strategy=strategy,
                symbol=symbol,
                side="SELL",
                price=price,
                quantity=state.position,
                pnl=pnl,
                cumulative_pnl=state.total_pnl
            )
            state.trades.append(trade)
            
            logger.debug(f"{strategy.value} SELL {symbol}: PnL = {pnl:.2f}")
        
        # Registra balance history
        total_value = state.current_balance + (state.position * price if state.position > 0 else 0)
        state.balance_history.append(total_value)
    
    def _update_drawdown(self, strategy: StrategyType):
        """Aggiorna il drawdown per la strategia."""
        state = self.states[strategy]
        
        if len(state.balance_history) == 0:
            return
        
        current_value = state.balance_history[-1]
        
        if current_value > state.peak_balance:
            state.peak_balance = current_value
        
        drawdown = (state.peak_balance - current_value) / state.peak_balance
        state.drawdown_history.append(drawdown)
        
        if drawdown > state.max_drawdown:
            state.max_drawdown = drawdown
    
    def get_performance_summary(self) -> Dict:
        """
        Restituisce un riepilogo delle performance.
        
        Returns:
            Dizionario con metriche per entrambe le strategie
        """
        summary = {}
        
        for strategy_type in [StrategyType.MONTE_CARLO, StrategyType.MONT_BLANCK]:
            state = self.states[strategy_type]
            
            # Calcola metriche aggiuntive
            win_rate = state.winning_trades / state.total_trades if state.total_trades > 0 else 0
            
            avg_pnl = state.total_pnl / state.total_trades if state.total_trades > 0 else 0
            
            # Calcola Sharpe ratio semplificato
            if len(state.balance_history) > 1:
                returns = pd.Series(state.balance_history).pct_change().dropna()
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe = 0
            
            summary[strategy_type.value] = {
                "initial_balance": self.initial_balance,
                "current_balance": state.current_balance,
                "total_value": state.balance_history[-1] if state.balance_history else self.initial_balance,
                "total_pnl": state.total_pnl,
                "total_return_pct": (state.total_pnl / self.initial_balance) * 100,
                "total_trades": state.total_trades,
                "winning_trades": state.winning_trades,
                "losing_trades": state.losing_trades,
                "win_rate": win_rate,
                "avg_pnl_per_trade": avg_pnl,
                "max_drawdown": state.max_drawdown,
                "max_drawdown_pct": state.max_drawdown * 100,
                "sharpe_ratio": sharpe,
                "current_position": state.position,
                "entry_price": state.entry_price
            }
        
        # Determina il vincitore
        mc_return = summary["MonteCarlo"]["total_return_pct"]
        mb_return = summary["MontBlanck"]["total_return_pct"]
        
        summary["winner"] = "MonteCarlo" if mc_return > mb_return else "MontBlanck" if mb_return > mc_return else "Tie"
        summary["return_difference"] = abs(mc_return - mb_return)
        
        return summary
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Restituisce un DataFrame con lo storico dei confronti.
        
        Returns:
            DataFrame con colonne per entrambe le strategie
        """
        if not self.comparison_history:
            return pd.DataFrame()
        
        data = []
        for r in self.comparison_history:
            data.append({
                "timestamp": r.timestamp,
                "symbol": r.symbol,
                "price": r.current_price,
                "mc_signal": r.mc_signal,
                "mc_balance": r.mc_balance,
                "mc_pnl": r.mc_pnl,
                "mc_drawdown": r.mc_drawdown,
                "mb_signal": r.mb_signal,
                "mb_balance": r.mb_balance,
                "mb_pnl": r.mb_pnl,
                "mb_drawdown": r.mb_drawdown
            })
        
        return pd.DataFrame(data)
    
    def reset(self):
        """Resetta tutti gli stati."""
        for strategy_type in self.states:
            self.states[strategy_type] = StrategyState(initial_balance=self.initial_balance)
        
        self.comparison_history = []
        self.price_data = defaultdict(list)
        
        logger.info("StrategyComparisonEngine reset")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Comparison Engine")
    parser.add_argument("--test", "-t", action="store_true", help="Run test with sample data")
    args = parser.parse_args()
    
    if args.test:
        print("=" * 70)
        print("  STRATEGY COMPARISON ENGINE - TEST")
        print("=" * 70)
        
        # Crea engine
        engine = StrategyComparisonEngine(initial_balance=10000.0)
        
        # Simula prezzi
        np.random.seed(42)
        n_points = 100
        base_price = 100
        
        # Genera prezzi con trend e rumore
        trend = np.linspace(0, 10, n_points)
        noise = np.random.normal(0, 2, n_points)
        prices = base_price + trend + np.cumsum(noise) * 0.1
        
        print(f"\nSimulando {n_points} punti prezzo...")
        print(f"Prezzo iniziale: {prices[0]:.2f}")
        print(f"Prezzo finale: {prices[-1]:.2f}")
        
        # Esegui confronto
        for i, price in enumerate(prices):
            engine.update("TEST", price)
        
        # Mostra risultati
        summary = engine.get_performance_summary()
        
        print("\n" + "-" * 70)
        print("  PERFORMANCE SUMMARY")
        print("-" * 70)
        
        for strategy in ["MonteCarlo", "MontBlanck"]:
            s = summary[strategy]
            print(f"\n{strategy}:")
            print(f"  Balance:     ${s['current_balance']:.2f}")
            print(f"  Total PnL:   ${s['total_pnl']:.2f} ({s['total_return_pct']:.2f}%)")
            print(f"  Trades:      {s['total_trades']} (W: {s['winning_trades']}, L: {s['losing_trades']})")
            print(f"  Win Rate:    {s['win_rate']:.1%}")
            print(f"  Max DD:      {s['max_drawdown_pct']:.2f}%")
            print(f"  Sharpe:      {s['sharpe_ratio']:.2f}")
        
        print(f"\n{'=' * 70}")
        print(f"  WINNER: {summary['winner']} (diff: {summary['return_difference']:.2f}%)")
        print(f"{'=' * 70}")
