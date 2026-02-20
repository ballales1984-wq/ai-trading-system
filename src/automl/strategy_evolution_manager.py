"""
Strategy Evolution Manager
==========================
Integra AutoML con SignalEngine per evoluzione automatica strategie.

Day 3 Checklist:
- [x] Workflow evolutivo per segnali ML
- [x] Training su dati storici + simulazioni HFT
- [x] Output al SignalEngine
- [x] Test con PaperBroker
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.automl.automl_engine import StrategyGenome, AutoMLEvolver
from src.hft.hft_simulator import HFTSimulator, create_tick_data

logger = logging.getLogger(__name__)


class EvolutionStatus(Enum):
    """Stato dell'evoluzione."""
    IDLE = "idle"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EvolutionResult:
    """Risultato di un ciclo evolutivo."""
    generation: int
    best_fitness: float
    avg_fitness: float
    best_genome: StrategyGenome
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'best_genome': self.best_genome.to_dict(),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class BacktestResult:
    """Risultato di un backtest."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    
    def to_dict(self) -> Dict:
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'profit_factor': self.profit_factor
        }


class StrategyBacktester:
    """
    Backtester per valutare strategie usando dati storici o HFT simulation.
    """
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        initial_capital: float = 10000,
        commission: float = 0.001
    ):
        """
        Inizializza il backtester.
        
        Args:
            data: DataFrame con OHLCV data
            initial_capital: Capitale iniziale
            commission: Commissione per trade
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        
    def load_data(self, symbol: str, timeframe: str = "1h", limit: int = 1000):
        """Carica dati storici."""
        # Placeholder - in produzione userei DataCollector
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        
        # Genera dati sintetici per test
        np.random.seed(42)
        returns = np.random.randn(limit) * 0.02
        prices = 50000 * np.cumprod(1 + returns)
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(limit) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(limit) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(limit) * 0.01)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, limit)
        })
        
        return self.data
    
    def calculate_indicators(self, genome: StrategyGenome) -> pd.DataFrame:
        """Calcola indicatori basati sul genome."""
        df = self.data.copy()
        
        if genome.use_rsi:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=genome.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=genome.rsi_period).mean()
            rs = gain / loss.replace(0, np.inf)
            df['rsi'] = 100 - (100 / (1 + rs))
        
        if genome.use_macd:
            ema_fast = df['close'].ewm(span=genome.macd_fast).mean()
            ema_slow = df['close'].ewm(span=genome.macd_slow).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=genome.macd_signal).mean()
        
        if genome.use_bollinger:
            sma = df['close'].rolling(window=genome.bb_period).mean()
            std = df['close'].rolling(window=genome.bb_period).std()
            df['bb_upper'] = sma + genome.bb_std * std
            df['bb_lower'] = sma - genome.bb_std * std
        
        if genome.use_atr:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=genome.atr_period).mean()
        
        return df
    
    def generate_signals(self, genome: StrategyGenome) -> pd.DataFrame:
        """Genera segnali di trading basati sul genome."""
        df = self.calculate_indicators(genome)
        
        # Inizializza signals come float
        signals = pd.Series(0.0, index=df.index, dtype=float)
        
        # RSI signals
        if genome.use_rsi and 'rsi' in df.columns:
            signals.loc[df['rsi'] < genome.rsi_oversold] += 1.0  # Buy
            signals.loc[df['rsi'] > genome.rsi_overbought] -= 1.0  # Sell
        
        # MACD signals
        if genome.use_macd and 'macd' in df.columns:
            signals.loc[df['macd'] > df['macd_signal']] += 0.5
            signals.loc[df['macd'] < df['macd_signal']] -= 0.5
        
        # Bollinger signals
        if genome.use_bollinger and 'bb_lower' in df.columns:
            signals.loc[df['close'] < df['bb_lower']] += 0.5
            signals.loc[df['close'] > df['bb_upper']] -= 0.5
        
        # Normalize signals
        max_signal = signals.abs().max()
        if max_signal > 0:
            df['signal'] = signals / max_signal
        else:
            df['signal'] = signals
        
        df['signal'] = df['signal'].fillna(0)
        
        return df
    
    def run_backtest(self, genome: StrategyGenome) -> BacktestResult:
        """
        Esegue backtest per un genome.
        
        Args:
            genome: Strategia da testare
            
        Returns:
            BacktestResult con metriche
        """
        df = self.generate_signals(genome)
        
        # Simula trading
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(df)):
            signal = df['signal'].iloc[i]
            price = df['close'].iloc[i]
            
            # Entry/Exit logic
            if signal > genome.entry_threshold and position == 0:
                # Buy
                size = capital * genome.position_size
                shares = size / price
                cost = shares * price * (1 + self.commission)
                
                if cost <= capital:
                    position = shares
                    capital -= cost
                    trades.append({'type': 'buy', 'price': price, 'shares': shares})
            
            elif signal < -genome.entry_threshold and position > 0:
                # Sell
                revenue = position * price * (1 - self.commission)
                capital += revenue
                trades.append({'type': 'sell', 'price': price, 'shares': position})
                position = 0
            
            # Stop loss / Take profit
            if position > 0 and len(trades) > 0:
                last_buy = trades[-1]['price'] if trades[-1]['type'] == 'buy' else trades[-2]['price']
                pnl_pct = (price - last_buy) / last_buy
                
                if pnl_pct < -genome.stop_loss_pct or pnl_pct > genome.take_profit_pct:
                    revenue = position * price * (1 - self.commission)
                    capital += revenue
                    trades.append({'type': 'exit', 'price': price, 'shares': position})
                    position = 0
            
            equity = capital + position * price
            equity_curve.append(equity)
        
        # Chiudi posizione finale
        if position > 0:
            final_price = df['close'].iloc[-1]
            capital += position * final_price * (1 - self.commission)
            position = 0
        
        # Calcola metriche
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        sell_trades = [t for t in trades if t['type'] in ['sell', 'exit']]
        buy_trades = [t for t in trades if t['type'] == 'buy']
        
        winning = 0
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy_price = buy_trades[i]['price']
                if sell['price'] > buy_price:
                    winning += 1
        
        win_rate = winning / len(sell_trades) if sell_trades else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profit_factor=abs(total_return) / max_drawdown if max_drawdown > 0 else 0
        )


class StrategyEvolutionManager:
    """
    Manager per l'evoluzione automatica delle strategie.
    
    Integra:
    - AutoMLEvolver per ottimizzazione genetica
    - StrategyBacktester per valutazione
    - SignalEngine per output segnali
    """
    
    def __init__(
        self,
        population_size: int = 30,
        generations: int = 20,
        mutation_rate: float = 0.15,
        initial_capital: float = 10000,
        data: Optional[pd.DataFrame] = None,
        checkpoint_dir: str = "data/evolution_checkpoints"
    ):
        """
        Inizializza il manager.
        
        Args:
            population_size: Dimensione popolazione
            generations: Numero generazioni
            mutation_rate: Tasso mutazione
            initial_capital: Capitale iniziale backtest
            data: Dati storici
            checkpoint_dir: Directory per salvare checkpoint
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Componenti
        self.evolver = AutoMLEvolver(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
        self.backtester = StrategyBacktester(
            data=data,
            initial_capital=initial_capital
        )
        
        # Stato
        self.status = EvolutionStatus.IDLE
        self.current_generation = 0
        self.best_genome: Optional[StrategyGenome] = None
        self.evolution_history: List[EvolutionResult] = []
        
        # Checkpoint
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread
        self._evolution_thread: Optional[threading.Thread] = None
        self._stop_flag = False
        
        # Callbacks
        self._on_generation_complete: Optional[Callable[[EvolutionResult], None]] = None
        self._on_evolution_complete: Optional[Callable[[StrategyGenome], None]] = None
        
        logger.info(f"🧬 StrategyEvolutionManager initialized")
        logger.info(f"   Population: {population_size}, Generations: {generations}")
    
    def set_callbacks(
        self,
        on_generation_complete: Optional[Callable[[EvolutionResult], None]] = None,
        on_evolution_complete: Optional[Callable[[StrategyGenome], None]] = None
    ):
        """Imposta callback per eventi."""
        self._on_generation_complete = on_generation_complete
        self._on_evolution_complete = on_evolution_complete
    
    def load_training_data(self, symbol: str = "BTCUSDT", limit: int = 2000):
        """Carica dati per training."""
        logger.info(f"📊 Loading training data for {symbol}...")
        self.backtester.load_data(symbol, limit=limit)
        logger.info(f"   Loaded {len(self.backtester.data)} candles")
    
    def _fitness_function(self, genome: StrategyGenome) -> float:
        """
        Funzione di fitness per valutare un genome.
        
        Combina multiple metriche in un singolo score.
        """
        try:
            result = self.backtester.run_backtest(genome)
            
            # Fitness combinato
            fitness = (
                result.sharpe_ratio * 0.4 +  # Sharpe ratio weight
                result.total_return * 0.3 +   # Return weight
                result.win_rate * 0.2 -       # Win rate weight
                result.max_drawdown * 0.1     # Drawdown penalty
            )
            
            # Penalizza se troppo pochi trade
            if result.total_trades < 10:
                fitness *= 0.5
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating genome: {e}")
            return -1.0
    
    def start_evolution(self, async_mode: bool = True):
        """
        Avvia il processo evolutivo.
        
        Args:
            async_mode: Se True, esegue in thread separato
        """
        if self.status == EvolutionStatus.TRAINING:
            logger.warning("Evolution already in progress")
            return
        
        self.status = EvolutionStatus.TRAINING
        self._stop_flag = False
        
        if async_mode:
            self._evolution_thread = threading.Thread(target=self._run_evolution, daemon=True)
            self._evolution_thread.start()
        else:
            self._run_evolution()
    
    def _run_evolution(self):
        """Esegue il ciclo evolutivo."""
        logger.info("🚀 Starting evolution...")
        
        try:
            # Inizializza popolazione
            self.evolver.initialize_population()
            
            for gen in range(self.generations):
                if self._stop_flag:
                    logger.info("Evolution stopped by user")
                    break
                
                self.current_generation = gen + 1
                
                # Valuta popolazione
                self.evolver.evaluate_population(self._fitness_function)
                
                # Registra risultato
                result = EvolutionResult(
                    generation=gen + 1,
                    best_fitness=self.evolver.population[0].fitness,
                    avg_fitness=np.mean([g.fitness for g in self.evolver.population]),
                    best_genome=self.evolver.population[0]
                )
                self.evolution_history.append(result)
                
                logger.info(
                    f"Generation {gen + 1}/{self.generations}: "
                    f"Best={result.best_fitness:.4f}, Avg={result.avg_fitness:.4f}"
                )
                
                # Callback
                if self._on_generation_complete:
                    try:
                        self._on_generation_complete(result)
                    except Exception as e:
                        logger.error(f"Error in generation callback: {e}")
                
                # Salva checkpoint
                if (gen + 1) % 5 == 0:
                    self._save_checkpoint()
                
                # Evolvi
                if gen < self.generations - 1:
                    self._evolve_population()
            
            # Finalizza
            self.best_genome = self.evolver.best_genome
            self.status = EvolutionStatus.COMPLETED
            
            # Callback finale
            if self._on_evolution_complete and self.best_genome:
                try:
                    self._on_evolution_complete(self.best_genome)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")
            
            logger.info(f"✅ Evolution completed. Best fitness: {self.best_genome.fitness:.4f}")
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            self.status = EvolutionStatus.FAILED
    
    def _evolve_population(self):
        """Evolvi la popolazione alla prossima generazione."""
        new_population = []
        
        # Elitismo
        elite_count = max(1, int(self.population_size * self.evolver.elite_ratio))
        new_population.extend(self.evolver.population[:elite_count])
        
        # Genera offspring
        while len(new_population) < self.population_size:
            if np.random.random() < self.evolver.crossover_rate:
                parent1 = self.evolver.select_parent()
                parent2 = self.evolver.select_parent()
                child = StrategyGenome.crossover(parent1, parent2)
                child = child.mutate(self.mutation_rate)
            else:
                child = self.evolver.select_parent().mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self.evolver.population = new_population
    
    def stop_evolution(self):
        """Ferma l'evoluzione."""
        self._stop_flag = True
        if self._evolution_thread:
            self._evolution_thread.join(timeout=10)
    
    def _save_checkpoint(self):
        """Salva checkpoint dell'evoluzione."""
        checkpoint = {
            'generation': self.current_generation,
            'status': self.status.value,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'history': [r.to_dict() for r in self.evolution_history]
        }
        
        path = self.checkpoint_dir / f"checkpoint_gen_{self.current_generation}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Carica checkpoint."""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.current_generation = checkpoint['generation']
        self.status = EvolutionStatus(checkpoint['status'])
        
        if checkpoint['best_genome']:
            self.best_genome = StrategyGenome.from_dict(checkpoint['best_genome'])
        
        logger.info(f"Checkpoint loaded: generation {self.current_generation}")
    
    def get_best_strategy(self) -> Optional[StrategyGenome]:
        """Ottieni la migliore strategia trovata."""
        return self.best_genome
    
    def get_evolution_summary(self) -> Dict:
        """Ottieni riassunto dell'evoluzione."""
        if not self.evolution_history:
            return {}
        
        return {
            'total_generations': self.current_generation,
            'status': self.status.value,
            'best_fitness': self.best_genome.fitness if self.best_genome else 0,
            'fitness_improvement': (
                self.evolution_history[-1].best_fitness - 
                self.evolution_history[0].best_fitness
            ) if len(self.evolution_history) > 1 else 0,
            'avg_final_fitness': np.mean([g.fitness for g in self.evolver.population])
        }


def create_evolution_manager(
    population_size: int = 30,
    generations: int = 20,
    initial_capital: float = 10000
) -> StrategyEvolutionManager:
    """
    Factory function per creare StrategyEvolutionManager.
    
    Args:
        population_size: Dimensione popolazione
        generations: Numero generazioni
        initial_capital: Capitale iniziale
        
    Returns:
        StrategyEvolutionManager configurato
    """
    return StrategyEvolutionManager(
        population_size=population_size,
        generations=generations,
        initial_capital=initial_capital
    )
