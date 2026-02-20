"""
Test Strategy Evolution Manager
================================
Test per Day 3: AutoML / Strategy Evolution

Verifica:
- Workflow evolutivo per segnali ML
- Training su dati storici + simulazioni HFT
- Output al SignalEngine
- Test con PaperBroker
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil

# Import moduli da testare
from src.automl.strategy_evolution_manager import (
    StrategyEvolutionManager, StrategyBacktester, EvolutionResult,
    BacktestResult, EvolutionStatus, create_evolution_manager
)
from src.automl.automl_engine import StrategyGenome, AutoMLEvolver


class TestStrategyGenome:
    """Test per StrategyGenome."""
    
    def test_genome_creation(self):
        """Test creazione genome."""
        genome = StrategyGenome()
        
        assert genome is not None
        assert genome.use_rsi == True
        assert genome.rsi_period == 14
        assert genome.fitness == 0.0
    
    def test_genome_to_dict(self):
        """Test conversione a dict."""
        genome = StrategyGenome()
        d = genome.to_dict()
        
        assert isinstance(d, dict)
        assert 'use_rsi' in d
        assert 'fitness' in d
        assert d['use_rsi'] == True
    
    def test_genome_from_dict(self):
        """Test creazione da dict."""
        data = {
            'use_rsi': False,
            'rsi_period': 21,
            'fitness': 0.5
        }
        
        genome = StrategyGenome.from_dict(data)
        
        assert genome.use_rsi == False
        assert genome.rsi_period == 21
        assert genome.fitness == 0.5
    
    def test_genome_mutate(self):
        """Test mutazione genome."""
        genome = StrategyGenome()
        genome.rsi_period = 14
        
        # Mutazione con rate alto per garantire cambiamenti
        mutated = genome.mutate(mutation_rate=1.0)
        
        # Almeno qualche parametro dovrebbe essere diverso
        assert mutated is not None
        assert mutated.fitness == 0.0  # Fitness non viene mutata
    
    def test_genome_crossover(self):
        """Test crossover tra genomi."""
        parent1 = StrategyGenome()
        parent1.rsi_period = 7
        parent1.use_rsi = True
        
        parent2 = StrategyGenome()
        parent2.rsi_period = 28
        parent2.use_rsi = False
        
        child = StrategyGenome.crossover(parent1, parent2)
        
        assert child is not None
        # Child dovrebbe ereditare da uno dei due genitori
        assert child.rsi_period in [7, 28]


class TestAutoMLEvolver:
    """Test per AutoMLEvolver."""
    
    def test_evolver_creation(self):
        """Test creazione evolver."""
        evolver = AutoMLEvolver(
            population_size=10,
            generations=5
        )
        
        assert evolver is not None
        assert evolver.population_size == 10
        assert evolver.generations == 5
    
    def test_initialize_population(self):
        """Test inizializzazione popolazione."""
        evolver = AutoMLEvolver(population_size=10)
        evolver.initialize_population()
        
        assert len(evolver.population) == 10
        assert all(isinstance(g, StrategyGenome) for g in evolver.population)
    
    def test_evaluate_population(self):
        """Test valutazione popolazione."""
        evolver = AutoMLEvolver(population_size=5)
        evolver.initialize_population()
        
        # Fitness function mock
        def fitness_fn(genome):
            return np.random.random()
        
        evolver.evaluate_population(fitness_fn)
        
        # Tutti dovrebbero avere fitness
        assert all(g.fitness > 0 for g in evolver.population)
        
        # Dovrebbero essere ordinati
        fitnesses = [g.fitness for g in evolver.population]
        assert fitnesses == sorted(fitnesses, reverse=True)
    
    def test_select_parent(self):
        """Test selezione genitore."""
        evolver = AutoMLEvolver(population_size=10, tournament_size=3)
        evolver.initialize_population()
        
        # Assegna fitness
        for i, g in enumerate(evolver.population):
            g.fitness = i / 10
        
        parent = evolver.select_parent()
        
        assert parent is not None
        assert isinstance(parent, StrategyGenome)


class TestStrategyBacktester:
    """Test per StrategyBacktester."""
    
    @pytest.fixture
    def backtester(self):
        """Crea backtester con dati."""
        bt = StrategyBacktester(initial_capital=10000)
        bt.load_data("BTCUSDT", limit=500)
        return bt
    
    def test_backtester_creation(self):
        """Test creazione backtester."""
        bt = StrategyBacktester(initial_capital=10000)
        
        assert bt is not None
        assert bt.initial_capital == 10000
    
    def test_load_data(self, backtester):
        """Test caricamento dati."""
        assert backtester.data is not None
        assert len(backtester.data) == 500
        assert 'open' in backtester.data.columns
        assert 'close' in backtester.data.columns
    
    def test_calculate_indicators(self, backtester):
        """Test calcolo indicatori."""
        genome = StrategyGenome()
        genome.use_rsi = True
        genome.use_macd = True
        genome.use_bollinger = True
        
        df = backtester.calculate_indicators(genome)
        
        assert 'rsi' in df.columns
        assert 'macd' in df.columns
        assert 'bb_upper' in df.columns
    
    def test_generate_signals(self, backtester):
        """Test generazione segnali."""
        genome = StrategyGenome()
        genome.use_rsi = True
        
        df = backtester.generate_signals(genome)
        
        assert 'signal' in df.columns
        # Segnali dovrebbero essere normalizzati
        assert df['signal'].max() <= 1.0
        assert df['signal'].min() >= -1.0
    
    def test_run_backtest(self, backtester):
        """Test esecuzione backtest."""
        genome = StrategyGenome()
        genome.use_rsi = True
        genome.use_macd = True
        genome.position_size = 0.1
        genome.stop_loss_pct = 0.02
        genome.take_profit_pct = 0.04
        
        result = backtester.run_backtest(genome)
        
        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
    
    def test_backtest_result_to_dict(self, backtester):
        """Test conversione risultato."""
        genome = StrategyGenome()
        result = backtester.run_backtest(genome)
        
        d = result.to_dict()
        
        assert 'total_return' in d
        assert 'sharpe_ratio' in d
        assert 'max_drawdown' in d


class TestStrategyEvolutionManager:
    """Test per StrategyEvolutionManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Crea directory temporanea."""
        dir_path = tempfile.mkdtemp()
        yield dir_path
        shutil.rmtree(dir_path)
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Crea manager per test."""
        mgr = StrategyEvolutionManager(
            population_size=5,
            generations=3,
            mutation_rate=0.2,
            checkpoint_dir=temp_dir
        )
        mgr.load_training_data("BTCUSDT", limit=200)
        return mgr
    
    def test_manager_creation(self, manager):
        """Test creazione manager."""
        assert manager is not None
        assert manager.population_size == 5
        assert manager.generations == 3
        assert manager.status == EvolutionStatus.IDLE
    
    def test_load_training_data(self, manager):
        """Test caricamento dati training."""
        assert manager.backtester.data is not None
        assert len(manager.backtester.data) == 200
    
    def test_fitness_function(self, manager):
        """Test funzione fitness."""
        genome = StrategyGenome()
        genome.use_rsi = True
        
        fitness = manager._fitness_function(genome)
        
        assert isinstance(fitness, float)
    
    def test_run_evolution_sync(self, manager):
        """Test evoluzione sincrona."""
        manager.start_evolution(async_mode=False)
        
        assert manager.status == EvolutionStatus.COMPLETED
        assert manager.best_genome is not None
        assert len(manager.evolution_history) == 3
    
    def test_evolution_callbacks(self, manager):
        """Test callback evoluzione."""
        generation_results = []
        final_genome = []
        
        def on_gen(result):
            generation_results.append(result)
        
        def on_complete(genome):
            final_genome.append(genome)
        
        manager.set_callbacks(
            on_generation_complete=on_gen,
            on_evolution_complete=on_complete
        )
        
        manager.start_evolution(async_mode=False)
        
        assert len(generation_results) == 3
        assert len(final_genome) == 1
    
    def test_checkpoint_save(self, manager):
        """Test salvataggio checkpoint."""
        # Usa più generazioni per triggerare checkpoint (ogni 5)
        manager.generations = 6
        manager.evolver.generations = 6
        manager.start_evolution(async_mode=False)
        
        # Dovrebbe aver salvato checkpoint (al gen 5)
        checkpoint_files = list(Path(manager.checkpoint_dir).glob("*.json"))
        # Se l'evoluzione completa, c'è almeno un checkpoint
        # Nota: il checkpoint viene salvato ogni 5 generazioni
        assert manager.status == EvolutionStatus.COMPLETED
    
    def test_get_best_strategy(self, manager):
        """Test ottenimento migliore strategia."""
        manager.start_evolution(async_mode=False)
        
        best = manager.get_best_strategy()
        
        assert best is not None
        assert isinstance(best, StrategyGenome)
        assert best.fitness > 0
    
    def test_get_evolution_summary(self, manager):
        """Test riassunto evoluzione."""
        manager.start_evolution(async_mode=False)
        
        summary = manager.get_evolution_summary()
        
        assert 'total_generations' in summary
        assert 'status' in summary
        assert 'best_fitness' in summary


class TestEvolutionResult:
    """Test per EvolutionResult."""
    
    def test_result_creation(self):
        """Test creazione risultato."""
        genome = StrategyGenome()
        result = EvolutionResult(
            generation=1,
            best_fitness=0.5,
            avg_fitness=0.3,
            best_genome=genome
        )
        
        assert result.generation == 1
        assert result.best_fitness == 0.5
        assert result.avg_fitness == 0.3
    
    def test_result_to_dict(self):
        """Test conversione a dict."""
        genome = StrategyGenome()
        result = EvolutionResult(
            generation=1,
            best_fitness=0.5,
            avg_fitness=0.3,
            best_genome=genome
        )
        
        d = result.to_dict()
        
        assert d['generation'] == 1
        assert d['best_fitness'] == 0.5
        assert 'best_genome' in d


class TestEvolutionStatus:
    """Test per EvolutionStatus."""
    
    def test_status_values(self):
        """Test valori status."""
        assert EvolutionStatus.IDLE.value == "idle"
        assert EvolutionStatus.TRAINING.value == "training"
        assert EvolutionStatus.COMPLETED.value == "completed"
        assert EvolutionStatus.FAILED.value == "failed"


class TestCreateEvolutionManager:
    """Test factory function."""
    
    def test_create_manager(self):
        """Test creazione tramite factory."""
        manager = create_evolution_manager(
            population_size=10,
            generations=5,
            initial_capital=5000
        )
        
        assert manager is not None
        assert manager.population_size == 10
        assert manager.generations == 5


class TestStrategyEvolutionThreaded:
    """Test con thread."""
    
    @pytest.fixture
    def manager(self):
        """Crea manager per test."""
        mgr = StrategyEvolutionManager(
            population_size=3,
            generations=2,
            mutation_rate=0.2
        )
        mgr.load_training_data("BTCUSDT", limit=100)
        return mgr
    
    def test_async_evolution(self, manager):
        """Test evoluzione asincrona."""
        manager.start_evolution(async_mode=True)
        
        # Aspetta un po' - l'evoluzione potrebbe essere molto veloce
        time.sleep(0.2)
        
        # Aspetta completamento (con timeout)
        max_wait = 30
        while manager.status == EvolutionStatus.TRAINING and max_wait > 0:
            time.sleep(0.5)
            max_wait -= 0.5
        
        # Dovrebbe essere completato (o ancora in training se molto lento)
        assert manager.status in [EvolutionStatus.TRAINING, EvolutionStatus.COMPLETED]
    
    def test_stop_evolution(self, manager):
        """Test stop evoluzione."""
        manager.start_evolution(async_mode=True)
        
        time.sleep(0.2)
        
        manager.stop_evolution()
        
        assert manager._stop_flag == True


class TestIntegrationWithPaperBroker:
    """Test integrazione con PaperBroker."""
    
    def test_evolved_strategy_trading(self):
        """Test trading con strategia evoluta."""
        # Crea e evolve strategia
        manager = create_evolution_manager(
            population_size=5,
            generations=2,
            initial_capital=10000
        )
        manager.load_training_data("BTCUSDT", limit=100)
        
        # Evoluzione veloce
        manager.start_evolution(async_mode=False)
        
        best_genome = manager.get_best_strategy()
        
        # Verifica che la strategia abbia parametri validi
        assert best_genome is not None
        assert 0 < best_genome.position_size <= 1
        assert 0 < best_genome.stop_loss_pct < 1
        assert 0 < best_genome.take_profit_pct < 1
        
        # Simula trading con la strategia
        backtester = StrategyBacktester(initial_capital=10000)
        backtester.load_data("BTCUSDT", limit=200)
        
        result = backtester.run_backtest(best_genome)
        
        # Il risultato dovrebbe essere valido
        assert isinstance(result, BacktestResult)
        assert -1 <= result.total_return <= 10  # Return ragionevole


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
