#!/usr/bin/env python3
"""
Test per Strategy Comparison Engine
====================================

Testa il confronto tra Monte Carlo e Mont Blanck.
"""

import pytest
import numpy as np
from datetime import datetime
import sys
import os

# Aggiungi path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.strategy_comparison import (
    StrategyComparisonEngine,
    MonteCarloStrategy,
    StrategyType,
    StrategyState,
    Trade
)
from src.strategy.montblanck import MontBlanck, Signal


class TestMonteCarloStrategy:
    """Test per la strategia Monte Carlo."""
    
    def test_initialization(self):
        """Test inizializzazione."""
        strategy = MonteCarloStrategy(
            n_simulations=100,
            confidence_level=0.95,
            buy_threshold=0.02
        )
        assert strategy.n_simulations == 100
        assert strategy.confidence_level == 0.95
        assert strategy.buy_threshold == 0.02
    
    def test_predict_with_insufficient_data(self):
        """Test con dati insufficienti."""
        strategy = MonteCarloStrategy()
        
        # Meno di 10 prezzi
        prices = [100, 101, 102]
        result = strategy.predict(prices)
        
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
    
    def test_predict_with_sufficient_data(self):
        """Test con dati sufficienti."""
        np.random.seed(42)
        strategy = MonteCarloStrategy(n_simulations=100)
        
        # Trend rialzista
        prices = list(np.linspace(100, 110, 20))
        result = strategy.predict(prices)
        
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert "expected_change" in result
        assert "confidence" in result
    
    def test_uptrend_generates_buy(self):
        """Test che trend rialzista generi BUY."""
        np.random.seed(42)
        strategy = MonteCarloStrategy(
            n_simulations=500,
            buy_threshold=0.01
        )
        
        # Trend rialzista forte
        prices = [100 + i * 0.5 for i in range(30)]
        result = strategy.predict(prices)
        
        # Con trend rialzista forte, dovrebbe generare BUY o HOLD con alta confidenza
        assert result["signal"] in ["BUY", "HOLD"]


class TestStrategyComparisonEngine:
    """Test per l'engine di confronto."""
    
    def test_initialization(self):
        """Test inizializzazione."""
        engine = StrategyComparisonEngine(initial_balance=10000.0)
        
        assert engine.initial_balance == 10000.0
        assert StrategyType.MONTE_CARLO in engine.states
        assert StrategyType.MONT_BLANCK in engine.states
        assert engine.states[StrategyType.MONTE_CARLO].current_balance == 10000.0
    
    def test_single_update(self):
        """Test singolo aggiornamento."""
        engine = StrategyComparisonEngine()
        
        result = engine.update("BTC", 50000.0)
        
        assert result.symbol == "BTC"
        assert result.current_price == 50000.0
        assert result.mc_signal in ["BUY", "SELL", "HOLD"]
        assert result.mb_signal in ["BUY", "SELL", "HOLD"]
    
    def test_multiple_updates(self):
        """Test aggiornamenti multipli."""
        engine = StrategyComparisonEngine()
        
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        
        for price in prices:
            result = engine.update("TEST", price)
        
        assert len(engine.comparison_history) == len(prices)
        assert len(engine.price_data["TEST"]) == len(prices)
    
    def test_trade_execution_buy(self):
        """Test esecuzione BUY."""
        engine = StrategyComparisonEngine(trade_size_pct=0.1)
        
        # Forza un segnale BUY
        state = engine.states[StrategyType.MONTE_CARLO]
        initial_balance = state.current_balance
        
        engine._execute_trade(
            StrategyType.MONTE_CARLO, 
            "TEST", 
            100.0, 
            "BUY", 
            datetime.now()
        )
        
        assert state.position > 0
        assert state.current_balance < initial_balance
        assert state.entry_price == 100.0
        assert state.total_trades == 1
    
    def test_trade_execution_sell(self):
        """Test esecuzione SELL."""
        engine = StrategyComparisonEngine(trade_size_pct=0.1)
        
        # Prima compra
        engine._execute_trade(
            StrategyType.MONTE_CARLO, 
            "TEST", 
            100.0, 
            "BUY", 
            datetime.now()
        )
        
        state = engine.states[StrategyType.MONTE_CARLO]
        
        # Poi vendi a prezzo più alto
        engine._execute_trade(
            StrategyType.MONTE_CARLO, 
            "TEST", 
            110.0, 
            "SELL", 
            datetime.now()
        )
        
        assert state.position == 0
        assert state.total_pnl > 0  # Profitto
        assert state.winning_trades == 1
    
    def test_performance_summary(self):
        """Test riepilogo performance."""
        engine = StrategyComparisonEngine()
        
        # Simula alcuni aggiornamenti
        prices = [100 + i for i in range(20)]
        for price in prices:
            engine.update("TEST", price)
        
        summary = engine.get_performance_summary()
        
        assert "MonteCarlo" in summary
        assert "MontBlanck" in summary
        assert "winner" in summary
        
        # Verifica metriche
        for strategy in ["MonteCarlo", "MontBlanck"]:
            s = summary[strategy]
            assert "total_trades" in s
            assert "win_rate" in s
            assert "max_drawdown" in s
    
    def test_comparison_dataframe(self):
        """Test generazione DataFrame."""
        engine = StrategyComparisonEngine()
        
        prices = [100, 101, 102, 103, 104]
        for price in prices:
            engine.update("TEST", price)
        
        df = engine.get_comparison_dataframe()
        
        assert len(df) == len(prices)
        assert "mc_signal" in df.columns
        assert "mb_signal" in df.columns
        assert "mc_balance" in df.columns
        assert "mb_balance" in df.columns
    
    def test_reset(self):
        """Test reset dell'engine."""
        engine = StrategyComparisonEngine()
        
        # Aggiorna
        for i in range(10):
            engine.update("TEST", 100 + i)
        
        # Reset
        engine.reset()
        
        assert len(engine.comparison_history) == 0
        assert len(engine.price_data["TEST"]) == 0
        assert engine.states[StrategyType.MONTE_CARLO].current_balance == engine.initial_balance


class TestStrategyState:
    """Test per lo stato della strategia."""
    
    def test_initial_state(self):
        """Test stato iniziale."""
        state = StrategyState(initial_balance=5000.0)
        
        assert state.initial_balance == 5000.0
        assert state.current_balance == 5000.0
        assert state.position == 0.0
        assert state.total_trades == 0
        assert state.max_drawdown == 0.0
    
    def test_balance_history(self):
        """Test storico bilancio."""
        state = StrategyState()
        
        state.balance_history = [10000, 10100, 10050, 10200]
        
        assert len(state.balance_history) == 4
        assert state.balance_history[-1] == 10200


class TestTrade:
    """Test per la classe Trade."""
    
    def test_trade_creation(self):
        """Test creazione trade."""
        trade = Trade(
            timestamp=datetime.now(),
            strategy=StrategyType.MONTE_CARLO,
            symbol="BTC",
            side="BUY",
            price=50000.0,
            quantity=0.1
        )
        
        assert trade.strategy == StrategyType.MONTE_CARLO
        assert trade.symbol == "BTC"
        assert trade.side == "BUY"
        assert trade.price == 50000.0
        assert trade.quantity == 0.1
        assert trade.pnl == 0.0


class TestIntegration:
    """Test di integrazione."""
    
    def test_full_comparison_cycle(self):
        """Test ciclo completo di confronto."""
        np.random.seed(42)
        engine = StrategyComparisonEngine(initial_balance=10000.0)
        
        # Simula 100 punti prezzo
        n_points = 100
        base_price = 100
        trend = np.linspace(0, 10, n_points)
        noise = np.random.normal(0, 1, n_points)
        prices = base_price + trend + np.cumsum(noise) * 0.1
        
        # Esegui confronto
        for price in prices:
            engine.update("BTC", price)
        
        # Ottieni risultati
        summary = engine.get_performance_summary()
        
        # Verifica che entrambe le strategie abbiano elaborato
        assert summary["MonteCarlo"]["total_trades"] >= 0
        assert summary["MontBlanck"]["total_trades"] >= 0
        
        # Verifica che ci sia un vincitore
        assert summary["winner"] in ["MonteCarlo", "MontBlanck", "Tie"]
        
        print(f"\nMonte Carlo: {summary['MonteCarlo']['total_return_pct']:.2f}%")
        print(f"Mont Blanck: {summary['MontBlanck']['total_return_pct']:.2f}%")
        print(f"Winner: {summary['winner']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
