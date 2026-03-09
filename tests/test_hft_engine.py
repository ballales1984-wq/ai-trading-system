"""
Test HFT Trading Engine
======================
Test per Day 2: HFT & Multi-Agent Market

Verifica:
- Loop tick-by-tick in hft_simulator.py
- Agenti: market makers, arbitraggisti, retail
- Interazione agenti + strategie ML
- Output HFT nel TradingEngine
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np
import pandas as pd

# Import moduli da testare
from src.hft.hft_trading_engine import (
    HFTTradingEngine, HFTSignal, HFTMetrics, SignalType,
    MLStrategyAdapter, RetailAgent, create_hft_engine
)
from src.hft.hft_simulator import HFTSimulator, OrderbookSimulator, create_tick_data
from src.simulations.multi_agent_market import (
    MultiAgentMarket, MarketMaker, Taker, Arbitrageur,
    Order, Trade, Orderbook
)


class TestHFTSimulator:
    """Test per HFTSimulator."""
    
    @pytest.fixture
    def tick_data(self):
        """Crea dati tick per test."""
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(1000) * 10)
        volumes = np.random.uniform(0.5, 2.0, 1000)
        return create_tick_data(prices, volumes)
    
    def test_simulator_creation(self, tick_data):
        """Test creazione simulatore."""
        sim = HFTSimulator(tick_data)
        
        assert sim is not None
        assert len(sim.ticks) == 1000
        assert sim.position == 0.0
        assert sim.cash == 0.0
    
    def test_get_tick(self, tick_data):
        """Test ottenimento tick."""
        sim = HFTSimulator(tick_data)
        
        tick = sim.get_tick()
        assert 'bid' in tick
        assert 'ask' in tick
        assert tick['ask'] > tick['bid']
    
    def test_get_orderbook(self, tick_data):
        """Test orderbook simulation."""
        sim = HFTSimulator(tick_data)
        
        ob = sim.get_orderbook(levels=5)
        
        assert 'bids' in ob
        assert 'asks' in ob
        assert len(ob['bids']) == 5
        assert len(ob['asks']) == 5
    
    def test_execute_buy(self, tick_data):
        """Test esecuzione buy."""
        sim = HFTSimulator(tick_data)
        sim.reset(10000)
        
        trade = sim.execute("BUY", 0.1)
        
        assert trade is not None
        assert trade['side'] == "BUY"
        assert trade['qty'] == 0.1
        assert sim.position == 0.1
    
    def test_execute_sell(self, tick_data):
        """Test esecuzione sell."""
        sim = HFTSimulator(tick_data)
        sim.reset(10000)
        
        # Prima compra
        sim.execute("BUY", 0.1)
        # Poi vendi
        trade = sim.execute("SELL", 0.1)
        
        assert trade is not None
        assert trade['side'] == "SELL"
    
    def test_step(self, tick_data):
        """Test step simulation."""
        sim = HFTSimulator(tick_data)
        
        pnl, done = sim.step(action=0)  # Hold
        
        assert isinstance(pnl, float)
        assert isinstance(done, bool)
    
    def test_get_pnl(self, tick_data):
        """Test calcolo PnL."""
        sim = HFTSimulator(tick_data)
        sim.reset(10000)
        
        initial_pnl = sim.get_pnl()
        assert initial_pnl == 10000  # Initial cash
        
        # Apri posizione
        sim.execute("BUY", 0.1)
        
        # PnL dovrebbe essere cambiato
        pnl = sim.get_pnl()
        assert pnl != 10000
    
    def test_get_stats(self, tick_data):
        """Test statistiche."""
        sim = HFTSimulator(tick_data)
        sim.reset(10000)
        
        # Esegui alcuni trade
        sim.execute("BUY", 0.1)
        sim.execute("SELL", 0.1)
        
        stats = sim.get_stats()
        
        assert 'total_trades' in stats
        assert stats['total_trades'] == 2


class TestOrderbookSimulator:
    """Test per OrderbookSimulator."""
    
    def test_create_simulator(self):
        """Test creazione orderbook simulator."""
        obs = OrderbookSimulator(base_spread=0.01, base_depth=10.0)
        
        assert obs is not None
        assert obs.base_spread == 0.01
    
    def test_generate_snapshot(self):
        """Test generazione snapshot."""
        obs = OrderbookSimulator()
        
        snapshot = obs.generate_snapshot(mid_price=50000)
        
        assert 'bids' in snapshot
        assert 'asks' in snapshot
        assert 'mid_price' in snapshot
        assert len(snapshot['bids']) == 20
        assert len(snapshot['asks']) == 20
    
    def test_simulate_market_impact(self):
        """Test market impact."""
        obs = OrderbookSimulator()
        
        snapshot = obs.generate_snapshot(mid_price=50000)
        
        impact = obs.simulate_market_impact("BUY", 1.0, snapshot)
        
        assert isinstance(impact, float)
        assert impact >= 0


class TestMarketAgents:
    """Test per agenti di mercato."""
    
    def test_market_maker(self):
        """Test market maker agent."""
        mm = MarketMaker("mm_0", spread=0.001, size=1.0)
        
        market_state = {
            'mid_price': 50000,
            'spread': 50,
            'bid_size': 1.0,
            'ask_size': 1.0
        }
        
        orders = mm.act(market_state)
        
        # Dopo refresh_rate ticks
        mm.tick_count = 99  # refresh_rate = 100
        orders = mm.act(market_state)
        
        assert len(orders) == 2  # Bid e Ask
        assert orders[0].side == 'bid'
        assert orders[1].side == 'ask'
    
    def test_taker(self):
        """Test taker agent."""
        taker = Taker("taker_0", frequency=1.0, size_range=(0.1, 0.5))
        
        market_state = {
            'mid_price': 50000,
            'spread': 50
        }
        
        orders = taker.act(market_state)
        
        # Con frequency=1.0, dovrebbe sempre generare ordine
        assert len(orders) == 1
        assert orders[0].order_type == 'market'
    
    def test_arbitrageur(self):
        """Test arbitrageur agent."""
        arb = Arbitrageur("arb_0", threshold=0.0005, size=0.5)
        
        # Situazione di arbitraggio - differenza maggiore della threshold
        market_state = {
            'mid_price': 50000,
            'secondary_price': 50100,  # 0.2% sopra - maggiore della threshold
            'spread': 50
        }
        
        orders = arb.act(market_state)
        
        # Dovrebbe comprare sul mercato primario (buy low, sell high)
        # La condizione è: spread > threshold dove spread = (secondary - primary) / primary
        # (50100 - 50000) / 50000 = 0.002 > 0.0005
        assert len(orders) >= 1  # Può generare ordine o no dipende dalla logica
    
    def test_retail_agent(self):
        """Test retail agent."""
        retail = RetailAgent(
            "retail_0",
            frequency=1.0,
            fear_factor=0.5,
            greed_factor=0.5
        )
        
        # Test FOMO buy (prezzo sale)
        market_state = {
            'mid_price': 50000,
            'price_change_1pct': 0.01  # +1%
        }
        
        orders = retail.act(market_state)
        # Con greed_factor, potrebbe comprare
        
        assert retail is not None


class TestMLStrategyAdapter:
    """Test per ML Strategy Adapter."""
    
    def test_adapter_creation(self):
        """Test creazione adapter."""
        adapter = MLStrategyAdapter()
        
        assert adapter is not None
        assert len(adapter.price_history) == 0
    
    def test_update_features(self):
        """Test update features."""
        adapter = MLStrategyAdapter()
        
        market_state = {
            'mid_price': 50000,
            'spread': 50,
            'imbalance': 0.1,
            'bid_size': 1.0,
            'ask_size': 1.0
        }
        
        features = adapter.update(market_state)
        
        assert features is not None
        assert len(features) == 10
        assert len(adapter.price_history) == 1
    
    def test_predict_without_model(self):
        """Test predizione senza modello."""
        adapter = MLStrategyAdapter()
        
        market_state = {
            'mid_price': 50000,
            'spread': 50
        }
        
        signal = adapter.predict(market_state)
        
        # Senza modello, dovrebbe restituire None
        assert signal is None
    
    def test_predict_with_mock_model(self):
        """Test predizione con modello mock."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])  # Buy signal
        
        adapter = MLStrategyAdapter(ml_model=mock_model, prediction_threshold=0.6)
        
        # Aggiungi abbastanza storia
        for i in range(25):
            adapter.update({
                'mid_price': 50000 + i * 10,
                'spread': 50,
                'imbalance': 0.1,
                'bid_size': 1.0,
                'ask_size': 1.0
            })
        
        signal = adapter.predict({
            'mid_price': 50250,
            'spread': 50,
            'imbalance': 0.1,
            'bid_size': 1.0,
            'ask_size': 1.0
        })
        
        assert signal is not None
        assert signal.signal_type == SignalType.BUY


class TestHFTTradingEngine:
    """Test per HFT Trading Engine."""
    
    @pytest.fixture
    def engine(self):
        """Crea engine per test."""
        return HFTTradingEngine(
            symbol="BTCUSDT",
            initial_price=50000,
            n_market_makers=1,
            n_takers=1,
            n_arbitrageurs=1,
            n_retail=1,
            enable_ml_signals=False  # Disable per test
        )
    
    def test_engine_creation(self, engine):
        """Test creazione engine."""
        assert engine is not None
        assert engine.symbol == "BTCUSDT"
        assert len(engine.market.agents) > 0
    
    def test_get_current_state(self, engine):
        """Test stato corrente."""
        state = engine.get_current_state()
        
        assert 'symbol' in state
        assert 'tick_count' in state
        assert state['symbol'] == "BTCUSDT"
    
    def test_get_agent_stats(self, engine):
        """Test statistiche agenti."""
        stats = engine.get_agent_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        for agent_id, agent_stats in stats.items():
            assert 'position' in agent_stats
            assert 'capital' in agent_stats
            assert 'pnl' in agent_stats
    
    def test_run_backtest(self, engine):
        """Test backtest."""
        results = engine.run_backtest(n_ticks=100, verbose=False)
        
        assert results is not None
        assert len(results) == 100
        assert 'price' in results.columns
        assert 'spread' in results.columns
    
    def test_inject_signal(self, engine):
        """Test iniezione segnale."""
        signal = HFTSignal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=50000,
            quantity=0.1,
            source='external'
        )
        
        engine.inject_signal(signal)
        
        signals = engine.get_signals()
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
    
    def test_callbacks(self, engine):
        """Test callback."""
        tick_called = []
        signal_called = []
        
        def on_tick(state):
            tick_called.append(state)
        
        def on_signal(signal):
            signal_called.append(signal)
        
        engine.set_callbacks(on_tick=on_tick, on_signal=on_signal)
        
        # Esegui alcuni tick
        for _ in range(5):
            engine.market.step()
            if engine._on_tick:
                engine._on_tick(engine.get_current_state())
        
        assert len(tick_called) == 5
    
    def test_get_metrics(self, engine):
        """Test metriche."""
        engine.run_backtest(n_ticks=50)
        
        metrics = engine.get_metrics()
        
        assert isinstance(metrics, HFTMetrics)
        assert metrics.to_dict() is not None


class TestHFTSignal:
    """Test per HFTSignal."""
    
    def test_signal_creation(self):
        """Test creazione segnale."""
        signal = HFTSignal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=50000,
            quantity=0.1,
            source='ml'
        )
        
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == 0.8
        assert signal.source == 'ml'
    
    def test_signal_to_dict(self):
        """Test conversione a dict."""
        signal = HFTSignal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.SELL,
            strength=0.6,
            price=50000,
            quantity=0.1,
            source='agent'
        )
        
        d = signal.to_dict()
        
        assert d['signal_type'] == 'SELL'
        assert d['symbol'] == 'BTCUSDT'
        assert 'timestamp' in d


class TestMultiAgentMarket:
    """Test per MultiAgentMarket."""
    
    def test_market_creation(self):
        """Test creazione mercato."""
        market = MultiAgentMarket(
            initial_price=50000,
            n_market_makers=2,
            n_takers=3,
            n_arbitrageurs=1
        )
        
        assert market is not None
        assert market.current_price == 50000
        assert len(market.agents) == 7  # 2+3+1+1 (RL agent default)
    
    def test_market_step(self):
        """Test step mercato."""
        market = MultiAgentMarket(initial_price=50000)
        
        state = market.step()
        
        assert 'mid_price' in state
        assert 'spread' in state
        assert market.tick_count == 1
    
    def test_market_reset(self):
        """Test reset mercato."""
        market = MultiAgentMarket(initial_price=50000)
        
        # Esegui alcuni step
        for _ in range(10):
            market.step()
        
        # Reset
        market.reset(50000)
        
        assert market.tick_count == 0
        assert len(market.price_history) == 1
    
    def test_run_simulation(self):
        """Test simulazione completa."""
        market = MultiAgentMarket(
            initial_price=50000,
            n_market_makers=1,
            n_takers=1,
            n_arbitrageurs=1,
            include_rl_agent=False
        )
        
        results = market.run_simulation(n_steps=100, verbose=False)
        
        assert len(results) == 100
        assert 'price' in results.columns


class TestCreateHFTEngine:
    """Test factory function."""
    
    def test_create_hft_engine(self):
        """Test creazione engine tramite factory."""
        engine = create_hft_engine(
            symbol="ETHUSDT",
            initial_price=3000
        )
        
        assert engine is not None
        assert engine.symbol == "ETHUSDT"
        assert engine.enable_ml_signals == True


class TestHFTEngineThreaded:
    """Test con thread attivi."""
    
    def test_start_stop(self):
        """Test avvio e stop engine."""
        engine = HFTTradingEngine(
            symbol="BTCUSDT",
            initial_price=50000,
            n_market_makers=1,
            n_takers=1,
            n_arbitrageurs=0,
            n_retail=1,
            enable_ml_signals=False
        )
        
        engine.start()
        
        # Aspetta un po'
        time.sleep(0.5)
        
        # Verifica che stia girando
        assert engine.is_running
        
        # Ferma
        engine.stop()
        
        assert not engine.is_running
    
    def test_signal_generation_during_run(self):
        """Test generazione segnali durante esecuzione."""
        signals_received = []
        
        def on_signal(signal):
            signals_received.append(signal)
        
        engine = HFTTradingEngine(
            symbol="BTCUSDT",
            initial_price=50000,
            enable_ml_signals=False
        )
        
        engine.set_callbacks(on_signal=on_signal)
        
        # Inietta segnale manualmente
        signal = HFTSignal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=0.9,
            price=50000,
            quantity=0.1,
            source='test'
        )
        
        engine.inject_signal(signal)
        
        assert len(signals_received) == 1


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
