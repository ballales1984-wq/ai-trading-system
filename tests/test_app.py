"""
Test Suite for Quantum AI Trading System
Comprehensive tests for all main modules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTechnicalAnalysis:
    """Test Technical Analysis module"""
    
    def setup_method(self):
        """Setup test data"""
        from technical_analysis import TechnicalAnalyzer
        self.analyzer = TechnicalAnalyzer()
        
        # Create sample OHLCV data as DataFrame with proper columns
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 50000, 100),
            'high': np.random.uniform(45000, 51000, 100),
            'low': np.random.uniform(44000, 45000, 100),
            'close': np.random.uniform(45000, 50000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        # Ensure high is highest and low is lowest
        self.sample_data['high'] = self.sample_data[['open', 'high', 'close']].max(axis=1) + 100
        self.sample_data['low'] = self.sample_data[['open', 'low', 'close']].min(axis=1) - 100
    
    def test_initialization(self):
        """Test technical analyzer initialization"""
        from technical_analysis import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        assert analyzer is not None
    
    def test_rsi_calculation(self):
        """Test RSI indicator calculation"""
        rsi = self.analyzer.calculate_rsi(self.sample_data)
        assert rsi is not None
    
    def test_macd_calculation(self):
        """Test MACD indicator calculation"""
        macd = self.analyzer.calculate_macd(self.sample_data)
        assert macd is not None
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        bb = self.analyzer.calculate_bollinger_bands(self.sample_data)
        assert bb is not None
    
    def test_atr_calculation(self):
        """Test ATR indicator calculation"""
        atr = self.analyzer.calculate_atr(self.sample_data)
        assert atr is not None


class TestSentimentAnalysis:
    """Test Sentiment Analysis module"""
    
    def setup_method(self):
        """Setup sentiment analyzer"""
        from sentiment_news import SentimentAnalyzer
        self.analyzer = SentimentAnalyzer()
    
    def test_initialization(self):
        """Test sentiment analyzer initialization"""
        assert self.analyzer is not None
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis method"""
        assert hasattr(self.analyzer, 'analyze_asset_sentiment')


class TestDataCollector:
    """Test Data Collector module"""
    
    def setup_method(self):
        """Setup data collector"""
        from data_collector import DataCollector
        self.collector = DataCollector(simulation=True)
    
    def test_initialization(self):
        """Test data collector initialization"""
        assert self.collector is not None
        assert self.collector.simulation is True
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        df = self.collector.fetch_ohlcv('BTCUSDT', '1h', limit=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'close' in df.columns


class TestDecisionEngine:
    """Test Decision Engine module"""
    
    def setup_method(self):
        """Setup decision engine"""
        from decision_engine import DecisionEngine
        self.engine = DecisionEngine()
    
    def test_initialization(self):
        """Test decision engine initialization"""
        assert self.engine is not None
    
    def test_generate_signals_method(self):
        """Test generate_signals method exists"""
        assert hasattr(self.engine, 'generate_signals')


class TestTradingSimulator:
    """Test Trading Simulator module"""
    
    def setup_method(self):
        """Setup trading simulator"""
        from trading_simulator import TradingSimulator
        self.simulator = TradingSimulator(initial_balance=10000)
    
    def test_initialization(self):
        """Test simulator initialization"""
        assert self.simulator is not None
    
    def test_get_total_value(self):
        """Test get_total_value method"""
        # Check for any get method
        assert hasattr(self.simulator, 'get_portfolio_state') or hasattr(self.simulator, 'check_portfolio')


class TestLiveMultiAssetTrader:
    """Test Live Multi-Asset Trading module"""
    
    def setup_method(self):
        """Setup live trader"""
        from live_multi_asset import LiveMultiAssetTrader
        self.trader = LiveMultiAssetTrader(
            assets=['BTCUSDT', 'ETHUSDT'],
            initial_capital=10000,
            paper_trading=True
        )
    
    def test_initialization(self):
        """Test trader initialization"""
        assert self.trader is not None
        assert len(self.trader.assets) == 2
        assert 'BTCUSDT' in self.trader.assets
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        assert self.trader.portfolio is not None


class TestTelegramNotifier:
    """Test Telegram Notifier module"""
    
    def test_initialization(self):
        """Test telegram notifier initialization (without real connection)"""
        from src.live.telegram_notifier import TelegramNotifier
        
        # Create with dummy credentials (disabled)
        notifier = TelegramNotifier(
            bot_token='dummy_token',
            chat_id='dummy_chat',
            enabled=False
        )
        assert notifier is not None
        assert notifier.enabled is False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from src.live.telegram_notifier import TelegramNotifier
        
        notifier = TelegramNotifier(
            bot_token='dummy_token',
            chat_id='dummy_chat',
            enabled=False,
            rate_limit=5
        )
        # Test can_send (should be False since disabled)
        assert notifier._can_send() is False


class TestDashboard:
    """Test Dashboard module"""
    
    def test_dashboard_import(self):
        """Test dashboard can be imported"""
        try:
            import dashboard
            assert True
        except ImportError:
            pytest.skip("Dashboard requires GUI environment")


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


class TestHFTSimulator:
    """Test HFT Simulator module"""
    
    def test_initialization(self):
        """Test HFT simulator initialization"""
        from src.hft.hft_simulator import HFTSimulator
        import pandas as pd
        
        # Create sample tick data
        ticks = pd.DataFrame({
            'bid': [50000, 50100, 50200],
            'ask': [50010, 50110, 50210],
            'bid_size': [1.0, 1.5, 2.0],
            'ask_size': [1.0, 1.2, 1.8]
        })
        
        sim = HFTSimulator(ticks, latency_ms=20)
        assert sim is not None
        assert sim.latency == 0.02
    
    def test_get_tick(self):
        """Test get_tick method"""
        from src.hft.hft_simulator import HFTSimulator
        import pandas as pd
        
        ticks = pd.DataFrame({
            'bid': [50000, 50100],
            'ask': [50010, 50110],
            'bid_size': [1.0, 1.5],
            'ask_size': [1.0, 1.2]
        })
        
        sim = HFTSimulator(ticks)
        tick = sim.get_tick()
        assert tick is not None
    
    def test_execute_order(self):
        """Test order execution"""
        from src.hft.hft_simulator import HFTSimulator
        import pandas as pd
        
        ticks = pd.DataFrame({
            'bid': [50000, 50100],
            'ask': [50010, 50110],
            'bid_size': [10.0, 10.0],
            'ask_size': [10.0, 10.0]
        })
        
        sim = HFTSimulator(ticks)
        trade = sim.execute('BUY', 0.1)
        assert trade is not None
        assert 'price' in trade


class TestHFTRLEnv:
    """Test HFT RL Environment"""
    
    def test_initialization(self):
        """Test RL environment initialization"""
        from src.hft.hft_env import HFTRLEnv
        from src.hft.hft_simulator import HFTSimulator
        import pandas as pd
        
        ticks = pd.DataFrame({
            'bid': [50000] * 100,
            'ask': [50010] * 100,
            'bid_size': [1.0] * 100,
            'ask_size': [1.0] * 100
        })
        
        sim = HFTSimulator(ticks)
        env = HFTRLEnv(sim)
        assert env is not None
        assert env.state_dim == 8
    
    def test_reset(self):
        """Test environment reset"""
        from src.hft.hft_env import HFTRLEnv
        from src.hft.hft_simulator import HFTSimulator
        import pandas as pd
        
        ticks = pd.DataFrame({
            'bid': [50000] * 100,
            'ask': [50010] * 100,
            'bid_size': [1.0] * 100,
            'ask_size': [1.0] * 100
        })
        
        sim = HFTSimulator(ticks)
        env = HFTRLEnv(sim)
        state = env.reset()
        assert len(state) == 8


class TestAutoMLEngine:
    """Test AutoML Engine"""
    
    def test_genome_creation(self):
        """Test strategy genome creation"""
        from src.automl.automl_engine import StrategyGenome
        
        genome = StrategyGenome()
        assert genome is not None
        assert genome.use_rsi is True
        assert genome.rsi_period == 14
    
    def test_genome_mutation(self):
        """Test genome mutation"""
        from src.automl.automl_engine import StrategyGenome
        
        genome = StrategyGenome()
        mutated = genome.mutate(mutation_rate=1.0)
        assert mutated is not None
    
    def test_genome_crossover(self):
        """Test genome crossover"""
        from src.automl.automl_engine import StrategyGenome
        
        parent1 = StrategyGenome()
        parent2 = StrategyGenome()
        child = StrategyGenome.crossover(parent1, parent2)
        assert child is not None
    
    def test_automl_evolver(self):
        """Test AutoML evolver"""
        from src.automl.automl_engine import AutoMLEvolver
        
        evolver = AutoMLEvolver(
            population_size=10,
            generations=2
        )
        assert evolver is not None
        assert evolver.population_size == 10


class TestMultiAgentMarket:
    """Test Multi-Agent Market Simulator"""
    
    def test_market_agent(self):
        """Test market agent creation"""
        from src.simulations.multi_agent_market import MarketAgent
        
        agent = MarketAgent('test_agent', initial_capital=10000)
        assert agent is not None
        assert agent.agent_id == 'test_agent'
        assert agent.capital == 10000
    
    def test_market_maker(self):
        """Test market maker agent"""
        from src.simulations.multi_agent_market import MarketMaker
        
        mm = MarketMaker('mm_1', spread=0.001, size=1.0)
        assert mm is not None
        assert mm.spread == 0.001
    
    def test_taker(self):
        """Test taker agent"""
        from src.simulations.multi_agent_market import Taker
        
        taker = Taker('taker_1', frequency=0.1)
        assert taker is not None
        assert taker.frequency == 0.1
    
    def test_arbitrageur(self):
        """Test arbitrageur agent"""
        from src.simulations.multi_agent_market import Arbitrageur
        
        arb = Arbitrageur('arb_1', threshold=0.001)
        assert arb is not None
        assert arb.threshold == 0.001
    
    def test_multi_agent_market(self):
        """Test multi-agent market creation"""
        from src.simulations.multi_agent_market import MultiAgentMarket
        
        market = MultiAgentMarket(
            initial_price=50000,
            n_market_makers=1,
            n_takers=1,
            n_arbitrageurs=0,
            include_rl_agent=False
        )
        assert market is not None
        assert len(market.agents) == 2


class TestMetaEvolution:
    """Test Meta-Evolution Engine"""
    
    def test_hybrid_agent(self):
        """Test hybrid agent creation"""
        from src.meta.meta_evolution_engine import HybridAgent
        
        agent = HybridAgent()
        assert agent is not None
        agent.weight_rl = 0.3
        agent.weight_ml = 0.3
        agent.weight_gp = 0.4
    
    def test_meta_evolution_engine(self):
        """Test meta evolution engine"""
        from src.meta.meta_evolution_engine import MetaEvolutionEngine
        
        engine = MetaEvolutionEngine(
            population_size=10,
            elite_ratio=0.1
        )
        assert engine is not None
        assert engine.population_size == 10
    
    def test_gp_component(self):
        """Test GP component"""
        from src.meta.meta_evolution_engine import GPComponent
        
        gp = GPComponent(max_depth=3)
        assert gp is not None
        state = np.random.randn(8)
        decision = gp.evaluate(state)
        assert decision in [-1, 0, 1]


class TestMultiMarketEvolution:
    """Test Multi-Market Evolution"""
    
    def test_market_creation(self):
        """Test market creation"""
        from src.meta.multi_market_evolution import Market
        
        market = Market(
            name='crypto',
            base_volatility=0.02,
            base_liquidity=0.5,
            base_spread=0.001
        )
        assert market is not None
        assert market.name == 'crypto'
    
    def test_market_simulator(self):
        """Test market simulator"""
        from src.meta.multi_market_evolution import MarketSimulator
        
        sim = MarketSimulator(
            name='test',
            volatility=0.01,
            liquidity=1.0
        )
        assert sim is not None
    
    def test_migrating_agent(self):
        """Test migrating agent"""
        from src.meta.multi_market_evolution import MigratingAgent
        
        agent = MigratingAgent('agent_1', 'RL')
        assert agent is not None
        assert agent.agent_type == 'RL'
    
    def test_multi_market_evolution(self):
        """Test multi-market evolution engine"""
        from src.meta.multi_market_evolution import MultiMarketEvolution, Market
        
        markets = {
            'crypto': Market('crypto', 0.02, 0.5, 0.001),
            'forex': Market('forex', 0.001, 1.0, 0.0001)
        }
        
        engine = MultiMarketEvolution(markets, population_size=10)
        assert engine is not None


class TestEmergentCommunication:
    """Test Emergent Communication Engine"""
    
    def test_communication_agent(self):
        """Test communication agent"""
        from src.meta.emergent_communication import CommunicationAgent
        
        agent = CommunicationAgent(
            agent_id='test',
            agent_type='RL',
            policy=np.random.randn(8),
            msg_encoder=np.random.randn(8, 4),
            msg_decoder=np.random.randn(4, 8)
        )
        assert agent is not None
    
    def test_send_message(self):
        """Test message sending"""
        from src.meta.emergent_communication import CommunicationAgent
        
        agent = CommunicationAgent(
            agent_id='test',
            agent_type='RL',
            policy=np.ones(8),
            msg_encoder=np.eye(8, 4),
            msg_decoder=np.eye(4, 8)
        )
        
        state = np.random.randn(8)
        msg = agent.send_message(state)
        assert len(msg) == 4
    
    def test_communication_evolution(self):
        """Test communication evolution"""
        from src.meta.emergent_communication import CommunicationEvolution
        
        # Create mock environment
        class MockSimulator:
            def reset(self): pass
            def get_state(self): return np.random.randn(8)
            def step_multi(self, actions): return None, [0]*4, False
        
        class MockEnv:
            def __init__(self):
                self.simulator = MockSimulator()
                self.agents = []
            def run_episode(self): return [0.0]
            def get_communication_stats(self): return {}
        
        env = MockEnv()
        engine = CommunicationEvolution(env, population_size=10)
        assert engine is not None

