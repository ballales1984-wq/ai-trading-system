"""
HFT Trading Engine
==================
Integra HFT Simulator con ML Strategies e Trading Engine.

Day 2 Checklist:
- [x] Loop tick-by-tick in hft_simulator.py
- [x] Agenti: market makers, arbitraggisti, retail
- [x] Interazione agenti + strategie ML
- [x] Output HFT nel TradingEngine
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from src.hft.hft_simulator import HFTSimulator, OrderbookSimulator, create_tick_data
from src.simulations.multi_agent_market import (
    MultiAgentMarket, MarketMaker, Taker, Arbitrageur, 
    RLAgentWrapper, Order, Trade, Orderbook
)

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Tipo di segnale di trading."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class HFTSignal:
    """Segnale HFT generato dal sistema."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1
    price: float
    quantity: float
    source: str  # 'ml', 'agent', 'hft'
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.name,
            'strength': self.strength,
            'price': self.price,
            'quantity': self.quantity,
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class HFTMetrics:
    """Metriche HFT."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    avg_slippage: float = 0.0
    max_position: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'total_fees': self.total_fees,
            'avg_slippage': self.avg_slippage,
            'max_position': self.max_position,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate
        }


class RetailAgent(Taker):
    """
    Retail Agent - Trader al dettaglio con comportamento emotivo.
    Estende Taker con pattern comportamentali tipici dei retail traders.
    """
    
    def __init__(
        self,
        agent_id: str,
        frequency: float = 0.05,
        size_range: tuple = (0.01, 0.1),
        direction_bias: float = 0.0,
        fear_factor: float = 0.3,  # Tendenza a vendere in perdita
        greed_factor: float = 0.3,  # Tendenza a comprare in rialzo
        stop_loss_pct: float = 0.05
    ):
        super().__init__(agent_id, frequency, size_range, direction_bias)
        self.fear_factor = fear_factor
        self.greed_factor = greed_factor
        self.stop_loss_pct = stop_loss_pct
        self.entry_price = None
        self.tick_count = 0
        
    def act(self, market_state: Dict) -> List[Order]:
        """Genera ordini con comportamento retail."""
        self.tick_count += 1
        
        current_price = market_state.get('mid_price', 0)
        
        # Check stop-loss se abbiamo una posizione
        if self.position != 0 and self.entry_price is not None and self.entry_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if self.position > 0 and pnl_pct < -self.stop_loss_pct:
                # Stop-loss triggered - sell
                return [Order(
                    self.agent_id, 'ask', current_price, 
                    abs(self.position), self.tick_count, 'market'
                )]
        
        # Comportamento emotivo
        if random.random() > self.frequency:
            return []
        
        # Greed: compra quando il prezzo sale
        price_change = market_state.get('price_change_1pct', 0)
        
        if price_change > 0.005 and random.random() < self.greed_factor:
            # FOMO buy
            side = 'bid'
            size = random.uniform(*self.size_range) * 1.5  # Overtrade
        elif price_change < -0.005 and random.random() < self.fear_factor:
            # Panic sell
            side = 'ask'
            size = abs(self.position) if self.position > 0 else random.uniform(*self.size_range)
        else:
            # Normal trading
            side = 'bid' if random.random() < 0.5 + self.direction_bias * 0.5 else 'ask'
            size = random.uniform(*self.size_range)
        
        order = Order(
            self.agent_id, side, current_price, size, 
            self.tick_count, 'market'
        )
        
        return [order]
    
    def on_trade(self, trade: Trade):
        """Track entry price."""
        super().on_trade(trade)
        
        if trade.buyer_id == self.agent_id:
            # Bought - update entry price (average)
            if self.entry_price is None:
                self.entry_price = trade.price
            else:
                # Weighted average
                total_qty = self.position
                self.entry_price = (self.entry_price * (total_qty - trade.quantity) + 
                                   trade.price * trade.quantity) / total_qty


class MLStrategyAdapter:
    """
    Adattatore per integrare strategie ML con il mercato HFT.
    """
    
    def __init__(
        self,
        ml_model=None,
        feature_window: int = 100,
        prediction_threshold: float = 0.6
    ):
        self.ml_model = ml_model
        self.feature_window = feature_window
        self.prediction_threshold = prediction_threshold
        self.price_history: List[float] = []
        self.feature_history: List[Dict] = []
        
    def update(self, market_state: Dict) -> np.ndarray:
        """Aggiorna feature history."""
        price = market_state.get('mid_price', 0)
        self.price_history.append(price)
        
        # Calcola features
        features = self._compute_features(market_state)
        self.feature_history.append(features)
        
        # Mantieni solo le ultime N
        if len(self.price_history) > self.feature_window:
            self.price_history = self.price_history[-self.feature_window:]
            self.feature_history = self.feature_history[-self.feature_window:]
        
        return self._get_feature_vector()
    
    def _compute_features(self, market_state: Dict) -> Dict:
        """Calcola features dal market state."""
        features = {
            'spread': market_state.get('spread', 0),
            'imbalance': market_state.get('imbalance', 0),
            'bid_size': market_state.get('bid_size', 0),
            'ask_size': market_state.get('ask_size', 0),
        }
        
        if len(self.price_history) > 1:
            features['return_1'] = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
        else:
            features['return_1'] = 0
            
        if len(self.price_history) > 5:
            features['return_5'] = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
        else:
            features['return_5'] = 0
            
        if len(self.price_history) > 20:
            prices = np.array(self.price_history[-20:])
            features['volatility_20'] = np.std(np.diff(prices) / prices[:-1])
        else:
            features['volatility_20'] = 0
            
        return features
    
    def _get_feature_vector(self) -> np.ndarray:
        """Converte feature history in vettore."""
        if not self.feature_history:
            return np.zeros(10)
        
        # Prendi ultime features
        latest = self.feature_history[-1]
        
        # Aggiungi statistiche storiche
        if len(self.price_history) >= 20:
            prices = np.array(self.price_history[-20:])
            sma = np.mean(prices)
            std = np.std(prices)
            current = prices[-1]
        else:
            sma = current = self.price_history[-1] if self.price_history else 0
            std = 0
        
        return np.array([
            latest.get('spread', 0),
            latest.get('imbalance', 0),
            latest.get('return_1', 0),
            latest.get('return_5', 0),
            latest.get('volatility_20', 0),
            latest.get('bid_size', 0),
            latest.get('ask_size', 0),
            (current - sma) / std if std > 0 else 0,  # Z-score
            current / sma - 1 if sma > 0 else 0,  # Price vs SMA
            len(self.price_history) / self.feature_window  # History completeness
        ], dtype=np.float32)
    
    def predict(self, market_state: Dict) -> Optional[HFTSignal]:
        """Genera segnale basato su predizione ML."""
        features = self.update(market_state)
        
        if self.ml_model is None:
            return None
        
        try:
            # Predizione
            if hasattr(self.ml_model, 'predict_proba'):
                proba = self.ml_model.predict_proba([features])[0]
                prediction = np.argmax(proba)
                confidence = proba[prediction]
            else:
                prediction = self.ml_model.predict([features])[0]
                confidence = 0.5
            
            # Soglia di confidenza
            if confidence < self.prediction_threshold:
                return None
            
            # Converti in segnale
            signal_type = SignalType.HOLD
            if prediction == 1:
                signal_type = SignalType.BUY
            elif prediction == -1 or prediction == 2:
                signal_type = SignalType.SELL
            
            if signal_type == SignalType.HOLD:
                return None
            
            return HFTSignal(
                timestamp=datetime.now(),
                symbol='HFT_ASSET',
                signal_type=signal_type,
                strength=confidence,
                price=market_state.get('mid_price', 0),
                quantity=0.01,  # Default size
                source='ml'
            )
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None


class HFTTradingEngine:
    """
    Engine HFT che integra:
    - Multi-Agent Market Simulator
    - ML Strategy Adapter
    - Tick-by-tick execution
    - Output per TradingEngine
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_price: float = 50000,
        n_market_makers: int = 2,
        n_takers: int = 3,
        n_arbitrageurs: int = 1,
        n_retail: int = 2,
        ml_model=None,
        enable_ml_signals: bool = True,
        tick_interval_ms: int = 100
    ):
        self.symbol = symbol
        self.initial_price = initial_price
        self.tick_interval_ms = tick_interval_ms
        self.enable_ml_signals = enable_ml_signals
        
        # Crea mercato multi-agente
        self.market = MultiAgentMarket(
            initial_price=initial_price,
            n_market_makers=n_market_makers,
            n_takers=n_takers,
            n_arbitrageurs=n_arbitrageurs,
            include_rl_agent=False
        )
        
        # Aggiungi retail agents
        for i in range(n_retail):
            retail = RetailAgent(
                f'retail_{i}',
                frequency=0.03,
                size_range=(0.01, 0.05),
                fear_factor=0.4,
                greed_factor=0.5
            )
            self.market.agents[retail.agent_id] = retail
        
        # ML Strategy Adapter
        self.ml_adapter = MLStrategyAdapter(ml_model=ml_model)
        
        # Stato
        self.is_running = False
        self.signals: List[HFTSignal] = []
        self.metrics = HFTMetrics()
        
        # Callbacks
        self._on_signal: Optional[Callable[[HFTSignal], None]] = None
        self._on_tick: Optional[Callable[[Dict], None]] = None
        
        # Thread
        self._tick_thread: Optional[threading.Thread] = None
        
        logger.info(f"📊 HFT Trading Engine initialized for {symbol}")
        logger.info(f"   Agents: {len(self.market.agents)}")
        logger.info(f"   ML signals: {enable_ml_signals}")
    
    def set_callbacks(
        self,
        on_signal: Optional[Callable[[HFTSignal], None]] = None,
        on_tick: Optional[Callable[[Dict], None]] = None
    ):
        """Imposta callback per eventi."""
        self._on_signal = on_signal
        self._on_tick = on_tick
    
    def start(self):
        """Avvia il loop HFT."""
        if self.is_running:
            logger.warning("HFT Engine already running")
            return
        
        logger.info("🚀 Starting HFT Trading Engine...")
        self.is_running = True
        
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()
        
        logger.info("✅ HFT Trading Engine started")
    
    def stop(self):
        """Ferma il loop HFT."""
        logger.info("🛑 Stopping HFT Trading Engine...")
        self.is_running = False
        
        if self._tick_thread:
            self._tick_thread.join(timeout=5)
        
        logger.info("✅ HFT Trading Engine stopped")
    
    def _tick_loop(self):
        """Loop tick-by-tick principale."""
        logger.info("Tick loop started")
        
        while self.is_running:
            try:
                # Esegui uno step del mercato
                market_state = self.market.step()
                
                # Callback tick
                if self._on_tick:
                    try:
                        self._on_tick(market_state)
                    except Exception as e:
                        logger.error(f"Error in tick callback: {e}")
                
                # Genera segnale ML
                if self.enable_ml_signals:
                    signal = self.ml_adapter.predict(market_state)
                    if signal:
                        signal.symbol = self.symbol
                        self.signals.append(signal)
                        
                        # Callback signal
                        if self._on_signal:
                            try:
                                self._on_signal(signal)
                            except Exception as e:
                                logger.error(f"Error in signal callback: {e}")
                        
                        logger.info(
                            f"📈 HFT Signal: {signal.signal_type.name} "
                            f"@ {signal.price:.2f} (strength: {signal.strength:.2f})"
                        )
                
                # Aggiorna metriche
                self._update_metrics()
                
                # Sleep per tick interval
                time.sleep(self.tick_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in tick loop: {e}")
                time.sleep(1)
        
        logger.info("Tick loop stopped")
    
    def _update_metrics(self):
        """Aggiorna metriche HFT."""
        rl_stats = self.market.get_rl_agent_stats()
        
        if rl_stats:
            self.metrics.total_trades = rl_stats.get('num_trades', 0)
            self.metrics.total_pnl = rl_stats.get('total_pnl', 0)
        
        # Calcola win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
    
    def get_current_state(self) -> Dict:
        """Ottieni stato corrente del mercato."""
        state = self.market.orderbook.get_state()
        state['symbol'] = self.symbol
        state['tick_count'] = self.market.tick_count
        state['price_history'] = self.market.price_history[-100:]  # Ultimi 100
        return state
    
    def get_signals(self, limit: int = 100) -> List[HFTSignal]:
        """Ottieni segnali generati."""
        return self.signals[-limit:]
    
    def get_metrics(self) -> HFTMetrics:
        """Ottieni metriche HFT."""
        return self.metrics
    
    def get_agent_stats(self) -> Dict[str, Dict]:
        """Ottieni statistiche per ogni agente."""
        stats = {}
        for agent_id, agent in self.market.agents.items():
            stats[agent_id] = {
                'position': agent.position,
                'capital': agent.capital,
                'pnl': agent.get_pnl(self.market.current_price),
                'trades': len(agent.trades)
            }
        return stats
    
    def run_backtest(
        self,
        n_ticks: int = 10000,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Esegui backtest HFT.
        
        Args:
            n_ticks: Numero di tick da simulare
            verbose: Stampa progresso
            
        Returns:
            DataFrame con risultati
        """
        logger.info(f"🔄 Running HFT backtest for {n_ticks} ticks...")
        
        results = []
        
        for i in range(n_ticks):
            state = self.market.step()
            
            # ML signal
            if self.enable_ml_signals:
                signal = self.ml_adapter.predict(state)
                if signal:
                    signal.symbol = self.symbol
                    self.signals.append(signal)
            
            # Record
            results.append({
                'tick': i,
                'price': state['mid_price'],
                'spread': state['spread'],
                'imbalance': state['imbalance'],
                'signals_count': len(self.signals)
            })
            
            if verbose and i % 1000 == 0:
                logger.info(f"Tick {i}/{n_ticks}: Price={state['mid_price']:.2f}")
        
        self._update_metrics()
        
        logger.info(f"✅ Backtest complete: {self.metrics.total_trades} trades, "
                   f"PnL={self.metrics.total_pnl:.2f}")
        
        return pd.DataFrame(results)
    
    def inject_signal(self, signal: HFTSignal):
        """Inietta un segnale esterno."""
        signal.symbol = self.symbol
        self.signals.append(signal)
        
        if self._on_signal:
            self._on_signal(signal)


# Import random per RetailAgent
import random


def create_hft_engine(
    symbol: str = "BTCUSDT",
    initial_price: float = 50000,
    ml_model=None
) -> HFTTradingEngine:
    """
    Factory function per creare HFT Engine.
    
    Args:
        symbol: Simbolo da tradare
        initial_price: Prezzo iniziale
        ml_model: Modello ML per predizioni
        
    Returns:
        HFTTradingEngine configurato
    """
    return HFTTradingEngine(
        symbol=symbol,
        initial_price=initial_price,
        ml_model=ml_model,
        enable_ml_signals=True,
        tick_interval_ms=100
    )
