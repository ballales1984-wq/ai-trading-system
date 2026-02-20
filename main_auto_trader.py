"""
Main Auto Trader
================
Flusso end-to-end completamente automatico per AI Trading.

Combina:
- Raccolta dati da mercati e sentiment
- Analisi tecnica e indicatori
- Decision Engine con Monte Carlo
- Esecuzione automatica ordini
- Feedback loop e aggiornamento portafoglio

Usage:
    python main_auto_trader.py --mode live --dry-run
    python main_auto_trader.py --mode backtest
    python main_auto_trader.py --mode dashboard
"""

import time
import logging
import argparse
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import threading
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_trader.log')
    ]
)
logger = logging.getLogger(__name__)

# Import moduli interni
try:
    from src.decision.decision_automatic import DecisionEngine, MonteCarloSimulator
    from src.execution.auto_executor import (
        AutoExecutor, SafetyConfig, SimulatedExchangeClient, OrderStatus
    )
    from src.decision.filtro_opportunita import OpportunityFilter
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    MODULES_AVAILABLE = False

# Import moduli esistenti
try:
    from data_collector import DataCollector
    DATA_COLLECTOR_AVAILABLE = True
except ImportError:
    DATA_COLLECTOR_AVAILABLE = False
    logger.warning("DataCollector not available")

try:
    from technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    logger.warning("TechnicalAnalyzer not available")

try:
    from sentiment_news import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logger.warning("SentimentAnalyzer not available")


# ==================== CONFIGURAZIONE ====================

@dataclass
class TradingConfig:
    """Configurazione del trading bot."""
    # Parametri generali
    loop_interval: int = 60  # secondi tra cicli
    assets: List[str] = None  # Lista asset da monitorare
    
    # Parametri portafoglio
    initial_balance: float = 100000.0  # USDT
    max_risk_per_trade: float = 0.02  # 2%
    
    # Parametri decisionali
    threshold_confidence: float = 0.6
    semantic_weight: float = 0.5
    numeric_weight: float = 0.5
    monte_carlo_sims: int = 1000
    
    # Parametri sicurezza esecuzione
    min_order_value: float = 10.0
    max_order_value: float = 10000.0
    max_orders_per_minute: int = 10
    dry_run: bool = True  # Se True, simula senza inviare ordini reali
    
    # Exchange
    exchange: str = "binance"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    
    def __post_init__(self):
        if self.assets is None:
            self.assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


# ==================== DATA AGGREGATOR ====================

class DataAggregator:
    """
    Aggrega dati da multiple fonti per il DecisionEngine.
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_collector = None
        self.technical_analyzer = None
        self.sentiment_analyzer = None
        
        # Inizializza moduli se disponibili
        if DATA_COLLECTOR_AVAILABLE:
            self.data_collector = DataCollector(simulation=config.dry_run)
        
        if TECHNICAL_ANALYSIS_AVAILABLE:
            self.technical_analyzer = TechnicalAnalyzer()
        
        if SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = SentimentAnalyzer()
    
    def fetch_market_data(self, assets: List[str]) -> Dict[str, Dict]:
        """
        Recupera dati di mercato per gli asset.
        
        Args:
            assets: Lista di simboli
            
        Returns:
            Dizionario {symbol: market_data}
        """
        market_data = {}
        
        for symbol in assets:
            try:
                if self.data_collector:
                    # Usa DataCollector reale
                    data = self.data_collector.get_market_data(symbol)
                    market_data[symbol] = data
                else:
                    # Simula dati
                    market_data[symbol] = self._simulate_market_data(symbol)
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                market_data[symbol] = self._simulate_market_data(symbol)
        
        return market_data
    
    def fetch_sentiment(self, assets: List[str]) -> Dict[str, Dict]:
        """
        Recupera dati sentiment per gli asset.
        
        Args:
            assets: Lista di simboli
            
        Returns:
            Dizionario {symbol: sentiment_data}
        """
        sentiment_data = {}
        
        for symbol in assets:
            try:
                if self.sentiment_analyzer:
                    # Usa SentimentAnalyzer reale
                    data = self.sentiment_analyzer.analyze(symbol)
                    sentiment_data[symbol] = data
                else:
                    # Simula sentiment
                    sentiment_data[symbol] = self._simulate_sentiment(symbol)
            except Exception as e:
                logger.error(f"Error fetching sentiment for {symbol}: {e}")
                sentiment_data[symbol] = self._simulate_sentiment(symbol)
        
        return sentiment_data
    
    def calculate_indicators(self, market_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calcola indicatori tecnici per gli asset.
        
        Args:
            market_data: Dati di mercato
            
        Returns:
            Dizionario {symbol: indicators}
        """
        indicators = {}
        
        for symbol, data in market_data.items():
            try:
                if self.technical_analyzer and 'candles' in data:
                    # Usa TechnicalAnalyzer reale
                    analysis = self.technical_analyzer.analyze(data['candles'])
                    indicators[symbol] = analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
                else:
                    # Simula indicatori
                    indicators[symbol] = self._simulate_indicators(symbol)
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
                indicators[symbol] = self._simulate_indicators(symbol)
        
        return indicators
    
    def _simulate_market_data(self, symbol: str) -> Dict:
        """Simula dati di mercato."""
        import random
        base_prices = {
            "BTCUSDT": 95000,
            "ETHUSDT": 3500,
            "SOLUSDT": 180,
            "XRPUSDT": 2.5,
            "BNBUSDT": 650,
            "ADAUSDT": 0.8
        }
        base_price = base_prices.get(symbol, 100)
        
        return {
            "symbol": symbol,
            "current_price": base_price * (1 + random.uniform(-0.02, 0.02)),
            "price_change_24h": random.uniform(-5, 5),
            "volume_24h": random.uniform(1e6, 1e9),
            "volatility_annual": random.uniform(0.3, 0.8),
            "expected_return": random.uniform(-0.1, 0.2)
        }
    
    def _simulate_sentiment(self, symbol: str) -> Dict:
        """Simula dati sentiment."""
        import random
        return {
            "symbol": symbol,
            "sentiment_score": random.uniform(-0.5, 0.5),
            "event_impact": random.uniform(-0.3, 0.3),
            "news_score": random.uniform(-0.3, 0.3),
            "trend_signal": random.uniform(-0.3, 0.3)
        }
    
    def _simulate_indicators(self, symbol: str) -> Dict:
        """Simula indicatori tecnici."""
        import random
        rsi = random.uniform(30, 70)
        macd_signal = random.uniform(-0.5, 0.5)
        
        return {
            "symbol": symbol,
            "rsi": rsi,
            "rsi_score": (rsi - 50) / 50,  # Normalizzato -1 a 1
            "macd_score": macd_signal,
            "momentum_score": random.uniform(-0.5, 0.5),
            "volume_score": random.uniform(-0.3, 0.3),
            "volatility_score": random.uniform(0, 0.5),
            "trend_signal": random.uniform(-0.5, 0.5)
        }


# ==================== AUTO TRADER ====================

class AutoTrader:
    """
    Trading bot completamente automatico.
    
    Flusso:
    1. Raccolta dati (mercato + sentiment)
    2. Analisi tecnica
    3. Generazione ordini (DecisionEngine)
    4. Esecuzione ordini (AutoExecutor)
    5. Aggiornamento portafoglio
    6. Feedback loop
    """
    
    def __init__(self, config: TradingConfig):
        """
        Inizializza l'AutoTrader.
        
        Args:
            config: Configurazione del trading bot
        """
        self.config = config
        self.running = False
        self.cycle_count = 0
        
        # Componenti
        self.data_aggregator = DataAggregator(config)
        
        # Decision Engine
        self.decision_engine = DecisionEngine(
            portfolio_balance=config.initial_balance,
            threshold_confidence=config.threshold_confidence,
            max_risk_per_trade=config.max_risk_per_trade,
            semantic_weight=config.semantic_weight,
            numeric_weight=config.numeric_weight,
            monte_carlo_sims=config.monte_carlo_sims
        )
        
        # Safety Config per Executor
        safety_config = SafetyConfig(
            min_order_value=config.min_order_value,
            max_order_value=config.max_order_value,
            max_orders_per_minute=config.max_orders_per_minute,
            dry_run=config.dry_run
        )
        
        # Auto Executor
        self.executor = AutoExecutor(safety_config=safety_config)
        
        # Statistiche
        self.stats = {
            "cycles_completed": 0,
            "orders_generated": 0,
            "orders_executed": 0,
            "total_volume": 0.0,
            "start_time": None,
            "last_cycle_time": None
        }
        
        # Storico
        self.history: List[Dict] = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Gestisce segnali di interruzione."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def prepare_asset_analysis(
        self,
        market_data: Dict[str, Dict],
        sentiment_data: Dict[str, Dict],
        indicators: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Prepara i dati degli asset per il DecisionEngine.
        
        Args:
            market_data: Dati di mercato
            sentiment_data: Dati sentiment
            indicators: Indicatori tecnici
            
        Returns:
            Lista di asset formattati per DecisionEngine
        """
        assets = []
        
        for symbol in self.config.assets:
            market = market_data.get(symbol, {})
            sentiment = sentiment_data.get(symbol, {})
            tech = indicators.get(symbol, {})
            
            asset = {
                "name": symbol,
                # Dati semantici
                "sentiment_score": sentiment.get("sentiment_score", 0),
                "event_impact": sentiment.get("event_impact", 0),
                "trend_signal": sentiment.get("trend_signal", 0),
                "news_score": sentiment.get("news_score", 0),
                # Dati numerici
                "rsi_score": tech.get("rsi_score", 0),
                "macd_score": tech.get("macd_score", 0),
                "momentum_score": tech.get("momentum_score", 0),
                "volume_score": tech.get("volume_score", 0),
                "volatility_score": tech.get("volatility_score", 0),
                # Dati di mercato
                "price": market.get("current_price", 100),
                "volatility_annual": market.get("volatility_annual", 0.5),
                "expected_return": market.get("expected_return", 0)
            }
            assets.append(asset)
        
        return assets
    
    def run_cycle(self) -> Dict:
        """
        Esegue un singolo ciclo di trading.
        
        Returns:
            Dizionario con risultati del ciclo
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info("=" * 60)
        logger.info(f"CYCLE {self.cycle_count} STARTED")
        logger.info("=" * 60)
        
        # 1️⃣ Raccolta dati
        logger.info("1. Fetching market data...")
        market_data = self.data_aggregator.fetch_market_data(self.config.assets)
        
        logger.info("2. Fetching sentiment data...")
        sentiment_data = self.data_aggregator.fetch_sentiment(self.config.assets)
        
        # 2️⃣ Analisi tecnica
        logger.info("3. Calculating technical indicators...")
        indicators = self.data_aggregator.calculate_indicators(market_data)
        
        # 3️⃣ Prepara input per Decision Engine
        logger.info("4. Preparing asset analysis...")
        asset_analysis = self.prepare_asset_analysis(market_data, sentiment_data, indicators)
        
        # 4️⃣ Generazione ordini
        logger.info("5. Generating orders with DecisionEngine...")
        orders = self.decision_engine.generate_orders(asset_analysis)
        self.stats["orders_generated"] += len(orders)
        
        logger.info(f"   Generated {len(orders)} orders")
        
        # 5️⃣ Esecuzione automatica
        logger.info("6. Executing orders...")
        executed_orders = self.executor.execute_orders(orders)
        
        executed_count = sum(1 for o in executed_orders if o.status == OrderStatus.EXECUTED)
        self.stats["orders_executed"] += executed_count
        self.stats["total_volume"] += sum(o.amount for o in executed_orders if o.status == OrderStatus.EXECUTED)
        
        logger.info(f"   Executed {executed_count}/{len(orders)} orders")
        
        # 6️⃣ Aggiorna statistiche
        cycle_end = datetime.now()
        cycle_duration = (cycle_end - cycle_start).total_seconds()
        
        self.stats["cycles_completed"] += 1
        self.stats["last_cycle_time"] = cycle_end.isoformat()
        
        # Riepilogo portafoglio
        portfolio = self.decision_engine.get_portfolio_summary()
        
        result = {
            "cycle": self.cycle_count,
            "timestamp": cycle_start.isoformat(),
            "duration_seconds": cycle_duration,
            "orders_generated": len(orders),
            "orders_executed": executed_count,
            "portfolio": portfolio,
            "executed_orders": [
                {
                    "asset": o.asset,
                    "action": o.action,
                    "amount": o.amount,
                    "status": o.status.value
                }
                for o in executed_orders
            ]
        }
        
        # Salva in storico
        self.history.append(result)
        
        logger.info(f"Cycle completed in {cycle_duration:.2f}s")
        logger.info(f"Portfolio value: {portfolio['total_value']:,.2f} USDT")
        
        return result
    
    def run_continuous(self):
        """
        Esegue il loop continuo di trading.
        """
        self.running = True
        self.stats["start_time"] = datetime.now().isoformat()
        
        logger.info("=" * 60)
        logger.info("AUTO TRADER STARTED")
        logger.info(f"Mode: {'DRY RUN (simulation)' if self.config.dry_run else 'LIVE'}")
        logger.info(f"Assets: {self.config.assets}")
        logger.info(f"Loop interval: {self.config.loop_interval}s")
        logger.info("=" * 60)
        
        while self.running:
            try:
                # Esegui ciclo
                self.run_cycle()
                
                # Attendi prossimo ciclo
                logger.info(f"\nWaiting {self.config.loop_interval}s until next cycle...")
                time.sleep(self.config.loop_interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()
                break
            except Exception as e:
                logger.exception(f"Error in trading cycle: {e}")
                logger.info(f"Retrying in {self.config.loop_interval}s...")
                time.sleep(self.config.loop_interval)
    
    def stop(self):
        """Ferma il trading bot."""
        logger.info("Stopping AutoTrader...")
        self.running = False
        
        # Stampa statistiche finali
        self.print_summary()
    
    def print_summary(self):
        """Stampa un riepilogo finale."""
        print("\n" + "=" * 60)
        print("AUTO TRADER SUMMARY")
        print("=" * 60)
        print(f"  Cycles completed: {self.stats['cycles_completed']}")
        print(f"  Orders generated: {self.stats['orders_generated']}")
        print(f"  Orders executed: {self.stats['orders_executed']}")
        print(f"  Total volume: {self.stats['total_volume']:,.2f} USDT")
        
        if self.stats["start_time"]:
            start = datetime.fromisoformat(self.stats["start_time"])
            duration = datetime.now() - start
            print(f"  Running time: {duration}")
        
        # Portfolio finale
        portfolio = self.decision_engine.get_portfolio_summary()
        print(f"\n  Final portfolio value: {portfolio['total_value']:,.2f} USDT")
        print(f"  Cash: {portfolio['cash']:,.2f} USDT")
        print(f"  Positions: {portfolio['n_positions']}")
        print("=" * 60)


# ==================== MAIN ====================

def main():
    """Entry point principale."""
    parser = argparse.ArgumentParser(description="AI Trading Bot - Automatic Execution")
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "single"],
        default="live",
        help="Trading mode"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Simulate without real orders"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Loop interval in seconds"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=100000.0,
        help="Initial portfolio balance"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Assets to trade"
    )
    
    args = parser.parse_args()
    
    # Crea configurazione
    config = TradingConfig(
        loop_interval=args.interval,
        assets=args.assets,
        initial_balance=args.balance,
        dry_run=args.dry_run
    )
    
    # Crea AutoTrader
    trader = AutoTrader(config)
    
    if args.mode == "single":
        # Singolo ciclo
        result = trader.run_cycle()
        print(json.dumps(result, indent=2, default=str))
    else:
        # Loop continuo
        trader.run_continuous()


if __name__ == "__main__":
    main()