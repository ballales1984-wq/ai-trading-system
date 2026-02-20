"""
Execute Signals Multi-Asset
============================
Modulo per l'esecuzione dei segnali di trading multi-asset.
Integra l'analisi logico-matematica con l'esecuzione su multipli exchange.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Callable

from logical_math_multiasset import IntegratedDecisionSystemMultiAsset
from logical_portfolio_module import Portfolio, NewsItem
from src.execution import BinanceRouter, BybitRouter, OKXRouter

# Configurazione logging
logging.basicConfig(
    filename="execution_log.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MultiAssetExecutionEngine:
    """
    Engine per l'esecuzione di segnali multi-asset su multipli exchange.
    """
    
    def __init__(self, portfolio: Portfolio, use_testnet: bool = True):
        """
        Inizializza l'engine di esecuzione.
        
        Args:
            portfolio: Portafoglio multi-asset
            use_testnet: Usa testnet per i test
        """
        self.portfolio = portfolio
        self.system = IntegratedDecisionSystemMultiAsset(portfolio)
        self.use_testnet = use_testnet

        # Router per ogni exchange
        self.routers = {
            "BINANCE": BinanceRouter(portfolio, testnet=use_testnet),
            "BYBIT": BybitRouter(portfolio, testnet=use_testnet),
            "OKX": OKXRouter(portfolio, testnet=use_testnet)
        }
        
        # Statistiche esecuzione
        self._execution_stats = {
            "total_signals": 0,
            "executed_orders": 0,
            "failed_orders": 0,
            "total_volume": 0.0
        }

    def run_news_cycle(self, news_feed: List[Dict]) -> List[Dict]:
        """
        Esegue un ciclo completo di analisi + decisione + esecuzione.
        
        Args:
            news_feed: Lista di dizionari contenenti titoli, asset collegati e sentiment
            
        Returns:
            Lista dei risultati dell'esecuzione
        """
        results = []
        
        # Step 1: Converti news_feed in NewsItem se necessario
        news_items = []
        for item in news_feed:
            if isinstance(item, NewsItem):
                news_items.append(item)
            elif isinstance(item, dict):
                news_items.append(NewsItem(
                    title=item.get("title", ""),
                    source=item.get("source", "Unknown"),
                    asset=item.get("asset"),
                    sentiment=item.get("sentiment")
                ))
        
        # Step 2: Generare segnali dal modulo logico-matematico
        signals = self.system.process_news_feed(news_items)
        self._execution_stats["total_signals"] += len(signals)

        # Step 3: Filtrare segnali eseguibili e inviare ordini
        for sig in signals:
            asset = sig.get("asset")
            action = sig.get("final_signal")
            confidence = sig.get("adjusted_confidence", 0)

            result = {
                "asset": asset,
                "signal": action,
                "confidence": confidence,
                "executed": False,
                "order": None,
                "error": None
            }

            if sig.get("can_execute") and action in ["BUY", "SELL"]:
                # Calcolo quantità proporzionale al portafoglio e confidenza
                qty = self.portfolio.total_value() * 0.05 * confidence
                qty = max(qty, 0)  # prevenire quantità negativa

                # Scegli exchange dal segnale o default Binance
                exchange = sig.get("preferred_exchange", "BINANCE").upper()
                router = self.routers.get(exchange)

                if router:
                    try:
                        order = router.place_order(asset=asset, side=action, quantity=qty)
                        if order:
                            result["executed"] = True
                            result["order"] = order
                            self._execution_stats["executed_orders"] += 1
                            self._execution_stats["total_volume"] += qty
                            logging.info(f"Order executed: {action} {qty:.2f} USDT of {asset} on {exchange}")
                        else:
                            result["error"] = "Order returned None"
                            self._execution_stats["failed_orders"] += 1
                    except Exception as e:
                        result["error"] = str(e)
                        self._execution_stats["failed_orders"] += 1
                        logging.error(f"Order failed for {asset} on {exchange}: {e}")
                else:
                    result["error"] = f"No router configured for exchange: {exchange}"
                    logging.warning(f"No router configured for exchange: {exchange}")
            else:
                result["error"] = f"Signal not executable: can_execute={sig.get('can_execute')}, action={action}"

            results.append(result)

        # Step 4: Aggiornamento portafoglio e log
        self.portfolio.log_portfolio()
        
        return results

    def run_live_loop(self, news_feed_generator: Callable[[], List[Dict]], interval_sec: int = 60):
        """
        Esegue continuamente il ciclo di trading ogni 'interval_sec' secondi.
        
        Args:
            news_feed_generator: Funzione che restituisce lista di news/sentiment
            interval_sec: Intervallo in secondi tra ogni ciclo
        """
        import time
        
        logger.info(f"Starting live trading loop with {interval_sec}s interval")
        
        while True:
            try:
                logger.info("=" * 50)
                logger.info(f"Running news cycle at {datetime.now()}")
                logger.info("=" * 50)
                
                news_feed = news_feed_generator()
                results = self.run_news_cycle(news_feed)
                
                # Log risultati
                executed = sum(1 for r in results if r.get("executed"))
                logger.info(f"Cycle completed: {executed}/{len(results)} orders executed")
                
            except Exception as e:
                logging.error(f"Error in live loop: {e}")
            
            time.sleep(interval_sec)

    def get_execution_stats(self) -> Dict:
        """
        Ritorna le statistiche di esecuzione.
        
        Returns:
            Dizionario con statistiche
        """
        return self._execution_stats.copy()
    
    def reset_stats(self):
        """Resetta le statistiche di esecuzione."""
        self._execution_stats = {
            "total_signals": 0,
            "executed_orders": 0,
            "failed_orders": 0,
            "total_volume": 0.0
        }


class SignalExecutor:
    """
    Classe helper per eseguire segnali singoli o batch.
    """
    
    def __init__(self, engine: MultiAssetExecutionEngine):
        """
        Inizializza l'esecutore.
        
        Args:
            engine: Engine di esecuzione
        """
        self.engine = engine
    
    def execute_single(self, signal: Dict) -> Dict:
        """
        Esegue un singolo segnale.
        
        Args:
            signal: Dizionario con asset, signal, confidence, etc.
            
        Returns:
            Risultato dell'esecuzione
        """
        results = self.engine.run_news_cycle([signal])
        return results[0] if results else None
    
    def execute_batch(self, signals: List[Dict]) -> List[Dict]:
        """
        Esegue un batch di segnali.
        
        Args:
            signals: Lista di segnali
            
        Returns:
            Lista di risultati
        """
        return self.engine.run_news_cycle(signals)


# === Esempio di utilizzo ===
if __name__ == "__main__":
    # Esempio di inizializzazione
    portfolio = Portfolio(balances={"USDT": 100000, "BTC": 0.5, "ETH": 5.0})
    portfolio.set_price("BTC", 95000)
    portfolio.set_price("ETH", 3500)
    
    engine = MultiAssetExecutionEngine(portfolio, use_testnet=True)

    # Esempio: ciclo con feed statico
    news_feed_example = [
        {"asset": "BTC", "title": "Bitcoin surges past $95K", "source": "CoinDesk", "sentiment": 0.9},
        {"asset": "ETH", "title": "Ethereum upgrade boosts network activity", "source": "CoinTelegraph", "sentiment": 0.7},
        {"asset": "SOL", "title": "Solana DeFi TVL reaches new high", "source": "The Block", "sentiment": 0.5}
    ]
    
    print("\n" + "=" * 60)
    print("MULTI-ASSET EXECUTION ENGINE TEST")
    print("=" * 60)
    print(f"Portfolio Value: {portfolio.total_value():.2f} USDT")
    print("=" * 60)
    
    results = engine.run_news_cycle(news_feed_example)
    
    print("\nRESULTS:")
    print("-" * 60)
    for r in results:
        status = "[OK] EXECUTED" if r.get("executed") else "[--] SKIPPED"
        print(f"  {r['asset']:6} | {r['signal']:4} | {status}")
        if r.get("error"):
            print(f"         | Error: {r['error']}")
    
    print("\n" + "=" * 60)
    print("EXECUTION STATS:")
    print("-" * 60)
    stats = engine.get_execution_stats()
    print(f"  Total Signals:  {stats['total_signals']}")
    print(f"  Executed:       {stats['executed_orders']}")
    print(f"  Failed:         {stats['failed_orders']}")
    print(f"  Total Volume:   {stats['total_volume']:.2f} USDT")
    print("=" * 60)
