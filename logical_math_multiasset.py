"""
Logical Math Multi-Asset Module
================================
Modulo per la valutazione matematica dei segnali di trading multi-asset.
Combina analisi logica con calcoli di rischio portafoglio.
"""

from typing import List, Dict
import random
import logging

# --- Riutilizziamo le classi ---
from logical_portfolio_module import NewsItem, Portfolio, LogicalPortfolioEngine

logger = logging.getLogger(__name__)


# --- Modulo Matematico Multi-Asset ---
class MathDecisionEngineMultiAsset:
    """
    Engine matematico per la valutazione dei segnali di trading
    con gestione del rischio multi-asset.
    """
    
    def __init__(self, portfolio: Portfolio, max_risk_perc: float = 0.2):
        """
        Inizializza l'engine matematico.
        
        Args:
            portfolio: Portafoglio multi-asset
            max_risk_perc: Percentuale massima del portafoglio per singola operazione
        """
        self.portfolio = portfolio
        self.max_risk_perc = max_risk_perc

    def evaluate_signal(self, signal_info: Dict) -> Dict:
        """
        Valuta il segnale logico usando calcoli di portafoglio multi-asset.
        
        Args:
            signal_info: Dizionario con informazioni sul segnale
                - asset: Simbolo dell'asset
                - signal: "BUY", "SELL", "HOLD"
                - confidence: Livello di confidenza (0-1)
                
        Returns:
            Dizionario con il risultato della valutazione
        """
        asset = signal_info["asset"]
        signal = signal_info["signal"]
        confidence = signal_info["confidence"]

        # Simulazione rischio multi-asset: fattore tra 0.7 e 1.0
        risk_factor = random.uniform(0.7, 1.0)
        adjusted_confidence = confidence * risk_factor

        # Controllo limite massimo per portafoglio
        max_alloc = self.portfolio.total_value() * self.max_risk_perc
        can_execute = self.portfolio.can_execute(asset, adjusted_confidence, max_allocation=max_alloc)

        final_signal = signal if can_execute else "HOLD"

        return {
            "asset": asset,
            "final_signal": final_signal,
            "adjusted_confidence": round(adjusted_confidence, 2),
            "can_execute": can_execute
        }


# --- Engine Integrato Multi-Asset ---
class IntegratedDecisionSystemMultiAsset:
    """
    Sistema integrato che combina l'analisi logica delle news
    con la valutazione matematica dei segnali.
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Inizializza il sistema integrato.
        
        Args:
            portfolio: Portafoglio multi-asset
        """
        self.portfolio = portfolio
        self.logical_engine = LogicalPortfolioEngine(portfolio)
        self.math_engine = MathDecisionEngineMultiAsset(portfolio)

    def process_news_feed(self, news_feed: List[NewsItem]) -> List[Dict]:
        """
        Processa un feed di notizie e genera segnali finali.
        
        Args:
            news_feed: Lista di NewsItem da analizzare
            
        Returns:
            Lista di dizionari con segnali finali
        """
        # Step 1: Analisi logica delle news
        logical_signals = self.logical_engine.analyze_news(news_feed)
        
        # Step 2: Valutazione matematica di ogni segnale
        final_signals = []

        for sig in logical_signals:
            final = self.math_engine.evaluate_signal(sig)
            # Preserva informazioni aggiuntive dal segnale logico
            final["source"] = sig.get("source", "Unknown")
            final["title"] = sig.get("title", "")
            final["sentiment"] = sig.get("sentiment", 0)
            final_signals.append(final)

        return final_signals


# === Esempio di utilizzo ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Portafoglio multi-asset
    my_portfolio = Portfolio(balances={"BTC": 100000, "ETH": 50000, "SOL": 20000, "ADA": 15000})
    
    # Imposta prezzi per il calcolo del valore
    my_portfolio.set_price("BTC", 95000)
    my_portfolio.set_price("ETH", 3500)
    my_portfolio.set_price("SOL", 180)
    my_portfolio.set_price("ADA", 0.8)

    system = IntegratedDecisionSystemMultiAsset(portfolio=my_portfolio)

    news_feed = [
        NewsItem("Bitcoin Surges Past $95K on ETF Inflows", "CoinDesk"),
        NewsItem("Ethereum Upgrade Boosts Network Activity", "CoinTelegraph"),
        NewsItem("Solana DeFi TVL Reaches New High", "The Block"),
        NewsItem("Fed Signals Potential Rate Cut in March", "Reuters"),
        NewsItem("Major Bank Launches Crypto Custody Service", "Bloomberg"),
        NewsItem("Cardano Smart Contract Adoption Increases", "CoinTelegraph"),
    ]

    final_signals = system.process_news_feed(news_feed)
    
    print("\n" + "=" * 60)
    print("FINAL SIGNALS")
    print("=" * 60)
    for s in final_signals:
        print(f"  {s['asset']:6} | {s['final_signal']:4} | Conf: {s['adjusted_confidence']:.2f} | Exec: {s['can_execute']}")
        print(f"         | Title: {s['title'][:50]}...")
        print()
    
    my_portfolio.log_portfolio()
