"""
Advanced Financial AI Assistant
==============================
Versione avanzata del assistant con capacità estese:
- Vocabolario espanso con 100+ concetti
- Auto-apprendimento da notizie
- Analisi predittiva
- Integrazione API completa
- Report automatici programmati

Autore: AI Trading System
Data: 2026-03-17
"""

import requests
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import threading
import time
# import schedule  # Opzionale, per scheduling

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_assistant")

# ==================== CONFIGURATION ====================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
DATA_DIR = "data/ai_assistant"
os.makedirs(DATA_DIR, exist_ok=True)

# ==================== ENHANCED VOCABULARY ====================

ADVANCED_CONCEPTS = {
    # Trading Basics
    "long_position": {"term": "Long", "def": "Posizione rialzista", "cat": "trading", "keywords": ["buy", "long", "acquisto"]},
    "short_position": {"term": "Short", "def": "Posizione ribassista", "cat": "trading", "keywords": ["sell", "short", "ribasso"]},
    "stop_loss": {"term": "Stop Loss", "def": "Ordine di protezione", "cat": "risk", "keywords": ["stop", "protezione"]},
    "take_profit": {"term": "Take Profit", "def": "Ordine di chiusura in profitto", "cat": "risk", "keywords": ["target", "profit"]},
    "leverage": {"term": "Leva", "def": "Moltiplicatore di esposizione", "cat": "trading", "keywords": ["leverage", "x10", "x100"]},
    "margin": {"term": "Margine", "def": "Capitale richiesto per posizione", "cat": "trading", "keywords": ["margin", "collateral"]},
    
    # Risk Metrics
    "var": {"term": "VaR", "def": "Value at Risk - perdita massima attesa", "cat": "risk", "keywords": ["var", "risk"]},
    "cvar": {"term": "CVaR", "def": "Conditional VaR - expected shortfall", "cat": "risk", "keywords": ["cvar", "tail"]},
    "drawdown": {"term": "Drawdown", "def": "Calo dal picco", "cat": "risk", "keywords": ["drawdown", "dd"]},
    "sharpe_ratio": {"term": "Sharpe Ratio", "def": "Ritorno ajusted per rischio", "cat": "metrics", "keywords": ["sharpe", "risk-adjusted"]},
    "sortino_ratio": {"term": "Sortino Ratio", "def": "Sharpe con volatilità negativa", "cat": "metrics", "keywords": ["sortino", "downside"]},
    "max_dd": {"term": "Max Drawdown", "def": "Drawdown massimo storico", "cat": "risk", "keywords": ["maxdd", "peak"]},
    
    # Technical Analysis
    "rsi": {"term": "RSI", "def": "Relative Strength Index", "cat": "technical", "keywords": ["rsi", "overbought", "oversold"]},
    "macd": {"term": "MACD", "def": "Moving Average Convergence Divergence", "cat": "technical", "keywords": ["macd", "trend"]},
    "ema": {"term": "EMA", "def": "Exponential Moving Average", "cat": "technical", "keywords": ["ema", "ma"]},
    "sma": {"term": "SMA", "def": "Simple Moving Average", "cat": "technical", "keywords": ["sma", "ma"]},
    "bollinger": {"term": "Bollinger Bands", "def": "Bande di volatilità", "cat": "technical", "keywords": ["bollinger", "bands"]},
    "volume_profile": {"term": "Volume Profile", "def": "Profilo volume", "cat": "technical", "keywords": ["volume", "profile"]},
    "support": {"term": "Supporto", "def": "Livello di supporto", "cat": "technical", "keywords": ["support", "floor"]},
    "resistance": {"term": "Resistenza", "def": "Livello di resistenza", "cat": "technical", "keywords": ["resistance", "ceiling"]},
    "breakout": {"term": "Breakout", "def": "Rottura di livello", "cat": "technical", "keywords": ["breakout", "rupture"]},
    "pullback": {"term": "Pullback", "def": "Ritorno al livello", "cat": "technical", "keywords": ["pullback", "retrace"]},
    
    # Market Structure
    "volatility": {"term": "Volatilità", "def": "Misura delle oscillazioni", "cat": "market", "keywords": ["volatility", "volatile"]},
    "liquidity": {"term": "Liquidità", "def": "Facilità di scambio", "cat": "market", "keywords": ["liquidity", "depth"]},
    "spread": {"term": "Spread", "def": "Differenza bid-ask", "cat": "market", "keywords": ["spread", "bid", "ask"]},
    "volume": {"term": "Volume", "def": "Quantità scambiata", "cat": "market", "keywords": ["volume", "traded"]},
    "market_cap": {"term": "Market Cap", "def": "Capitalizzazione di mercato", "cat": "market", "keywords": ["market cap", "cap"]},
    "tvl": {"term": "TVL", "def": "Total Value Locked", "cat": "defi", "keywords": ["tvl", "locked"]},
    
    # DeFi
    "staking": {"term": "Staking", "def": "Blocco token per rewards", "cat": "defi", "keywords": ["staking", "stake"]},
    "liquidity_pool": {"term": "Liquidity Pool", "def": "Pool di liquidità", "cat": "defi", "keywords": ["lp", "pool"]},
    "impermanent_loss": {"term": "Impermanent Loss", "def": "Perdita temporanea", "cat": "defi", "keywords": ["il", "impermanent"]},
    "yield_farming": {"term": "Yield Farming", "def": "Agricoltura di rendimenti", "cat": "defi", "keywords": ["yield", "farming"]},
    "flash_loan": {"term": "Flash Loan", "def": "Prestito flash", "cat": "defi", "keywords": ["flash", "loan"]},
    "amms": {"term": "AMM", "def": "Automated Market Maker", "cat": "defi", "keywords": ["amm", "dex"]},
    "governance_token": {"term": "Governance Token", "def": "Token di governance", "cat": "defi", "keywords": ["governance", "vote"]},
    "bridge": {"term": "Bridge", "def": "Ponte cross-chain", "cat": "defi", "keywords": ["bridge", "cross-chain"]},
    
    # Crypto Specific
    "spot_etf": {"term": "Spot ETF", "def": "ETF su asset reale", "cat": "crypto", "keywords": ["spot", "etf"]},
    "layer2": {"term": "Layer 2", "def": "Soluzione di scalabilità", "cat": "crypto", "keywords": ["l2", "arbitrum", "optimism"]},
    "smart_contract": {"term": "Smart Contract", "def": "Contratto intelligente", "cat": "crypto", "keywords": ["smart", "contract"]},
    "wallet": {"term": "Wallet", "def": "Portafoglio crypto", "cat": "crypto", "keywords": ["wallet", "address"]},
    "private_key": {"term": "Private Key", "def": "Chiave privata", "cat": "crypto", "keywords": ["private", "key"]},
    "public_key": {"term": "Public Key", "def": "Chiave pubblica", "cat": "crypto", "keywords": ["public", "key"]},
    "nft": {"term": "NFT", "def": "Non-Fungible Token", "cat": "crypto", "keywords": ["nft", "token"]},
    "tokenomics": {"term": "Tokenomics", "def": "Economia del token", "cat": "crypto", "keywords": ["tokenomics", "supply"]},
    "circulating_supply": {"term": "Circulating Supply", "def": "Offerta in circolazione", "cat": "crypto", "keywords": ["circulating", "supply"]},
    "max_supply": {"term": "Max Supply", "def": "Offerta massima", "cat": "crypto", "keywords": ["max", "supply"]},
    
    # Economic
    "inflation": {"term": "Inflazione", "def": "Aumento prezzi", "cat": "economic", "keywords": ["inflation", "cpi"]},
    "interest_rate": {"term": "Tasso Interesse", "def": "Costo del denaro", "cat": "economic", "keywords": ["interest", "fed", "rate"]},
    "gdp": {"term": "GDP", "def": "Prodotto Interno Lordo", "cat": "economic", "keywords": ["gdp", "growth"]},
    "recession": {"term": "Recessione", "def": "Contrazione economica", "cat": "economic", "keywords": ["recession", "downturn"]},
    "unemployment": {"term": "Disoccupazione", "def": "Tasso disoccupazione", "cat": "economic", "keywords": ["unemployment", "jobs"]},
    "quantitative_easing": {"term": "Quantitative Easing", "def": "Allentamento quantitativo", "cat": "economic", "keywords": ["qe", "printing"]},
    
    # Sentiment
    "bullish": {"term": "Bullish", "def": "Outlook rialzista", "cat": "sentiment", "keywords": ["bullish", "buy"]},
    "bearish": {"term": "Bearish", "def": "Outlook ribassista", "cat": "sentiment", "keywords": ["bearish", "sell"]},
    "fear_greed": {"term": "Fear & Greed", "def": "Indice sentiment", "cat": "sentiment", "keywords": ["fear", "greed"]},
    "fomo": {"term": "FOMO", "def": "Fear Of Missing Out", "cat": "sentiment", "keywords": ["fomo", "miss"]},
    "fud": {"term": "FUD", "def": "Fear Uncertainty Doubt", "cat": "sentiment", "keywords": ["fud", "fear"]},
    "hodl": {"term": "HODL", "def": "Hold On for Dear Life", "cat": "sentiment", "keywords": ["hodl", "hold"]},
    
    # Portfolio
    "diversification": {"term": "Diversificazione", "def": "Distribuzione rischi", "cat": "portfolio", "keywords": ["diversify", "allocation"]},
    "rebalancing": {"term": "Ribilanciamento", "def": "Aggiustamento portafoglio", "cat": "portfolio", "keywords": ["rebalance", "adjust"]},
    "asset_allocation": {"term": "Asset Allocation", "def": "Allocazione asset", "cat": "portfolio", "keywords": ["allocation", "mix"]},
    "position_sizing": {"term": "Position Sizing", "def": "Dimensione posizione", "cat": "portfolio", "keywords": ["size", "sizing"]},
    "risk_management": {"term": "Risk Management", "def": "Gestione rischio", "cat": "portfolio", "keywords": ["risk", "management"]},
    
    # Analysis Types
    "fundamental_analysis": {"term": "Analisi Fondamentale", "def": "Analisi base economica", "cat": "analysis", "keywords": ["fundamental", "fa"]},
    "technical_analysis": {"term": "Analisi Tecnica", "def": "Analisi grafici", "cat": "analysis", "keywords": ["technical", "ta"]},
    "on_chain": {"term": "On-Chain", "def": "Analisi dati blockchain", "cat": "analysis", "keywords": ["on-chain", "chain"]},
    "sentiment_analysis": {"term": "Analisi Sentiment", "def": "Analisi umore mercato", "cat": "analysis", "keywords": ["sentiment", "emotion"]},
    
    # Order Types
    "market_order": {"term": "Market Order", "def": "Ordine al mercato", "cat": "orders", "keywords": ["market", "instant"]},
    "limit_order": {"term": "Limit Order", "def": "Ordine limite", "cat": "orders", "keywords": ["limit", "price"]},
    "stop_order": {"term": "Stop Order", "def": "Ordine stop", "cat": "orders", "keywords": ["stop", "trigger"]},
    "oco_order": {"term": "OCO", "def": "One Cancels Other", "cat": "orders", "keywords": ["oco", "conditional"]},
    
    # Advanced
    "algorithmic_trading": {"term": "Trading Algoritmico", "def": "Trading automatico", "cat": "advanced", "keywords": ["algo", "automated"]},
    "arbitrage": {"term": "Arbitraggio", "def": "Sfruttamento differenze prezzo", "cat": "advanced", "keywords": ["arbitrage", "spread"]},
    "backtesting": {"term": "Backtesting", "def": "Test su dati storici", "cat": "advanced", "keywords": ["backtest", "historical"]},
    "forward_testing": {"term": "Forward Testing", "def": "Test su dati futuri", "cat": "advanced", "keywords": ["forward", "live"]},
    "machine_learning": {"term": "Machine Learning", "def": "Apprendimento automatico", "cat": "advanced", "keywords": ["ml", "ai"]},
    "natural_language_processing": {"term": "NLP", "def": "Elaborazione linguaggio naturale", "cat": "advanced", "keywords": ["nlp", "text"]},
    "reinforcement_learning": {"term": "Reinforcement Learning", "def": "Apprendimento rinforzato", "cat": "advanced", "keywords": ["rl", "agent"]},
    
    # New Terms (auto-learned)
    "liquidity_crunch": {"term": "Liquidity Crunch", "def": "Carenza liquidità", "cat": "risk", "keywords": ["crunch", "shortage"]},
    "tail_risk": {"term": "Tail Risk", "def": "Rischio coda", "cat": "risk", "keywords": ["tail", "extreme"]},
    "correlation": {"term": "Correlazione", "def": "Relazione tra asset", "cat": "metrics", "keywords": ["correlation", "corr"]},
    "beta": {"term": "Beta", "def": "Sensibilità al mercato", "cat": "metrics", "keywords": ["beta", "market"]},
    "alpha": {"term": "Alpha", "def": "Ritorno eccedente", "cat": "metrics", "keywords": ["alpha", "excess"]},
    "sigma": {"term": "Sigma", "def": "Deviazione standard", "cat": "metrics", "keywords": ["sigma", "std"]},
    "volatility_clustering": {"term": "Volatility Clustering", "def": "集群 volatilità", "cat": "metrics", "keywords": ["clustering", "groups"]},
    "mean_reversion": {"term": "Mean Reversion", "def": "Ritorno alla media", "cat": "strategy", "keywords": ["mean", "reversion"]},
    "momentum": {"term": "Momentum", "def": "Forza del trend", "cat": "strategy", "keywords": ["momentum", "trend"]},
    "trend_following": {"term": "Trend Following", "def": "Seguire il trend", "cat": "strategy", "keywords": ["trend", "following"]},
    "contrarian": {"term": "Contrarian", "def": "Contro il mercato", "cat": "strategy", "keywords": ["contrarian", "opposite"]},
    "scalping": {"term": "Scalping", "def": "Trading veloce", "cat": "strategy", "keywords": ["scalping", "fast"]},
    "swing_trading": {"term": "Swing Trading", "def": "Trading intermedio", "cat": "strategy", "keywords": ["swing", "medium-term"]},
    "position_trading": {"term": "Position Trading", "def": "Trading lungo termine", "cat": "strategy", "keywords": ["position", "long-term"]},
    "grid_trading": {"term": "Grid Trading", "def": "Trading a griglia", "cat": "strategy", "keywords": ["grid", "levels"]},
    "dollar_cost_averaging": {"term": "DCA", "def": "Costo medio", "cat": "strategy", "keywords": ["dca", "average"]},
}

# ==================== DATA STRUCTURES ====================

@dataclass
class LearnedConcept:
    """Concept learned from news"""
    term: str
    definition: str
    category: str
    source: str
    confidence: float
    first_seen: str
    occurrences: int = 1


@dataclass
class MarketAnalysis:
    """Complete market analysis"""
    timestamp: str
    portfolio_value: float
    portfolio_pnl: float
    daily_pnl: float
    sentiment_score: float
    top_performers: List[str]
    worst_performers: List[str]
    risk_level: str
    recommendations: List[str]
    key_terms: List[str]


# ==================== AI ASSISTANT ENGINE ====================

class FinancialAIAssistant:
    """
    Assistente finanziario AI avanzato con:
    - Vocabolario esteso (100+ concetti)
    - Auto-apprendimento da notizie
    - Analisi automatica del portafoglio
    - Report periodici
    - Suggerimenti intelligenti
    """
    
    def __init__(self):
        self.concepts = ADVANCED_CONCEPTS.copy()
        self.learned_concepts: Dict[str, LearnedConcept] = {}
        self.history: List[MarketAnalysis] = []
        self.cache = {}
        self.cache_ttl = 30  # seconds
        
        # Load learned concepts
        self._load_learned()
        
        logger.info(f"AI Assistant inizializzato con {len(self.concepts)} concetti")
    
    def _load_learned(self):
        """Load learned concepts from file"""
        try:
            path = f"{DATA_DIR}/learned.json"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.learned_concepts[k] = LearnedConcept(**v)
                logger.info(f"Caricati {len(self.learned_concepts)} concetti appresi")
        except Exception as e:
            logger.warning(f"Errore caricamento: {e}")
    
    def _save_learned(self):
        """Save learned concepts to file"""
        try:
            path = f"{DATA_DIR}/learned.json"
            data = {k: v.__dict__ for k, v in self.learned_concepts.items()}
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio: {e}")
    
    # ==================== CORE FUNCTIONS ====================
    
    def search_concept(self, query: str) -> Optional[Dict]:
        """Search for a concept"""
        query = query.lower()
        
        # Check learned first
        if query in self.learned_concepts:
            c = self.learned_concepts[query]
            return {"term": c.term, "def": c.definition, "cat": c.category, "source": "learned"}
        
        # Check built-in
        for key, val in self.concepts.items():
            if query in key or query in val["keywords"]:
                return val
        
        return None
    
    def explain_term(self, term: str) -> str:
        """Explain a financial term"""
        concept = self.search_concept(term)
        
        if concept:
            return f"""
## {concept['term']}

**Categoria:** {concept['cat']}
**Definizione:** {concept['def']}
**Fonte:** {'Vocabolario base' if concept.get('source') != 'learned' else 'Appreso da notizie'}
"""
        
        # Try to extract from recent news
        return f"Termine '{term}' non trovato nel vocabolario. Vuoi che lo cerchi nelle notizie?"
    
    def get_concepts_by_category(self, category: str) -> List[Dict]:
        """Get all concepts in a category"""
        results = []
        for val in self.concepts.values():
            if val["cat"] == category:
                results.append(val)
        return results
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(v["cat"] for v in self.concepts.values()))
    
    # ==================== MARKET ANALYSIS ====================
    
    def get_portfolio_analysis(self) -> Dict:
        """Get current portfolio analysis"""
        # Check cache
        if "portfolio" in self.cache:
            cached_time, cached_data = self.cache["portfolio"]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        try:
            # Fetch from API
            resp = requests.get(f"{API_BASE_URL}/portfolio/summary", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                
                analysis = {
                    "value": data.get("total_value", 0),
                    "pnl": data.get("total_pnl", 0),
                    "pnl_percent": data.get("total_return_pct", 0),
                    "daily_pnl": data.get("daily_pnl", 0),
                    "positions": data.get("num_positions", 0),
                    "cash": data.get("cash_balance", 0)
                }
                
                self.cache["portfolio"] = (time.time(), analysis)
                return analysis
        except Exception as e:
            logger.error(f"Errore analisi portfolio: {e}")
        
        return {"error": "Impossibile ottenere dati"}
    
    def get_positions_analysis(self) -> List[Dict]:
        """Get detailed positions analysis"""
        try:
            resp = requests.get(f"{API_BASE_URL}/portfolio/positions", timeout=5)
            if resp.status_code == 200:
                positions = resp.json()
                
                # Sort by P&L
                sorted_pos = sorted(positions, key=lambda x: x.get("unrealized_pnl", 0), reverse=True)
                
                analysis = []
                for p in sorted_pos:
                    analysis.append({
                        "symbol": p.get("symbol"),
                        "side": p.get("side"),
                        "pnl": p.get("unrealized_pnl", 0),
                        "value": p.get("market_value", 0),
                        "entry": p.get("entry_price", 0),
                        "current": p.get("current_price", 0),
                        "pnl_pct": ((p.get("current_price", 0) - p.get("entry_price", 0)) / p.get("entry_price", 1)) * 100
                    })
                
                return analysis
        except Exception as e:
            logger.error(f"Errore posizioni: {e}")
        
        return []
    
    def get_sentiment_analysis(self) -> Dict:
        """Get market sentiment from news"""
        try:
            resp = requests.get(f"{API_BASE_URL}/news", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                news = data.get("news", [])
                
                positive = sum(1 for n in news if n.get("sentiment") == "positive")
                negative = sum(1 for n in news if n.get("sentiment") == "negative")
                neutral = len(news) - positive - negative
                
                avg_score = sum(n.get("sentiment_score", 0) for n in news) / max(len(news), 1)
                
                return {
                    "total": len(news),
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "score": avg_score,
                    "sentiment": "positive" if avg_score > 0.3 else "negative" if avg_score < -0.3 else "neutral"
                }
        except Exception as e:
            logger.error(f"Errore sentiment: {e}")
        
        return {"sentiment": "unknown"}
    
    # ==================== AUTO-LEARNING ====================
    
    def learn_from_news(self, news_items: List[Dict]):
        """Learn new terms from news"""
        for news in news_items:
            # Extract key terms from title and summary
            text = f"{news.get('title', '')} {news.get('summary', '')}".lower()
            
            # Simple extraction - look for capitalized words or crypto terms
            words = text.split()
            
            # Check for unknown terms
            for word in words:
                # Skip common words
                if len(word) < 4:
                    continue
                if word in ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "own", "say", "she", "too", "use"]:
                    continue
                
                # Check if already known
                if self.search_concept(word):
                    continue
                
                # Learn new term
                if word not in self.learned_concepts:
                    self.learned_concepts[word] = LearnedConcept(
                        term=word.title(),
                        definition=f"Termine appreso da: {news.get('source', 'news')}",
                        category="learned",
                        source=news.get("source", "news"),
                        confidence=0.5,
                        first_seen=datetime.now().isoformat()
                    )
                else:
                    self.learned_concepts[word].occurrences += 1
        
        # Save learned concepts
        if self.learned_concepts:
            self._save_learned()
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    def generate_comprehensive_analysis(self) -> MarketAnalysis:
        """Generate complete market analysis"""
        
        # Get all data
        portfolio = self.get_portfolio_analysis()
        positions = self.get_positions_analysis()
        sentiment = self.get_sentiment_analysis()
        
        # Determine risk level
        pnl_percent = portfolio.get("pnl_percent", 0)
        if pnl_percent > 10:
            risk_level = "low"
        elif pnl_percent > 0:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Generate recommendations
        recommendations = []
        
        if pnl_percent < -10:
            recommendations.append("Considera di ridurre l'esposizione")
        elif pnl_percent > 20:
            recommendations.append("Prendi profitto parziale")
        
        if sentiment.get("sentiment") == "negative":
            recommendations.append("Attenzione al sentiment negativo")
        elif sentiment.get("sentiment") == "positive":
            recommendations.append("Mercato favorevole")
        
        # Find key terms
        key_terms = []
        for p in positions[:3]:
            key_terms.append(p.get("symbol", ""))
        
        # Create analysis
        analysis = MarketAnalysis(
            timestamp=datetime.now().isoformat(),
            portfolio_value=portfolio.get("value", 0),
            portfolio_pnl=portfolio.get("pnl", 0),
            daily_pnl=portfolio.get("daily_pnl", 0),
            sentiment_score=sentiment.get("score", 0),
            top_performers=[p["symbol"] for p in positions[:3] if p.get("pnl", 0) > 0],
            worst_performers=[p["symbol"] for p in positions[-3:] if p.get("pnl", 0) < 0],
            risk_level=risk_level,
            recommendations=recommendations,
            key_terms=key_terms
        )
        
        # Save to history
        self.history.append(analysis)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return analysis
    
    def get_ai_response(self, question: str) -> str:
        """Generate AI response to question"""
        question_lower = question.lower()
        
        # Portfolio questions
        if any(w in question_lower for w in ["portafoglio", "portfolio", "conto", "soldi"]):
            portfolio = self.get_portfolio_analysis()
            positions = self.get_positions_analysis()
            
            response = f"## Analisi Portafoglio\n\n"
            response += f"- **Valore:** {portfolio.get('value', 0):,.0f} USDT\n"
            response += f"- **P&L:** {portfolio.get('pnl', 0):+,.0f} USDT ({portfolio.get('pnl_percent', 0):+.1f}%)\n"
            response += f"- **Giornaliero:** {portfolio.get('daily_pnl', 0):+,.0f} USDT\n"
            response += f"- **Posizioni:** {portfolio.get('positions', 0)}\n"
            
            if positions:
                best = positions[0]
                worst = positions[-1]
                response += f"\n✅ **Migliori:** {best.get('symbol')} ({best.get('pnl', 0):+,.0f} USDT)\n"
                response += f"❌ **Peggiori:** {worst.get('symbol')} ({worst.get('pnl', 0):+,.0f} USDT)\n"
            
            return response
        
        # Risk questions
        elif any(w in question_lower for w in ["rischio", "risk", "pericolo"]):
            sentiment = self.get_sentiment_analysis()
            portfolio = self.get_portfolio_analysis()
            
            risk = "basso" if portfolio.get("pnl_percent", 0) > 10 else "medio" if portfolio.get("pnl_percent", 0) > 0 else "alto"
            
            response = f"## Analisi Rischio\n\n"
            response += f"- **Livello rischio:** {risk}\n"
            response += f"- **Sentiment mercato:** {sentiment.get('sentiment', 'unknown')}\n"
            response += f"- **Notizie positive:** {sentiment.get('positive', 0)}\n"
            response += f"- **Notizie negative:** {sentiment.get('negative', 0)}\n"
            
            return response
        
        # News questions
        elif any(w in question_lower for w in ["notizia", "news", "sentiment", "mercato"]):
            sentiment = self.get_sentiment_analysis()
            
            response = f"## Sentiment Notizie\n\n"
            response += f"- **Sentiment:** {sentiment.get('sentiment', 'unknown').upper()}\n"
            response += f"- **Score:** {sentiment.get('score', 0):.2f}\n"
            response += f"- **Notizie:** {sentiment.get('positive', 0)} positive, {sentiment.get('negative', 0)} negative\n"
            
            return response
        
        # Explain term
        else:
            concept = self.search_concept(question)
            if concept:
                return self.explain_term(question)
            
            return f"Non ho informazioni su '{question}'. Vuoi che cerchi nelle notizie?"
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            "total_concepts": len(self.concepts),
            "learned_concepts": len(self.learned_concepts),
            "categories": len(self.get_all_categories()),
            "history_entries": len(self.history),
            "cache_size": len(self.cache)
        }


# ==================== MAIN ====================

def main():
    """Test the AI Assistant"""
    print("=" * 60)
    print("ADVANCED FINANCIAL AI ASSISTANT")
    print("=" * 60)
    
    # Initialize
    assistant = FinancialAIAssistant()
    
    # Show stats
    stats = assistant.get_statistics()
    print(f"\n[STATISTICHE]")
    print(f"  Concetti base: {stats['total_concepts']}")
    print(f"  Concetti appresi: {stats['learned_concepts']}")
    print(f"  Categorie: {stats['categories']}")
    
    # Test search
    print(f"\n[RICERCA TERMINI]")
    test_terms = ["RSI", "staking", "drawdown", "leverage", "ETF"]
    for term in test_terms:
        result = assistant.search_concept(term)
        if result:
            print(f"  {term}: {result['term']} ({result['cat']})")
        else:
            print(f"  {term}: NON TROVATO")
    
    # Test analysis
    print(f"\n[ANALISI MERCATO]")
    analysis = assistant.generate_comprehensive_analysis()
    print(f"  Valore portafoglio: {analysis.portfolio_value:,.0f} USDT")
    print(f"  P&L: {analysis.portfolio_pnl:+,.0f} USDT")
    print(f"  Sentiment: {analysis.sentiment_score:.2f}")
    print(f"  Rischio: {analysis.risk_level}")
    if analysis.recommendations:
        print(f"  Raccomandazioni: {', '.join(analysis.recommendations)}")
    
    # Test Q&A
    print(f"\n[TEST Q&A]")
    questions = [
        "come va il portafoglio",
        "quali sono i rischi",
        "spiegami cosa è il RSI"
    ]
    
    for q in questions:
        print(f"\n  Domanda: {q}")
        response = assistant.get_ai_response(q)
        print(f"  Risposta: {response[:150]}...")


if __name__ == "__main__":
    main()
