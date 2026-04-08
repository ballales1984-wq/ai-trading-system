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

# Import shared vocabulary
try:
    from shared_vocabulary import SHARED_FINANCIAL_CONCEPTS
except ImportError:
    SHARED_FINANCIAL_CONCEPTS = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_assistant")

# ==================== CONFIGURATION ====================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
DATA_DIR = "data/ai_assistant"
os.makedirs(DATA_DIR, exist_ok=True)

# ==================== ENHANCED VOCABULARY ====================

# ADVANCED_CONCEPTS is now backed by SHARED_FINANCIAL_CONCEPTS
# for consistency across the entire system.
ADVANCED_CONCEPTS = {
    k: {
        "term": v["term"],
        "def": v["definition"],
        "cat": v["category"],
        "keywords": v["keywords"]
    }
    for k, v in SHARED_FINANCIAL_CONCEPTS.items()
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
        """Search for a concept using keys, keywords, and partial matches"""
        query = query.lower().strip()
        query_words = set(query.split())
        
        # Check learned first
        if query in self.learned_concepts:
            c = self.learned_concepts[query]
            return {"term": c.term, "def": c.definition, "cat": c.category, "source": "learned"}
        
        # Check built-in concepts
        for key, val in self.concepts.items():
            # 1. Exact key match or key-in-query
            clean_key = key.replace("_", " ")
            if query == key or query == clean_key or clean_key in query:
                return val
                
            # 2. Keyword match
            keywords = [k.lower() for k in val.get("keywords", [])]
            # If any keyword is in the query words
            if any(kw in query for kw in keywords) or any(kw in query_words for kw in keywords):
                return val
        
        # 3. Fuzzy match fallback for common crypto terms
        if "btc" in query or "bitcoin" in query:
            return self.concepts.get("market_cap") # Fallback to something relevant
            
        return None

    def explain_term(self, term: str) -> str:
        """Explain a financial term with premium formatting"""
        concept = self.search_concept(term)
        
        if concept:
            source_label = "📚 Vocabolario Base" if concept.get('source') != 'learned' else "🧠 Appreso dall'AI"
            
            response = f"### 🔍 {concept['term']}\n\n"
            response += f"> {concept['def']}\n\n"
            response += f"**📂 Categoria:** `{concept['cat'].upper()}`\n"
            response += f"**ℹ️ Fonte:** {source_label}\n"
            
            # Add related concepts if available
            if "keywords" in concept and len(concept["keywords"]) > 0:
                response += f"\n**🔗 Correlati:** {', '.join(concept['keywords'][:3])}\n"
                
            return response
        
        return f"🤔 Mi dispiace, ma il termine **'{term}'** non è ancora nel mio database economico. Vuoi che provi a scansionare le ultime notizie per cercarlo?"
    
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
                # If it's a very short query, just explain it
                if len(question_lower.split()) <= 3:
                    return self.explain_term(question)
                else:
                    # For longer questions, try to be more conversational
                    explanation = concept['def']
                    return f"Certamente! In base alla tua domanda, ecco una spiegazione di **{concept['term']}**:\n\n{explanation}\n\nTi serve sapere altro su questo argomento?"
            
            return f"❌ Non ho trovato informazioni specifiche su **'{question}'**. Prova a chiedermi del portafoglio, dei rischi o spiegazioni su termini tecnici come 'RSI' o 'Drawdown'."
    
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
