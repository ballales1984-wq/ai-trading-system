"""
Semantic Vocabulary Module
=========================
Modulo avanzato per l'estrazione e gestione di concetti finanziari
usando NLP, embedding semantici e ricerca vettoriale.

Autore: AI Trading System
Data: 2026-03-17
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pickle
import os

logger = logging.getLogger(__name__)

# ==================== FINANCIAL CONCEPTS DATABASE ====================

FINANCIAL_CONCEPTS = {
    # Trading Basics
    "long_position": {
        "term": "Long Position",
        "definition": "Posizione d'acquisto che guadagna quando il prezzo sale",
        "category": "trading",
        "keywords": ["buy", "long", "acquisto", "rialzo", "rialzista"]
    },
    "short_position": {
        "term": "Short Position",
        "definition": "Posizione di vendita che guadagna quando il prezzo scende",
        "category": "trading",
        "keywords": ["sell", "short", "ribasso", "ribassista", "shortare"]
    },
    "stop_loss": {
        "term": "Stop Loss",
        "definition": "Ordine automatico per limitare le perdite",
        "category": "risk_management",
        "keywords": ["stop", "stop loss", "protezione", "limite"]
    },
    "take_profit": {
        "term": "Take Profit",
        "definition": "Ordine automatico per chiudere in profitto",
        "category": "risk_management",
        "keywords": ["take profit", "target", "obiettivo", "chiusura"]
    },
    "leverage": {
        "term": "Leverage",
        "definition": "Moltiplicatore che amplifica l'esposizione con meno capitale",
        "category": "trading",
        "keywords": ["leverage", "leva", "margine", "x10", "x100"]
    },
    
    # Risk Metrics
    "var": {
        "term": "Value at Risk (VaR)",
        "definition": "Massima perdita attesa in un intervallo di confidenza",
        "category": "risk",
        "keywords": ["var", "value at risk", "rischio", "confidenza"]
    },
    "cvar": {
        "term": "Conditional VaR (CVaR)",
        "definition": "Perdita attesa quando si supera il VaR",
        "category": "risk",
        "keywords": ["cvar", "expected shortfall", "tail risk"]
    },
    "drawdown": {
        "term": "Drawdown",
        "definition": "Calo massimo dal picco al minimo del portafoglio",
        "category": "risk",
        "keywords": ["drawdown", "perdita", "calo", "ribasso"]
    },
    "sharpe_ratio": {
        "term": "Sharpe Ratio",
        "definition": "Ritorno aggiustato per il rischio",
        "category": "metrics",
        "keywords": ["sharpe", "ratio", "performance", "risk adjusted"]
    },
    "sortino_ratio": {
        "term": "Sortino Ratio",
        "definition": "Sharpe Ratio usando solo la volatilità negativa",
        "category": "metrics",
        "keywords": ["sortino", "downside risk"]
    },
    
    # Market Terms
    "volatility": {
        "term": "Volatilità",
        "definition": "Misura delle oscillazioni dei prezzi",
        "category": "market",
        "keywords": ["volatility", "volatile", "oscillazione", "instabilita"]
    },
    "liquidity": {
        "term": "Liquidità",
        "definition": "Facilità di comprare/vendere un asset",
        "category": "market",
        "keywords": ["liquidity", "liquidita", "depth", "volume"]
    },
    "spread": {
        "term": "Spread",
        "definition": "Differenza tra bid e ask",
        "category": "market",
        "keywords": ["spread", "bid", "ask", "denaro", "lettera"]
    },
    "volume": {
        "term": "Volume",
        "definition": "Quantità di asset scambiati",
        "category": "market",
        "keywords": ["volume", "trading volume", "scambi"]
    },
    
    # Technical Analysis
    "rsi": {
        "term": "RSI",
        "definition": "Relative Strength Index - indicatore di momentum",
        "category": "technical",
        "keywords": ["rsi", "momentum", "overbought", "oversold"]
    },
    "macd": {
        "term": "MACD",
        "definition": "Moving Average Convergence Divergence",
        "category": "technical",
        "keywords": ["macd", "trend", "moving average", "convergence"]
    },
    "moving_average": {
        "term": "Media Mobile",
        "definition": "Media dei prezzi su un periodo",
        "category": "technical",
        "keywords": ["ma", "moving average", "sma", "ema", "media mobile"]
    },
    "support": {
        "term": "Supporto",
        "definition": "Livello dove i prezzi tendono a trovare interesse d'acquisto",
        "category": "technical",
        "keywords": ["support", "supporto", "bottom", "floor"]
    },
    "resistance": {
        "term": "Resistenza",
        "definition": "Livello dove i prezzi incontrano vendite",
        "category": "technical",
        "keywords": ["resistance", "resistenza", "ceiling", "tetto"]
    },
    
    # DeFi & Crypto
    "staking": {
        "term": "Staking",
        "definition": "Blocco di token per validare transazioni e ricevere rewards",
        "category": "defi",
        "keywords": ["staking", "stake", "reward", "validator"]
    },
    "liquidity_pool": {
        "term": "Liquidity Pool",
        "definition": "Pool di liquidità per trading automatizzato",
        "category": "defi",
        "keywords": ["liquidity pool", "ammm", "dex", "swap"]
    },
    "impermanent_loss": {
        "term": "Impermanent Loss",
        "definition": "Perdita temporanea nei liquidity pool",
        "category": "defi",
        "keywords": ["impermanent loss", "il", "divergence"]
    },
    "yield_farming": {
        "term": "Yield Farming",
        "definition": "Strategia per massimizzare i rendimenti",
        "category": "defi",
        "keywords": ["yield farming", "farming", "apy", "apr"]
    },
    "layer2": {
        "term": "Layer 2",
        "definition": "Soluzioni di scalabilità sulla blockchain principale",
        "category": "crypto",
        "keywords": ["l2", "layer 2", "arbitrum", "optimism", "rollup"]
    },
    "spot_etf": {
        "term": "Spot ETF",
        "definition": "ETF che detiene l'asset fisico",
        "category": "crypto",
        "keywords": ["spot etf", "bitcoin etf", "ether etf"]
    },
    
    # Economic Terms
    "inflation": {
        "term": "Inflazione",
        "definition": "Aumento generalizzato dei prezzi",
        "category": "economics",
        "keywords": ["inflation", "inflazione", "cpi", "ppi"]
    },
    "interest_rate": {
        "term": "Tasso d'interesse",
        "definition": "Costo del denaro prestato",
        "category": "economics",
        "keywords": ["interest rate", "tasso", "fed", "ecb", "policy"]
    },
    "gdp": {
        "term": "GDP",
        "definition": "Prodotto Interno Lordo",
        "category": "economics",
        "keywords": ["gdp", "gdp growth", "pil"]
    },
    "recession": {
        "term": "Recessione",
        "definition": "Contrazione economica per due trimestri consecutivi",
        "category": "economics",
        "keywords": ["recession", "recessione", "downturn", "contraction"]
    },
    
    # Sentiment
    "bullish": {
        "term": "Bullish",
        "definition": "Outlook positivo sui prezzi",
        "category": "sentiment",
        "keywords": ["bullish", "rialzista", "buy", "accumulation"]
    },
    "bearish": {
        "term": "Bearish",
        "definition": "Outlook negativo sui prezzi",
        "category": "sentiment",
        "keywords": ["bearish", "ribassista", "sell", "distribution"]
    },
    "fear_greed": {
        "term": "Fear & Greed Index",
        "definition": "Indicatore del sentiment di mercato",
        "category": "sentiment",
        "keywords": ["fear", "greed", "sentiment", "emotion"]
    },
    
    # Portfolio
    "diversification": {
        "term": "Diversificazione",
        "definition": "Distribuzione del capitale su asset diversi",
        "category": "portfolio",
        "keywords": ["diversification", "diversify", "portfolio", "allocazione"]
    },
    "rebalancing": {
        "term": "Ribilanciamento",
        "definition": "Aggiustamento periodico del portafoglio",
        "category": "portfolio",
        "keywords": ["rebalancing", "ribilanciamento", "adjust"]
    },
    "asset_allocation": {
        "term": "Asset Allocation",
        "definition": "Distribuzione del capitale tra asset classi",
        "category": "portfolio",
        "keywords": ["allocation", "allocazione", "mix", "weights"]
    },
    
    # New/Recent Terms (learned)
    "tail_risk": {
        "term": "Tail Risk",
        "definition": "Rischio di eventi estremi rari",
        "category": "risk",
        "keywords": ["tail risk", "black swan", "extreme"]
    },
    "liquidity_crunch": {
        "term": "Liquidity Crunch",
        "definition": "Carenza improvvisa di liquidità",
        "category": "risk",
        "keywords": ["liquidity crunch", "liquidity crisis"]
    },
    "earnings_miss": {
        "term": "Earnings Miss",
        "definition": "Risultati inferiori alle attese",
        "category": "earnings",
        "keywords": ["earnings miss", "guidance lowered", "outlook"]
    }
}


# ==================== CONCEPT CLASS ====================

@dataclass
class Concept:
    """Rappresenta un concetto finanziario"""
    id: str
    term: str
    definition: str
    category: str
    keywords: List[str]
    related_concepts: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    occurrence_count: int = 1
    confidence: float = 0.5
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ==================== SEMANTIC VOCABULARY ENGINE ====================

class SemanticVocabularyEngine:
    """
    Motore semantico per gestione vocabolario finanziario.
    Supporta estrazione concetti, embedding e ricerca semantica.
    """
    
    def __init__(self, storage_path: str = "data/vocabulary"):
        self.storage_path = storage_path
        self.concepts: Dict[str, Concept] = {}
        self._embeddings_cache: Dict[str, List[float]] = {}
        
        # Crea directory storage
        os.makedirs(storage_path, exist_ok=True)
        
        # Carica concetti esistenti
        self._load_concepts()
        
        # Inizializza con concetti base
        self._initialize_base_concepts()
        
        logger.info(f"SemanticVocabularyEngine inizializzato con {len(self.concepts)} concetti")
    
    def _load_concepts(self):
        """Carica concetti da file"""
        try:
            file_path = os.path.join(self.storage_path, "concepts.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        self.concepts[item['id']] = Concept(**item)
                logger.info(f"Caricati {len(self.concepts)} concetti da file")
        except Exception as e:
            logger.warning(f"Errore caricamento concetti: {e}")
    
    def _save_concepts(self):
        """Salva concetti su file"""
        try:
            file_path = os.path.join(self.storage_path, "concepts.json")
            data = [c.to_dict() for c in self.concepts.values()]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Errore salvataggio concetti: {e}")
    
    def _initialize_base_concepts(self):
        """Inizializza con concetti base se vuoto"""
        if not self.concepts:
            for key, value in FINANCIAL_CONCEPTS.items():
                concept = Concept(
                    id=key,
                    term=value['term'],
                    definition=value['definition'],
                    category=value['category'],
                    keywords=value['keywords'],
                    confidence=1.0
                )
                self.concepts[key] = concept
            self._save_concepts()
    
    def add_concept(
        self, 
        term: str, 
        definition: str, 
        category: str = "general",
        keywords: List[str] = None,
        related: List[str] = None,
        confidence: float = 0.5
    ) -> Concept:
        """Aggiunge un nuovo concetto"""
        # Crea ID unico
        concept_id = term.lower().replace(" ", "_")
        
        if concept_id in self.concepts:
            # Aggiorna concetto esistente
            concept = self.concepts[concept_id]
            concept.occurrence_count += 1
            concept.last_updated = datetime.now().isoformat()
            concept.confidence = min(1.0, concept.confidence + 0.1)
            
            # Aggiungi keywords nuove
            if keywords:
                for kw in keywords:
                    if kw not in concept.keywords:
                        concept.keywords.append(kw)
        else:
            # Crea nuovo concetto
            concept = Concept(
                id=concept_id,
                term=term,
                definition=definition,
                category=category,
                keywords=keywords or [],
                related_concepts=related or [],
                confidence=confidence
            )
            self.concepts[concept_id] = concept
        
        self._save_concepts()
        return concept
    
    def find_concept(self, query: str) -> Optional[Concept]:
        """Trova un concetto per keyword o termine"""
        query_lower = query.lower()
        
        # Cerca nel ID
        if query_lower in self.concepts:
            return self.concepts[query_lower]
        
        # Cerca nei termini
        for concept in self.concepts.values():
            if query_lower in concept.term.lower():
                return concept
            
            # Cerca nelle keywords
            for kw in concept.keywords:
                if query_lower in kw.lower():
                    return concept
        
        return None
    
    def search_by_category(self, category: str) -> List[Concept]:
        """Cerca concetti per categoria"""
        return [
            c for c in self.concepts.values() 
            if c.category.lower() == category.lower()
        ]
    
    def get_all_categories(self) -> List[str]:
        """Restituisce tutte le categorie"""
        return list(set(c.category for c in self.concepts.values()))
    
    def get_related(self, concept_id: str) -> List[Concept]:
        """Restituisce concetti correlati"""
        if concept_id not in self.concepts:
            return []
        
        concept = self.concepts[concept_id]
        related = []
        
        for rel_id in concept.related_concepts:
            if rel_id in self.concepts:
                related.append(self.concepts[rel_id])
        
        return related
    
    def get_statistics(self) -> Dict:
        """Restituisce statistiche sul vocabolario"""
        categories = defaultdict(int)
        confidence_sum = 0
        
        for concept in self.concepts.values():
            categories[concept.category] += 1
            confidence_sum += concept.confidence
        
        return {
            "total_concepts": len(self.concepts),
            "categories": dict(categories),
            "avg_confidence": confidence_sum / len(self.concepts) if self.concepts else 0,
            "most_common_category": max(categories.items(), key=lambda x: x[1])[0] if categories else None
        }
    
    def extract_concepts_from_text(self, text: str) -> List[Tuple[Concept, float]]:
        """
        Estrae concetti rilevanti da un testo.
        Usa matching semplice delle keywords.
        """
        text_lower = text.lower()
        matches = []
        
        for concept in self.concepts.values():
            score = 0
            
            # Match nel termine
            if concept.term.lower() in text_lower:
                score += 0.5
            
            # Match nelle keywords
            for kw in concept.keywords:
                if kw.lower() in text_lower:
                    score += 0.3
            
            # Boost per categoria rilevante
            for cat in ["trading", "risk", "market", "defi", "crypto"]:
                if cat in text_lower and concept.category == cat:
                    score += 0.2
            
            if score > 0:
                matches.append((concept, min(score, 1.0)))
        
        # Ordina per score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]  # Top 5
    
    def explain_term(self, term: str) -> str:
        """Genera spiegazione per un termine"""
        concept = self.find_concept(term)
        
        if not concept:
            # Prova a estrarre da testo
            matches = self.extract_concepts_from_text(term)
            if matches:
                concept = matches[0][0]
        
        if concept:
            related = self.get_related(concept.id)
            related_text = ""
            if related:
                related_text = "\n\n**Concetti correlati:**\n"
                for r in related[:3]:
                    related_text += f"- {r.term}: {r.definition[:60]}...\n"
            
            return f"""
## {concept.term}

**Categoria:** {concept.category}
**Confidence:** {concept.confidence:.0%}

**Definizione:**
{concept.definition}

**Keywords:** {', '.join(concept.keywords[:5])}
{related_text}
"""
        
        return f"Termine '{term}' non trovato nel vocabolario."


# ==================== KEYWORD EXTRACTION (Optional Enhancement) ====================

def extract_keywords_with_keybert(text: str, top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Estrae keywords usando KeyBERT.
    Richiede: pip install keybert
    """
    try:
        from keybert import KeyBERT
        
        # Usa modello leggero
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2),
            top_n=top_n,
            stop_words=None
        )
        return keywords
    except ImportError:
        logger.warning("KeyBERT non installato")
        return []
    except Exception as e:
        logger.error(f"Errore KeyBERT: {e}")
        return []


def analyze_sentiment_financial(text: str) -> Dict:
    """
    Analizza il sentiment di un testo finanziario.
    """
    positive_words = [
        "bullish", "surge", "rally", "gain", "profit", "up", "high",
        "growth", "upgrade", "bull", "breakout", "moon", "green",
        "adoption", "record", "success", "positive", "increase"
    ]
    
    negative_words = [
        "bearish", "crash", "drop", "loss", "down", "low", "bear",
        "sell", "decline", "warning", "risk", "fear", "red",
        "recession", "bail", "hack", "fail", "negative", "decrease"
    ]
    
    text_lower = text.lower()
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        sentiment = "positive"
        score = min(1.0, pos_count / (pos_count + neg_count + 1))
    elif neg_count > pos_count:
        sentiment = "negative"
        score = -min(1.0, neg_count / (pos_count + neg_count + 1))
    else:
        sentiment = "neutral"
        score = 0.0
    
    return {
        "sentiment": sentiment,
        "score": score,
        "positive_words": pos_count,
        "negative_words": neg_count
    }


# ==================== MAIN FUNCTION ====================

def main():
    """Test del modulo"""
    # Fix encoding for Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("TEST SEMANTIC VOCABULARY ENGINE")
    print("=" * 60)
    
    # Inizializza motore
    engine = SemanticVocabularyEngine()
    
    # Statistiche
    stats = engine.get_statistics()
    print(f"\n📊 Statistiche:")
    print(f"   Concetti totali: {stats['total_concepts']}")
    print(f"   Categorie: {stats['categories']}")
    print(f"   Avg confidence: {stats['avg_confidence']:.2%}")
    
    # Test ricerca
    print("\n🔍 Test ricerca:")
    
    test_terms = ["RSI", "staking", "drawdown", "volatilità"]
    for term in test_terms:
        concept = engine.find_concept(term)
        if concept:
            print(f"   ✓ {term} -> {concept.term} ({concept.category})")
        else:
            print(f"   ✗ {term} non trovato")
    
    # Test estrazione
    print("\n📝 Test estrazione da testo:")
    test_text = """
    Bitcoin ha registrato un forte rialzo oggi, con la volatilità 
    che rimane alta. Gli investitori sono bullish sullo staking di ETH.
    Attenzione al drawdown che potrebbe aumentare.
    """
    
    matches = engine.extract_concepts_from_text(test_text)
    for concept, score in matches:
        print(f"   - {concept.term}: {score:.2f}")
    
    # Test sentiment
    print("\n💭 Test sentiment:")
    sentiment = analyze_sentiment_financial(test_text)
    print(f"   Sentiment: {sentiment['sentiment']} (score: {sentiment['score']:.2f})")
    
    # Test spiegazione
    print("\n📖 Test spiegazione termine:")
    explanation = engine.explain_term("stop_loss")
    print(explanation[:200] + "...")


if __name__ == "__main__":
    main()
