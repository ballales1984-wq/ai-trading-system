"""
Concept Engine v2.0
==================
Knowledge layer finanziario con:
- Vector DB (FAISS) per ricerca semantica
- Embeddings (sentence-transformers)
- Estrazione concetti ibrida (regole + embedding)

Autore: AI Trading System
Data: 2026-03-18
"""

import os
import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==================== CONCEPT CLASS ====================

@dataclass
class FinancialConcept:
    """Rappresenta un concetto finanziario con embedding"""
    id: str
    term: str
    definition: str
    category: str
    keywords: List[str]
    related_concepts: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    occurrence_count: int = 1
    confidence: float = 0.5
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        return data


@dataclass
class ConceptMatch:
    """Risultato di una ricerca semantica"""
    concept: FinancialConcept
    score: float
    match_type: str  # "semantic", "keyword", "hybrid"


# ==================== EMBEDDING MANAGER ====================

class EmbeddingManager:
    """
    Gestisce embeddings usando sentence-transformers.
    Grattuito e locale.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carica il modello di embedding"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Caricamento modello embedding: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Modello embedding caricato!")
        except ImportError:
            logger.warning("sentence-transformers non installato. Installare con: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"Errore caricamento modello: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Genera embedding per una lista di testi"""
        if self.model is None:
            return None
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Errore encoding: {e}")
            return None
    
    def encode_single(self, text: str) -> Optional[np.ndarray]:
        """Genera embedding per un singolo testo"""
        result = self.encode([text])
        return result[0] if result is not None and len(result) > 0 else None


# ==================== VECTOR STORE (FAISS) ====================

class VectorStore:
    """
    Vector store usando FAISS (gratuito, velocissimo).
    Alternative: Chroma, Weaviate (più pesanti)
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.concept_ids: List[str] = []
        self._init_index()
    
    def _init_index(self):
        """Inizializza index FAISS"""
        try:
            import faiss
            # IndexFlatIP = Inner Product (cosine similarity con vettori normalizzati)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("FAISS index inizializzato!")
        except ImportError:
            logger.warning("FAISS non installato. Installare con: pip install faiss-cpu")
            self.index = None
        except Exception as e:
            logger.error(f"Errore inizializzazione FAISS: {e}")
            self.index = None
    
    def add_vectors(self, ids: List[str], vectors: np.ndarray):
        """Aggiunge vettori all'indice"""
        if self.index is None:
            logger.warning("FAISS non disponibile")
            return
        
        # Normalizza per cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors_normalized = vectors / norms
        
        self.index.add(vectors_normalized)
        self.concept_ids.extend(ids)
        logger.info(f"Aggiunti {len(ids)} vettori all'indice")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Cerca i k vettori più simili"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalizza query
        query_norm = query_vector / np.linalg.norm(query_vector)
        query_norm = query_norm.reshape(1, -1)
        
        # Ricerca
        scores, indices = self.index.search(query_norm, min(k, self.index.ntotal))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.concept_ids):
                results.append((self.concept_ids[idx], float(score)))
        
        return results


# ==================== FINANCIAL CONCEPTS DATABASE ====================

FINANCIAL_CONCEPTS = {
    # Trading Basics
    "long_position": {
        "term": "Long Position",
        "definition": "Posizione d'acquisto che guadagna quando il prezzo sale",
        "category": "trading",
        "keywords": ["buy", "long", "acquisto", "rialzo", "rialzista", "purchase"]
    },
    "short_position": {
        "term": "Short Position",
        "definition": "Posizione di vendita che guadagna quando il prezzo scende",
        "category": "trading",
        "keywords": ["sell", "short", "ribasso", "ribassista", "shortare", "shorting"]
    },
    "stop_loss": {
        "term": "Stop Loss",
        "definition": "Ordine automatico per limitare le perdite a un livello predefined",
        "category": "risk_management",
        "keywords": ["stop", "stop loss", "protezione", "limite", "exit"]
    },
    "take_profit": {
        "term": "Take Profit",
        "definition": "Ordine automatico per chiudere in profitto a un target",
        "category": "risk_management",
        "keywords": ["take profit", "target", "obiettivo", "chiusura", "exit profit"]
    },
    "leverage": {
        "term": "Leverage",
        "definition": "Moltiplicatore che amplifica l'esposizione con meno capitale",
        "category": "trading",
        "keywords": ["leverage", "leva", "margine", "x10", "x100", "leveraged"]
    },
    
    # Risk Metrics
    "var": {
        "term": "Value at Risk (VaR)",
        "definition": "Massima perdita attesa in un intervallo di confidenza (es. 95%)",
        "category": "risk",
        "keywords": ["var", "value at risk", "rischio", "confidenza", "maximum loss"]
    },
    "cvar": {
        "term": "Conditional VaR (CVaR)",
        "definition": "Perdita attesa media quando si supera il VaR (Expected Shortfall)",
        "category": "risk",
        "keywords": ["cvar", "expected shortfall", "tail risk", "average loss"]
    },
    "drawdown": {
        "term": "Drawdown",
        "definition": "Calo massimo dal picco al minimo del portafoglio",
        "category": "risk",
        "keywords": ["drawdown", "perdita", "calo", "ribasso", "peak to trough"]
    },
    "sharpe_ratio": {
        "term": "Sharpe Ratio",
        "definition": "Ritorno aggiustato per il rischio (excess return / std dev)",
        "category": "metrics",
        "keywords": ["sharpe", "ratio", "performance", "risk adjusted", "risk-reward"]
    },
    "sortino_ratio": {
        "term": "Sortino Ratio",
        "definition": "Sharpe Ratio usando solo la volatilità negativa (downside)",
        "category": "metrics",
        "keywords": ["sortino", "downside risk", "downside volatility"]
    },
    
    # Market Terms
    "volatility": {
        "term": "Volatilità",
        "definition": "Misura delle oscillazioni dei prezzi nel tempo",
        "category": "market",
        "keywords": ["volatility", "volatile", "oscillazione", "instabilita", "variance"]
    },
    "liquidity": {
        "term": "Liquidità",
        "definition": "Facilità di comprare/vendere un asset senza impattare il prezzo",
        "category": "market",
        "keywords": ["liquidity", "liquidita", "depth", "volume", "slippage"]
    },
    "spread": {
        "term": "Spread",
        "definition": "Differenza tra prezzo bid (acquisto) e ask (vendita)",
        "category": "market",
        "keywords": ["spread", "bid", "ask", "denaro", "lettera", "bid-ask"]
    },
    "volume": {
        "term": "Volume",
        "definition": "Quantità di asset scambiati in un periodo",
        "category": "market",
        "keywords": ["volume", "trading volume", "scambi", "turnover"]
    },
    "market_cap": {
        "term": "Market Cap",
        "definition": "Valore totale di un asset (prezzo * offerta in circolazione)",
        "category": "market",
        "keywords": ["market cap", "capitalization", "valuation"]
    },
    
    # Technical Analysis
    "rsi": {
        "term": "RSI",
        "definition": "Relative Strength Index - indicatore di momentum (0-100)",
        "category": "technical",
        "keywords": ["rsi", "momentum", "overbought", "oversold", "strength"]
    },
    "macd": {
        "term": "MACD",
        "definition": "Moving Average Convergence Divergence - indicatore di trend",
        "category": "technical",
        "keywords": ["macd", "trend", "moving average", "convergence", "divergence", "signal"]
    },
    "moving_average": {
        "term": "Media Mobile",
        "definition": "Media dei prezzi su un periodo (SMA, EMA)",
        "category": "technical",
        "keywords": ["ma", "moving average", "sma", "ema", "media mobile", "average price"]
    },
    "bollinger_bands": {
        "term": "Bollinger Bands",
        "definition": "Bande di volatilità basate su media mobile ± 2 deviazioni standard",
        "category": "technical",
        "keywords": ["bollinger", "bands", "volatility", "envelope", "bands"]
    },
    "support": {
        "term": "Supporto",
        "definition": "Livello di prezzo dove la domanda supera l'offerta",
        "category": "technical",
        "keywords": ["support", "supporto", "bottom", "floor", "demand"]
    },
    "resistance": {
        "term": "Resistenza",
        "definition": "Livello di prezzo dove l'offerta supera la domanda",
        "category": "technical",
        "keywords": ["resistance", "resistenza", "ceiling", "tetto", "supply"]
    },
    
    # DeFi & Crypto
    "staking": {
        "term": "Staking",
        "definition": "Blocco di token per validare transazioni e ricevere rewards",
        "category": "defi",
        "keywords": ["staking", "stake", "reward", "validator", "delegation"]
    },
    "liquidity_pool": {
        "term": "Liquidity Pool",
        "definition": "Pool di liquidità per trading automatizzato (AMM)",
        "category": "defi",
        "keywords": ["liquidity pool", "amm", "dex", "swap", "pool"]
    },
    "impermanent_loss": {
        "term": "Impermanent Loss",
        "definition": "Perdita temporanea vs holding nei liquidity pool",
        "category": "defi",
        "keywords": ["impermanent loss", "il", "divergence", " IL"]
    },
    "yield_farming": {
        "term": "Yield Farming",
        "definition": "Strategia per massimizzare i rendimenti spostando liquidity",
        "category": "defi",
        "keywords": ["yield farming", "farming", "apy", "apr", "yield"]
    },
    "layer2": {
        "term": "Layer 2",
        "definition": "Soluzioni di scalabilità sulla blockchain principale (L2)",
        "category": "crypto",
        "keywords": ["l2", "layer 2", "arbitrum", "optimism", "rollup", "scalability"]
    },
    "spot_etf": {
        "term": "Spot ETF",
        "definition": "ETF che detiene l'asset fisico (es. Bitcoin spot ETF)",
        "category": "crypto",
        "keywords": ["spot etf", "bitcoin etf", "ether etf", "exchange traded"]
    },
    
    # Economic Terms
    "inflation": {
        "term": "Inflazione",
        "definition": "Aumento generalizzato dei prezzi nel tempo",
        "category": "economics",
        "keywords": ["inflation", "inflazione", "cpi", "ppi", "price index"]
    },
    "interest_rate": {
        "term": "Tasso d'interesse",
        "definition": "Costo del denaro prestato, strumento di politica monetaria",
        "category": "economics",
        "keywords": ["interest rate", "tasso", "fed", "ecb", "policy", "monetary"]
    },
    "gdp": {
        "term": "GDP",
        "definition": "Prodotto Interno Lordo - misura dell'attività economica",
        "category": "economics",
        "keywords": ["gdp", "gdp growth", "pil", "economy"]
    },
    "recession": {
        "term": "Recessione",
        "definition": "Contrazione economica per due trimestri consecutivi",
        "category": "economics",
        "keywords": ["recession", "recessione", "downturn", "contraction", "economic contraction"]
    },
    "unemployment": {
        "term": "Unemployment",
        "definition": "Tasso di disoccupazione - indicatore economico chiave",
        "category": "economics",
        "keywords": ["unemployment", "jobs", "labor", "employment", "payroll"]
    },
    
    # Sentiment
    "bullish": {
        "term": "Bullish",
        "definition": "Outlook positivo sui prezzi, aspettativa di rialzo",
        "category": "sentiment",
        "keywords": ["bullish", "rialzista", "buy", "accumulation", "optimistic"]
    },
    "bearish": {
        "term": "Bearish",
        "definition": "Outlook negativo sui prezzi, aspettativa di ribasso",
        "category": "sentiment",
        "keywords": ["bearish", "ribassista", "sell", "distribution", "pessimistic"]
    },
    "fear_greed": {
        "term": "Fear & Greed Index",
        "definition": "Indicatore del sentiment di mercato (0=Extreme Fear, 100=Extreme Greed)",
        "category": "sentiment",
        "keywords": ["fear", "greed", "sentiment", "emotion", "market sentiment"]
    },
    
    # Portfolio
    "diversification": {
        "term": "Diversificazione",
        "definition": "Distribuzione del capitale su asset diversi per ridurre rischio",
        "category": "portfolio",
        "keywords": ["diversification", "diversify", "portfolio", "allocazione", "spread"]
    },
    "rebalancing": {
        "term": "Ribilanciamento",
        "definition": "Aggiustamento periodico del portafoglio per mantenere allocazione target",
        "category": "portfolio",
        "keywords": ["rebalancing", "ribilanciamento", "adjust", "reallocate"]
    },
    "asset_allocation": {
        "term": "Asset Allocation",
        "definition": "Distribuzione del capitale tra diverse asset classi",
        "category": "portfolio",
        "keywords": ["allocation", "allocazione", "mix", "weights", "distribution"]
    },
    
    # Risk
    "tail_risk": {
        "term": "Tail Risk",
        "definition": "Rischio di eventi estremi rari (black swan)",
        "category": "risk",
        "keywords": ["tail risk", "black swan", "extreme", "outlier"]
    },
    "liquidity_crunch": {
        "term": "Liquidity Crunch",
        "definition": "Carenza improvvisa di liquidità nel mercato",
        "category": "risk",
        "keywords": ["liquidity crunch", "liquidity crisis", "funding crisis"]
    },
    "correlation": {
        "term": "Correlazione",
        "definition": "Misura di come due asset si muovono insieme",
        "category": "risk",
        "keywords": ["correlation", "correlazione", "beta", "relationship"]
    },
    
    # Regimes
    "trending_market": {
        "term": "Trending Market",
        "definition": "Mercato con trend definito (rialzista o ribassista)",
        "category": "regime",
        "keywords": ["trend", "trending", "directional", "momentum"]
    },
    "range_bound": {
        "term": "Range Bound",
        "definition": "Mercato che oscilla tra supporto e resistenza",
        "category": "regime",
        "keywords": ["range", "sideways", "consolidation", "oscillating"]
    },
    "high_volatility": {
        "term": "High Volatility",
        "definition": "Periodo di alta volatilità con grandi movimenti",
        "category": "regime",
        "keywords": ["high volatility", "volatile", "turbulent", "chaotic"]
    },
    "low_volatility": {
        "term": "Low Volatility",
        "definition": "Periodo di bassa volatilità e consolidamento",
        "category": "regime",
        "keywords": ["low volatility", "calm", "quiet", "stable"]
    },
}


# ==================== CONCEPT ENGINE ====================

class ConceptEngine:
    """
    Motore di concetti finanziari con:
    - Vector search (FAISS)
    - Embeddings (sentence-transformers)
    - Hybrid search (semantic + keyword)
    """
    
    def __init__(
        self,
        storage_path: str = "data/concept_engine",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Inizializza componenti
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.vector_store = VectorStore(dimension=384)
        
        # Database concetti
        self.concepts: Dict[str, FinancialConcept] = {}
        
        # Carica o inizializza
        self._load_concepts()
        self._build_vector_index()
        
        logger.info(f"ConceptEngine inizializzato con {len(self.concepts)} concetti")
    
    def _load_concepts(self):
        """Carica concetti da file"""
        try:
            file_path = os.path.join(self.storage_path, "concepts.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        self.concepts[item['id']] = FinancialConcept(**item)
                logger.info(f"Caricati {len(self.concepts)} concetti")
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
    
    def _build_vector_index(self):
        """Costruisce indice vettoriale"""
        if not self.concepts:
            self._initialize_base_concepts()
        
        # Genera testi per embedding
        texts = []
        ids = []
        
        for concept_id, concept in self.concepts.items():
            # Combina term + definition + keywords per embedding
            text = f"{concept.term}: {concept.definition}. Keywords: {', '.join(concept.keywords)}"
            texts.append(text)
            ids.append(concept_id)
        
        # Genera embeddings
        if texts and self.embedding_manager.model:
            embeddings = self.embedding_manager.encode(texts)
            if embeddings is not None:
                self.vector_store.add_vectors(ids, embeddings)
                logger.info(f"Indice vettoriale costruito con {len(ids)} vettori")
    
    def _initialize_base_concepts(self):
        """Inizializza con concetti base"""
        for key, value in FINANCIAL_CONCEPTS.items():
            concept = FinancialConcept(
                id=key,
                term=value['term'],
                definition=value['definition'],
                category=value['category'],
                keywords=value['keywords'],
                confidence=1.0
            )
            self.concepts[key] = concept
        
        self._save_concepts()
        logger.info(f"Inizializzati {len(self.concepts)} concetti base")
    
    def search(
        self,
        query: str,
        k: int = 5,
        hybrid: bool = True
    ) -> List[ConceptMatch]:
        """
        Ricerca concetti con hybrid search.
        
        Args:
            query: Testo di ricerca
            k: Numero di risultati
            hybrid: Se True, combina semantic + keyword
        
        Returns:
            Lista di ConceptMatch ordinati per rilevanza
        """
        results = []
        
        # 1. Ricerca semantica (vector)
        if self.embedding_manager.model and self.vector_store.index:
            query_embedding = self.embedding_manager.encode_single(query)
            if query_embedding is not None:
                vector_results = self.vector_store.search(query_embedding, k=k*2)
                for concept_id, score in vector_results:
                    if concept_id in self.concepts:
                        results.append(ConceptMatch(
                            concept=self.concepts[concept_id],
                            score=score,
                            match_type="semantic"
                        ))
        
        # 2. Ricerca keyword (se hybrid)
        if hybrid:
            keyword_results = self._keyword_search(query, k=k*2)
            for concept, kw_score in keyword_results:
                # Aggiungi o boosta risultato esistente
                existing = next((r for r in results if r.concept.id == concept.id), None)
                if existing:
                    # Media pesata: semantic 70% + keyword 30%
                    existing.score = (existing.score * 0.7) + (kw_score * 0.3)
                    existing.match_type = "hybrid"
                else:
                    results.append(ConceptMatch(
                        concept=concept,
                        score=kw_score,
                        match_type="keyword"
                    ))
        
        # Ordina per score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:k]
    
    def _keyword_search(self, query: str, k: int = 5) -> List[Tuple[FinancialConcept, float]]:
        """Ricerca keyword semplice"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        matches = []
        for concept in self.concepts.values():
            score = 0
            
            # Match nel termine
            if concept.term.lower() in query_lower:
                score += 0.5
            
            # Match nelle keywords
            for kw in concept.keywords:
                if kw.lower() in query_lower:
                    score += 0.3
                elif kw.lower() in query_words:
                    score += 0.2
            
            if score > 0:
                matches.append((concept, min(score, 1.0)))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]
    
    def extract_from_text(self, text: str, k: int = 5) -> List[ConceptMatch]:
        """Estrae concetti rilevanti da un testo (news, article)"""
        return self.search(text, k=k, hybrid=True)
    
    def add_concept(
        self,
        term: str,
        definition: str,
        category: str = "general",
        keywords: List[str] = None,
        related: List[str] = None,
        confidence: float = 0.5
    ) -> FinancialConcept:
        """Aggiunge un nuovo concetto"""
        concept_id = term.lower().replace(" ", "_")
        
        # Genera embedding
        embedding = None
        if self.embedding_manager.model:
            text = f"{term}: {definition}"
            embedding = self.embedding_manager.encode_single(text)
        
        if concept_id in self.concepts:
            # Aggiorna esistente
            concept = self.concepts[concept_id]
            concept.occurrence_count += 1
            concept.last_updated = datetime.now().isoformat()
            concept.confidence = min(1.0, concept.confidence + 0.1)
            
            if keywords:
                for kw in keywords:
                    if kw not in concept.keywords:
                        concept.keywords.append(kw)
        else:
            # Crea nuovo
            concept = FinancialConcept(
                id=concept_id,
                term=term,
                definition=definition,
                category=category,
                keywords=keywords or [],
                related_concepts=related or [],
                embedding=embedding.tolist() if embedding is not None else None,
                confidence=confidence
            )
            self.concepts[concept_id] = concept
            
            # Aggiungi a vector index
            if embedding is not None:
                self.vector_store.add_vectors([concept_id], embedding.reshape(1, -1))
        
        self._save_concepts()
        return concept
    
    def get_by_category(self, category: str) -> List[FinancialConcept]:
        """Restituisce tutti i concetti di una categoria"""
        return [
            c for c in self.concepts.values()
            if c.category.lower() == category.lower()
        ]
    
    def get_all_categories(self) -> List[str]:
        """Restituisce tutte le categorie"""
        return list(set(c.category for c in self.concepts.values()))
    
    def get_statistics(self) -> Dict:
        """Statistiche sul concept engine"""
        categories = defaultdict(int)
        confidence_sum = 0
        
        for concept in self.concepts.values():
            categories[concept.category] += 1
            confidence_sum += concept.confidence
        
        return {
            "total_concepts": len(self.concepts),
            "categories": dict(categories),
            "avg_confidence": confidence_sum / len(self.concepts) if self.concepts else 0,
            "vector_index_size": self.vector_store.index.ntotal if self.vector_store.index else 0,
            "embedding_model": self.embedding_manager.model_name
        }
    
    def explain(self, concept_id: str) -> str:
        """Genera spiegazione per un concetto"""
        if concept_id not in self.concepts:
            return f"Concetto '{concept_id}' non trovato"
        
        concept = self.concepts[concept_id]
        
        related = []
        if concept.related_concepts:
            for rel_id in concept.related_concepts:
                if rel_id in self.concepts:
                    related.append(self.concepts[rel_id])
        
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


# ==================== SIMPLE SENTIMENT (GRATIS) ====================

def analyze_sentiment(text: str) -> Dict:
    """
    Analisi sentiment semplice e gratis.
    Per versione avanzata: usa FinBERT (richiede installazione)
    """
    positive_words = [
        "bullish", "surge", "rally", "gain", "profit", "up", "high", "growth",
        "upgrade", "bull", "breakout", "moon", "green", "adoption", "record",
        "success", "positive", "increase", "boom", "bull", "optimistic"
    ]
    
    negative_words = [
        "bearish", "crash", "drop", "loss", "down", "low", "bear", "sell",
        "decline", "warning", "risk", "fear", "red", "recession", "bail",
        "hack", "fail", "negative", "decrease", "crash", "plunge", "pessimistic"
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
        "positive_count": pos_count,
        "negative_count": neg_count
    }


# ==================== MAIN TEST ====================

def main():
    """Test del Concept Engine"""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("CONCEPT ENGINE v2.0 - TEST")
    print("=" * 60)
    
    # Inizializza engine
    engine = ConceptEngine()
    
    # Statistiche
    stats = engine.get_statistics()
    print(f"\n📊 Statistiche:")
    print(f"   Concetti totali: {stats['total_concepts']}")
    print(f"   Vector index: {stats['vector_index_size']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    print(f"   Categorie: {stats['categories']}")
    
    # Test ricerca semantica
    print("\n🔍 Test ricerca semantica:")
    queries = [
        "come funziona lo staking di crypto",
        "misurare il rischio di portafoglio",
        "trend rialzista del mercato",
        "liquidità degli exchange"
    ]
    
    for q in queries:
        results = engine.search(q, k=3)
        print(f"\n   Query: '{q}'")
        for r in results:
            print(f"   - {r.concept.term} ({r.match_type}, score: {r.score:.2f})")
    
    # Test estrazione da news
    print("\n📰 Test estrazione da news:")
    news = """
    Bitcoin ha registrato un forte rialzo oggi, con la volatilità 
    che rimane alta nonostante il momentum positivo. Gli investitori 
    sono bullish sullo staking di ETH mentre attenzione al drawdown 
    che potrebbe aumentare in caso di correzione.
    """
    
    matches = engine.extract_from_text(news, k=5)
    for m in matches:
        print(f"   - {m.concept.term}: {m.score:.2f}")
    
    # Test sentiment
    print("\n💭 Test sentiment:")
    sent = analyze_sentiment(news)
    print(f"   Sentiment: {sent['sentiment']} (score: {sent['score']:.2f})")
    
    print("\n✅ Test completato!")


if __name__ == "__main__":
    main()
