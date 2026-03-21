"""
Sentiment-Concept Bridge
=======================
Integration layer between SentimentAnalyzer and ConceptEngine.
Provides concept-aware sentiment analysis by combining:
- Semantic concept matching from ConceptEngine
- News sentiment from SentimentAnalyzer

This bridge enhances sentiment analysis by:
1. Identifying financial concepts mentioned in news
2. Using concept relationships to improve sentiment context
3. Providing semantic similarity between news and known concepts
4. Enabling concept-based sentiment filtering

Author: AI Trading System
Data: 2026-03-21
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import concept_engine components
try:
    from concept_engine import (
        ConceptEngine, 
        FinancialConcept, 
        ConceptMatch,
        FINANCIAL_CONCEPTS
    )
    CONCEPT_ENGINE_AVAILABLE = True
except ImportError:
    CONCEPT_ENGINE_AVAILABLE = False
    logger.warning("concept_engine not available. Install required: pip install sentence-transformers faiss-cpu")

# Try to import sentiment_news components
try:
    from sentiment_news import (
        SentimentAnalyzer,
        NewsItem,
        SentimentData
    )
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logger.warning("sentiment_news not available")


# ==================== DATA STRUCTURES ====================

@dataclass
class ConceptSentiment:
    """Sentiment enriched with concept information"""
    # Basic sentiment data
    asset: str
    timestamp: datetime
    sentiment_score: float
    confidence: float
    
    # Concept-enriched fields
    detected_concepts: List[str] = field(default_factory=list)
    concept_details: List[Dict[str, Any]] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Sentiment breakdown by concept
    concept_sentiments: Dict[str, float] = field(default_factory=dict)
    
    # News metadata
    news_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'asset': self.asset,
            'timestamp': self.timestamp.isoformat(),
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'detected_concepts': self.detected_concepts,
            'concept_details': self.concept_details,
            'related_concepts': self.related_concepts,
            'concept_sentiments': self.concept_sentiments,
            'news_count': self.news_count,
            'positive_count': self.positive_count,
            'negative_count': self.neutral_count,
            'neutral_count': self.neutral_count
        }


@dataclass
class ConceptNewsItem:
    """NewsItem enriched with concept annotations"""
    # Original news data
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    sentiment: float = 0.0
    relevance: float = 0.0
    category: str = "general"
    
    # Concept enrichment
    concepts_detected: List[str] = field(default_factory=list)
    concept_scores: Dict[str, float] = field(default_factory=dict)
    primary_concept: Optional[str] = None
    
    def to_news_item(self) -> NewsItem:
        """Convert back to standard NewsItem"""
        return NewsItem(
            title=self.title,
            description=self.description,
            source=self.source,
            url=self.url,
            published_at=self.published_at,
            sentiment=self.sentiment,
            relevance=self.relevance,
            category=self.category
        )


# ==================== SENTIMENT-CONCEPT BRIDGE ====================

class SentimentConceptBridge:
    """
    Bridge between SentimentAnalyzer and ConceptEngine.
    
    Provides:
    - Concept-aware sentiment analysis
    - Semantic news enrichment
    - Concept-based sentiment filtering
    - Enhanced concept detection using sentiment cues
    """
    
    def __init__(
        self,
        enable_semantic: bool = True,
        min_concept_score: float = 0.3,
        max_concepts_per_news: int = 5
    ):
        """
        Initialize the bridge.
        
        Args:
            enable_semantic: Enable semantic similarity matching
            min_concept_score: Minimum score to consider a concept match
            max_concepts_per_news: Maximum concepts to extract per news item
        """
        self.enable_semantic = enable_semantic
        self.min_concept_score = min_concept_score
        self.max_concepts_per_news = max_concepts_per_news
        
        # Initialize components
        self.sentiment_analyzer = None
        self.concept_engine = None
        
        self._initialize_components()
        
        # Cache for concept-keyword mapping
        self._concept_keyword_cache: Dict[str, List[str]] = {}
        self._build_keyword_cache()
        
        logger.info("SentimentConceptBridge initialized")
    
    def _initialize_components(self):
        """Initialize sentiment analyzer and concept engine"""
        if SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("SentimentAnalyzer loaded")
            except Exception as e:
                logger.error(f"Failed to initialize SentimentAnalyzer: {e}")
        else:
            logger.warning("SentimentAnalyzer not available")
        
        if CONCEPT_ENGINE_AVAILABLE:
            try:
                self.concept_engine = ConceptEngine()
                logger.info(f"ConceptEngine loaded with {len(self.concept_engine.concepts)} concepts")
            except Exception as e:
                logger.error(f"Failed to initialize ConceptEngine: {e}")
        else:
            logger.warning("ConceptEngine not available - using keyword-only mode")
    
    def _build_keyword_cache(self):
        """Build mapping from keywords to concepts"""
        if not CONCEPT_ENGINE_AVAILABLE:
            return
            
        try:
            for concept_id, concept_data in FINANCIAL_CONCEPTS.items():
                keywords = concept_data.get('keywords', [])
                for keyword in keywords:
                    if keyword not in self._concept_keyword_cache:
                        self._concept_keyword_cache[keyword] = []
                    self._concept_keyword_cache[keyword].append(concept_id)
        except Exception as e:
            logger.warning(f"Error building keyword cache: {e}")
    
    # ==================== CONCEPT DETECTION ====================
    
    def detect_concepts_in_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect financial concepts in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (concept_id, score) tuples
        """
        if not text:
            return []
        
        text_lower = text.lower()
        concepts_found = []
        
        # 1. Keyword-based detection (fast)
        for keyword, concept_ids in self._concept_keyword_cache.items():
            if keyword in text_lower:
                for concept_id in concept_ids:
                    # Avoid duplicates
                    if not any(c[0] == concept_id for c in concepts_found):
                        concepts_found.append((concept_id, 0.7))  # Base score
        
        # 2. Semantic similarity (if enabled and available)
        if self.enable_semantic and self.concept_engine is not None:
            try:
                semantic_matches = self.concept_engine.find_concepts(text, k=3)
                for match in semantic_matches:
                    concept_id = match.concept.id
                    score = match.score
                    
                    # Update existing or add new
                    existing = [i for i, c in enumerate(concepts_found) if c[0] == concept_id]
                    if existing:
                        # Blend semantic with keyword score
                        idx = existing[0]
                        old_score = concepts_found[idx][1]
                        concepts_found[idx] = (concept_id, (old_score + score) / 2)
                    else:
                        concepts_found.append((concept_id, score))
            except Exception as e:
                logger.debug(f"Semantic concept detection failed: {e}")
        
        # Sort by score and limit
        concepts_found.sort(key=lambda x: x[1], reverse=True)
        return concepts_found[:self.max_concepts_per_news]
    
    def enrich_news_with_concepts(self, news_items: List[NewsItem]) -> List[ConceptNewsItem]:
        """
        Enrich news items with concept annotations.
        
        Args:
            news_items: List of news items to enrich
            
        Returns:
            List of ConceptNewsItem with concept data
        """
        enriched_items = []
        
        for news in news_items:
            # Combine title and description for analysis
            text = f"{news.title} {news.description}"
            
            # Detect concepts
            concepts = self.detect_concepts_in_text(text)
            
            # Build enriched item
            enriched = ConceptNewsItem(
                title=news.title,
                description=news.description,
                source=news.source,
                url=news.url,
                published_at=news.published_at,
                sentiment=news.sentiment,
                relevance=news.relevance,
                category=news.category,
                concepts_detected=[c[0] for c in concepts],
                concept_scores=dict(concepts),
                primary_concept=concepts[0][0] if concepts else None
            )
            
            enriched_items.append(enriched)
        
        return enriched_items
    
    # ==================== SENTIMENT ANALYSIS ====================
    
    def analyze_asset_sentiment_with_concepts(
        self, 
        asset: str,
        use_concepts: bool = True
    ) -> ConceptSentiment:
        """
        Analyze sentiment for an asset with concept enrichment.
        
        Args:
            asset: Asset symbol
            use_concepts: Whether to include concept analysis
            
        Returns:
            ConceptSentiment with enriched data
        """
        if self.sentiment_analyzer is None:
            return self._empty_concept_sentiment(asset)
        
        # Get basic sentiment
        sentiment_data = self.sentiment_analyzer.analyze_asset_sentiment(asset)
        news = self.sentiment_analyzer.fetch_news(assets=[asset])
        
        # Detect concepts in news
        all_concepts: Dict[str, List[float]] = {}
        concept_news_mapping: Dict[str, List[int]] = {}
        
        for i, news_item in enumerate(news):
            text = f"{news_item.title} {news_item.description}"
            concepts = self.detect_concepts_in_text(text)
            
            for concept_id, score in concepts:
                if concept_id not in all_concepts:
                    all_concepts[concept_id] = []
                    concept_news_mapping[concept_id] = []
                
                all_concepts[concept_id].append(news_item.sentiment)
                concept_news_mapping[concept_id].append(i)
        
        # Calculate concept sentiments
        concept_sentiments = {}
        concept_details = []
        
        for concept_id, sentiments in all_concepts.items():
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                concept_sentiments[concept_id] = avg_sentiment
                
                # Get concept info if available
                concept_info = {}
                if CONCEPT_ENGINE_AVAILABLE and concept_id in FINANCIAL_CONCEPTS:
                    concept_info = FINANCIAL_CONCEPTS[concept_id]
                
                concept_details.append({
                    'id': concept_id,
                    'term': concept_info.get('term', concept_id),
                    'sentiment': avg_sentiment,
                    'mentions': len(sentiments),
                    'category': concept_info.get('category', 'unknown')
                })
        
        # Get related concepts (from primary concept)
        related_concepts = []
        if concept_details and CONCEPT_ENGINE_AVAILABLE:
            primary_id = concept_details[0]['id']
            if primary_id in self.concept_engine.concepts:
                related = self.concept_engine.concepts[primary_id].related_concepts
                related_concepts = related[:5]
        
        # Build result
        return ConceptSentiment(
            asset=asset,
            timestamp=datetime.now(),
            sentiment_score=sentiment_data.sentiment_score,
            confidence=sentiment_data.confidence,
            detected_concepts=list(all_concepts.keys()),
            concept_details=concept_details,
            related_concepts=related_concepts,
            concept_sentiments=concept_sentiments,
            news_count=sentiment_data.news_count,
            positive_count=sentiment_data.positive_count,
            negative_count=sentiment_data.negative_count,
            neutral_count=sentiment_data.neutral_count
        )
    
    def _empty_concept_sentiment(self, asset: str) -> ConceptSentiment:
        """Create empty ConceptSentiment"""
        return ConceptSentiment(
            asset=asset,
            timestamp=datetime.now(),
            sentiment_score=0.0,
            confidence=0.0
        )
    
    # ==================== FILTERS ====================
    
    def filter_by_concept(
        self,
        news_items: List[NewsItem],
        concepts: List[str]
    ) -> List[NewsItem]:
        """
        Filter news items that mention specific concepts.
        
        Args:
            news_items: News to filter
            concepts: List of concept IDs to filter by
            
        Returns:
            Filtered news items
        """
        filtered = []
        
        for news in news_items:
            text = f"{news.title} {news.description}"
            detected = self.detect_concepts_in_text(text)
            detected_ids = [c[0] for c in detected]
            
            if any(c in detected_ids for c in concepts):
                filtered.append(news)
        
        return filtered
    
    def get_sentiment_by_concept(
        self,
        asset: str,
        concept_id: str
    ) -> Dict[str, Any]:
        """
        Get sentiment specifically for a concept within an asset's news.
        
        Args:
            asset: Asset symbol
            concept_id: Concept ID to analyze
            
        Returns:
            Sentiment data for that concept
        """
        if self.sentiment_analyzer is None:
            return {}
        
        news = self.sentiment_analyzer.fetch_news(assets=[asset])
        enriched = self.enrich_news_with_concepts(news)
        
        # Filter by concept
        concept_news = [e for e in enriched if concept_id in e.concepts_detected]
        
        if not concept_news:
            return {
                'concept_id': concept_id,
                'asset': asset,
                'sentiment': 0.0,
                'count': 0,
                'confidence': 0.0
            }
        
        # Calculate sentiment for this concept
        sentiments = [n.sentiment for n in concept_news]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        return {
            'concept_id': concept_id,
            'asset': asset,
            'sentiment': avg_sentiment,
            'count': len(sentiments),
            'confidence': min(1.0, len(sentiments) / 5),
            'news_titles': [n.title for n in concept_news[:3]]
        }
    
    # ==================== COMBINED ANALYSIS ====================
    
    def get_comprehensive_analysis(
        self,
        asset: str,
        include_social: bool = True,
        include_geopolitical: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive sentiment + concept analysis.
        
        Args:
            asset: Asset symbol
            include_social: Include social sentiment
            include_geopolitical: Include geopolitical analysis
            
        Returns:
            Complete analysis dictionary
        """
        result = {
            'asset': asset,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Concept-aware sentiment
        concept_sentiment = self.analyze_asset_sentiment_with_concepts(asset)
        result['concept_sentiment'] = concept_sentiment.to_dict()
        
        # 2. Standard sentiment analysis
        if self.sentiment_analyzer:
            result['news_sentiment'] = self.sentiment_analyzer.analyze_asset_sentiment(asset).to_dict()
            
            if include_social:
                result['social_sentiment'] = self.sentiment_analyzer.get_social_sentiment(asset)
            
            if include_geopolitical:
                geo_impacts = self.sentiment_analyzer.get_geopolitical_impact([asset])
                result['geopolitical_impact'] = geo_impacts.get(asset, 0.0)
        
        # 3. Top concepts by sentiment
        if concept_sentiment.concept_sentiments:
            sorted_concepts = sorted(
                concept_sentiment.concept_sentiments.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            result['top_concepts'] = [
                {'id': c[0], 'sentiment': c[1]} 
                for c in sorted_concepts[:5]
            ]
        
        return result


# ==================== FACTORY FUNCTION ====================

def create_bridge(
    enable_semantic: bool = True,
    min_concept_score: float = 0.3
) -> Optional[SentimentConceptBridge]:
    """
    Factory function to create the bridge.
    
    Args:
        enable_semantic: Enable semantic features
        min_concept_score: Minimum concept match score
        
    Returns:
        SentimentConceptBridge or None if dependencies unavailable
    """
    if not SENTIMENT_AVAILABLE:
        logger.error("Cannot create bridge: sentiment_news not available")
        return None
    
    try:
        return SentimentConceptBridge(
            enable_semantic=enable_semantic,
            min_concept_score=min_concept_score
        )
    except Exception as e:
        logger.error(f"Failed to create bridge: {e}")
        return None


# ==================== STANDALONE USAGE ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("SENTIMENT-CONCEPT BRIDGE TEST")
    print("="*60)
    
    bridge = create_bridge()
    
    if bridge is None:
        print("❌ Failed to initialize bridge")
    else:
        # Test 1: Concept detection in text
        print("\n🔍 Testing concept detection:")
        test_text = "Bitcoin surge amid high volatility and strong bullish momentum"
        concepts = bridge.detect_concepts_in_text(test_text)
        print(f"   Text: '{test_text}'")
        print(f"   Concepts found: {concepts[:3]}")
        
        # Test 2: Asset sentiment with concepts
        print("\n📊 Testing concept-aware sentiment for BTC:")
        sentiment = bridge.analyze_asset_sentiment_with_concepts("BTC")
        print(f"   Sentiment Score: {sentiment.sentiment_score:.3f}")
        print(f"   Confidence: {sentiment.confidence:.2%}")
        print(f"   Concepts Detected: {sentiment.detected_concepts[:5]}")
        
        # Test 3: Comprehensive analysis
        print("\n📈 Comprehensive analysis for ETH:")
        comprehensive = bridge.get_comprehensive_analysis("ETH")
        print(f"   Combined sentiment: {comprehensive.get('concept_sentiment', {}).get('sentiment_score', 0):.3f}")
        
        if 'top_concepts' in comprehensive:
            print("   Top concepts:")
            for tc in comprehensive['top_concepts']:
                print(f"      - {tc['id']}: {tc['sentiment']:.3f}")
        
        print("\n✅ Bridge test complete!")
