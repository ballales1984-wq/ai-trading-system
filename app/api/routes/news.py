"""
News Feed Routes
================
REST API for crypto news feed and sentiment analysis.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

from app.api.mock_data import get_news as mock_get_news

# Import demo mode functions from portfolio (they share the same state)
from app.core.demo_mode import get_demo_mode

# AI Integration
try:
    from sentiment_concept_bridge import create_bridge
    news_bridge = create_bridge()
except ImportError:
    news_bridge = None

router = APIRouter()


# ============================================================================
# DATA MODELS
# ============================================================================

class NewsItem(BaseModel):
    """News item model."""
    id: str = Field(..., description="Unique news identifier")
    title: str = Field(..., description="News headline")
    source: str = Field(..., description="News source (e.g., CoinDesk)")
    url: str = Field(..., description="URL to original article")
    summary: str = Field(..., description="Brief summary of the article")
    sentiment: str = Field(..., description="Sentiment: positive, negative, neutral")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1 to 1")
    symbols: List[str] = Field(default=[], description="Related trading symbols")
    published_at: datetime = Field(..., description="Publication timestamp")
    category: str = Field(..., description="News category: market, technology, regulation, etc.")


class NewsListResponse(BaseModel):
    """Response model for news list."""
    news: List[NewsItem]
    total: int
    last_updated: datetime


class NewsBySymbolResponse(BaseModel):
    """Response model for news filtered by symbol."""
    symbol: str
    news: List[NewsItem]
    total: int
    last_updated: datetime


# ============================================================================
# ROUTES
# ============================================================================

@router.get("", response_model=NewsListResponse)
async def get_news(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of news items to return"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment: positive, negative, neutral"),
    category: Optional[str] = Query(None, description="Filter by category: market, technology, regulation, defi, exchange, macro, network"),
    refresh: Optional[str] = Query(None, description="Force refresh with new data (use 'true' or timestamp)"),
) -> NewsListResponse:
    """
    Get latest crypto news feed.
    
    Returns a list of news items with sentiment analysis and related symbols.
    Supports filtering by sentiment and category.
    Use refresh parameter to get fresh data with varied news.
    """
    # Production mode: try to fetch from real news API using the Bridge
    if news_bridge and not get_demo_mode():
        try:
            # Use 'BTC' as default if no filter provided, or get general market news
            asset_to_query = "BTC"  # Default for general news
            analysis = news_bridge.get_comprehensive_analysis(asset_to_query)
            
            # Map Bridge result to NewsItem models
            # Bridge provides 'news_sentiment' with 'news_count', etc.
            # To get list of news, we might need to call fetch_news from bridge's analyzer
            if bridge_analyzer := getattr(news_bridge, 'sentiment_analyzer', None):
                raw_news = bridge_analyzer.fetch_news(assets=[asset_to_query], limit=limit, force_real=True)
                news_items = []
                for n in raw_news:
                    # Detect concepts for this specific item for extra enrichment
                    concepts = news_bridge.detect_concepts_in_text(f"{n.title} {n.description}")
                    
                    news_items.append(NewsItem(
                        id=str(hash(n.url)),
                        title=n.title,
                        source=n.source,
                        url=n.url,
                        summary=n.description[:200] + "..." if len(n.description) > 200 else n.description,
                        sentiment="positive" if n.sentiment > 0.1 else ("negative" if n.sentiment < -0.1 else "neutral"),
                        sentiment_score=n.sentiment,
                        symbols=[asset_to_query],
                        published_at=n.published_at,
                        category=n.category or "market"
                    ))
                
                return NewsListResponse(
                    news=news_items,
                    total=len(news_items),
                    last_updated=datetime.utcnow()
                )
        except Exception as e:
            from fastapi import logger as fast_logger
            fast_logger.warning(f"AI News Bridge failed: {e}. Falling back to mock.")

    # Fallback to mock data
    news_data = mock_get_news(limit=limit, refresh=refresh)
    
    # Apply sentiment filter if provided
    if sentiment:
        sentiment_lower = sentiment.lower()
        news_data = [n for n in news_data if n["sentiment"] == sentiment_lower]
    
    # Apply category filter if provided
    if category:
        category_lower = category.lower()
        news_data = [n for n in news_data if n["category"] == category_lower]
    
    # Convert to Pydantic models
    news_items = [NewsItem(**n) for n in news_data]
    
    return NewsListResponse(
        news=news_items,
        total=len(news_items),
        last_updated=datetime.utcnow()
    )


@router.get("/{symbol}", response_model=NewsBySymbolResponse)
async def get_news_by_symbol(
    symbol: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of news items to return"),
    refresh: Optional[str] = Query(None, description="Force refresh with new data"),
) -> NewsBySymbolResponse:
    """
    Get news filtered by trading symbol.
    
    Returns news items related to a specific cryptocurrency (e.g., BTCUSDT, ETHUSDT).
    """
    # Normalize symbol
    symbol_upper = symbol.upper()
    asset_to_query = symbol_upper.replace("USDT", "")
    
    # Production mode: try to fetch from real news API using the Bridge
    if news_bridge and not get_demo_mode():
        try:
            if bridge_analyzer := getattr(news_bridge, 'sentiment_analyzer', None):
                raw_news = bridge_analyzer.fetch_news(assets=[asset_to_query], limit=limit, force_real=True)
                news_items = []
                for n in raw_news:
                    news_items.append(NewsItem(
                        id=str(hash(n.url)),
                        title=n.title,
                        source=n.source,
                        url=n.url,
                        summary=n.description[:200] + "..." if len(n.description) > 200 else n.description,
                        sentiment="positive" if n.sentiment > 0.1 else ("negative" if n.sentiment < -0.1 else "neutral"),
                        sentiment_score=n.sentiment,
                        symbols=[symbol_upper],
                        published_at=n.published_at,
                        category=n.category or "market"
                    ))
                
                return NewsBySymbolResponse(
                    symbol=symbol_upper,
                    news=news_items,
                    total=len(news_items),
                    last_updated=datetime.utcnow()
                )
        except Exception as e:
            from fastapi import logger as fast_logger
            fast_logger.warning(f"AI News Bridge (symbol) failed: {e}. Falling back to mock.")

    # Fallback to mock data
    news_data = mock_get_news(symbol=symbol_upper, limit=limit, refresh=refresh)
    news_items = [NewsItem(**n) for n in news_data]
    
    return NewsBySymbolResponse(
        symbol=symbol_upper,
        news=news_items,
        total=len(news_items),
        last_updated=datetime.utcnow()
    )


@router.get("/sentiment/overview", response_model=dict)
async def get_sentiment_overview() -> dict:
    """
    Get overall market sentiment based on recent news.
    
    Returns aggregated sentiment statistics from the latest news items.
    """
    # Use mock data if demo mode is enabled
    if get_demo_mode():
        news_data = mock_get_news(limit=20)
        
        # Calculate sentiment statistics
        total = len(news_data)
        positive = sum(1 for n in news_data if n["sentiment"] == "positive")
        negative = sum(1 for n in news_data if n["sentiment"] == "negative")
        neutral = sum(1 for n in news_data if n["sentiment"] == "neutral")
        
        # Calculate average sentiment score
        avg_score = sum(n["sentiment_score"] for n in news_data) / total if total > 0 else 0
        
        # Determine overall sentiment
        if avg_score > 0.3:
            overall = "Bullish"
        elif avg_score < -0.3:
            overall = "Bearish"
        else:
            overall = "Neutral"
        
        return {
            "overall_sentiment": overall,
            "average_score": round(avg_score, 2),
            "distribution": {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "total": total
            },
            "percentages": {
                "positive": round(positive / total * 100, 1) if total > 0 else 0,
                "negative": round(negative / total * 100, 1) if total > 0 else 0,
                "neutral": round(neutral / total * 100, 1) if total > 0 else 0,
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # Production mode: try to calculate from real news data using the Bridge
    if news_bridge and not get_demo_mode():
        try:
            # Use 'BTC' as general market proxy for overall sentiment
            analysis = news_bridge.get_comprehensive_analysis("BTC")
            concept_sent = analysis.get('concept_sentiment', {})
            
            return {
                "overall_sentiment": concept_sent.get('sentiment_label', 'Neutral').capitalize(),
                "average_score": concept_sent.get('sentiment_score', 0.0),
                "distribution": {
                    "positive": concept_sent.get('positive_count', 0),
                    "negative": concept_sent.get('negative_count', 0),
                    "neutral": concept_sent.get('neutral_count', 0),
                    "total": concept_sent.get('news_count', 0)
                },
                "percentages": {
                    "positive": round(concept_sent.get('positive_count', 0) / concept_sent.get('news_count', 1) * 100, 1) if concept_sent.get('news_count', 0) > 0 else 0,
                    "negative": round(concept_sent.get('negative_count', 0) / concept_sent.get('news_count', 1) * 100, 1) if concept_sent.get('news_count', 0) > 0 else 0,
                    "neutral": round(concept_sent.get('neutral_count', 0) / concept_sent.get('news_count', 1) * 100, 1) if concept_sent.get('news_count', 0) > 0 else 0,
                },
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            from fastapi import logger as fast_logger
            fast_logger.warning(f"AI Sentiment Overview failed: {e}. Falling back to default.")

    # Production mode: fallback or default
    return {
        "overall_sentiment": "Neutral",
        "average_score": 0.0,
        "distribution": {"positive": 0, "negative": 0, "neutral": 0, "total": 0},
        "percentages": {"positive": 0, "negative": 0, "neutral": 0},
        "last_updated": datetime.utcnow().isoformat()
    }
