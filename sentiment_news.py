"""
Sentiment and News Analysis Module
Collects and analyzes news and sentiment data for crypto and commodities
"""

import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import hashlib

import pandas as pd
import numpy as np

import config

# Configure logging
logger = logging.getLogger(__name__)

# Free alternative APIs
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
FEAR_GREED_URL = "https://api.alternative.me/fng/"


# ==================== DATA STRUCTURES ====================

@dataclass
class NewsItem:
    """Represents a news article"""
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    sentiment: float = 0.0  # -1 to 1
    relevance: float = 0.0   # 0 to 1
    category: str = "general"


@dataclass
class SentimentData:
    """Aggregated sentiment data for an asset"""
    asset: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float      # 0 to 1
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    keywords_matched: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'asset': self.asset,
            'timestamp': self.timestamp.isoformat(),
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'news_count': self.news_count,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'neutral_count': self.neutral_count,
            'sentiment_label': self.get_label()
        }
    
    def get_label(self) -> str:
        if self.sentiment_score > 0.2:
            return 'positive'
        elif self.sentiment_score < -0.2:
            return 'negative'
        return 'neutral'


@dataclass
class GeopoliticalEvent:
    """Represents a geopolitical event"""
    title: str
    description: str
    impact_assets: Dict[str, float]  # asset -> impact score
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    source: str


# ==================== SENTIMENT ANALYZER CLASS ====================

class SentimentAnalyzer:
    """
    Analyzes news and sentiment for crypto and commodity markets.
    Includes keyword-based sentiment and simulated social media sentiment.
    """
    
    def __init__(self, news_api_key: str = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            news_api_key: Optional API key for news service
        """
        self.news_api_key = news_api_key or config.NEWS_SETTINGS.get('news_api_key', '')
        self.news_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Keywords for different asset classes
        self.keywords = {
            'crypto': config.NEWS_SETTINGS.get('crypto_keywords', []),
            'gold': config.NEWS_SETTINGS.get('gold_keywords', []),
            'oil': config.NEWS_SETTINGS.get('oil_keywords', []),
            'forex': config.NEWS_SETTINGS.get('forex_keywords', []),
            'geopolitical': config.NEWS_SETTINGS.get('geopolitical_keywords', []),
        }
        
        # Sentiment keywords
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'gain', 'rise', 'up', 'growth',
            'profit', 'success', 'breakout', 'adoption', 'partnership',
            'innovation', 'launch', 'upgrade', 'positive', 'optimistic'
        ]
        
        self.negative_keywords = [
            'bearish', 'crash', 'drop', 'fall', 'decline', 'loss',
            'failure', 'hack', 'ban', 'restriction', 'regulation',
            'scandal', 'warning', 'risk', 'uncertainty', 'fear'
        ]
        
        logger.info("SentimentAnalyzer initialized")
    
    # ==================== NEWS FETCHING ====================
    
    def fetch_news(self, query: str = None, assets: List[str] = None,
                   hours: int = 24) -> List[NewsItem]:
        """
        Fetch news articles related to specified assets.
        
        Args:
            query: Search query (optional)
            assets: List of asset symbols to search for
            hours: Look back period in hours
            
        Returns:
            List of NewsItem objects
        """
        # In simulation mode, generate simulated news
        if not self.news_api_key or config.SIMULATION_MODE:
            return self._generate_simulated_news(assets, hours)
        
        # Real API implementation would go here
        return self._fetch_real_news(query, assets, hours)
    
    def _fetch_real_news(self, query: str, assets: List[str], 
                        hours: int) -> List[NewsItem]:
        """Fetch real news from NewsAPI"""
        import requests
        from datetime import datetime, timedelta
        
        news_items = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        
        # Search for each asset
        for asset in assets:
            try:
                # Use the query parameter or asset name
                search_term = query if query else asset
                
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': search_term,
                    'from': start_date.isoformat(),
                    'to': end_date.isoformat(),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'apiKey': self.news_api_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        news_item = NewsItem(
                            title=article.get('title', ''),
                            description=article.get('description', ''),
                            source=article.get('source', {}).get('name', 'Unknown'),
                            url=article.get('url', ''),
                            published_at=article.get('publishedAt', ''),
                            sentiment_score=self._analyze_sentiment(
                                article.get('title', '') + ' ' + article.get('description', '')
                            ),
                            asset=asset
                        )
                        news_items.append(news_item)
                        
                elif response.status_code == 401:
                    logger.error("NewsAPI: Invalid API key")
                elif response.status_code == 429:
                    logger.error("NewsAPI: Rate limit exceeded")
                    
            except Exception as e:
                logger.warning(f"Error fetching news for {asset}: {e}")
        
        return news_items if news_items else self._generate_simulated_news(assets, hours)
    
    def _generate_simulated_news(self, assets: List[str], 
                                 hours: int) -> List[NewsItem]:
        """Generate simulated news for testing"""
        news_items = []
        
        # Sample news templates
        templates = [
            {
                'title': '{asset} {action} amid market volatility',
                'description': 'Traders are closely watching {asset} as market conditions shift. Analysts suggest {action} could continue in the near term.',
                'category': 'market'
            },
            {
                'title': 'New adoption milestone for {asset}',
                'description': 'Major institution announces support for {asset}, marking significant mainstream adoption.',
                'category': 'adoption'
            },
            {
                'title': 'Regulatory update affects {asset}',
                'description': 'New regulatory framework announced, potentially impacting {asset} trading in multiple jurisdictions.',
                'category': 'regulation'
            },
            {
                'title': 'Technical analysis: {asset} {direction}',
                'description': 'Chart patterns suggest {direction} momentum for {asset} with key resistance levels identified.',
                'category': 'analysis'
            },
            {
                'title': '{asset} price prediction from analyst',
                'description': 'Leading analyst projects {direction} movement for {asset} citing fundamental factors.',
                'category': 'analysis'
            },
            {
                'title': 'Market sentiment shifts for {asset}',
                'description': 'Social media sentiment shows {direction} bias for {asset} among retail investors.',
                'category': 'sentiment'
            },
        ]
        
        actions = ['fluctuating', 'stabilizing', 'gaining attention', 'showing resilience']
        directions = ['bullish', 'bearish', 'upward', 'downward']
        
        # Generate news for each asset
        num_articles = random.randint(3, 8)
        
        for _ in range(num_articles):
            if assets:
                asset = random.choice(assets)
            else:
                asset = random.choice(['Bitcoin', 'Ethereum', 'Gold', 'Oil'])
            
            template = random.choice(templates)
            action = random.choice(actions)
            direction = random.choice(directions)
            
            # Add some randomness to make different articles
            variation = random.randint(1000, 9999)
            
            news = NewsItem(
                title=template['title'].format(asset=asset, action=action, direction=direction),
                description=template['description'].format(asset=asset, action=action, direction=direction),
                source=random.choice(config.NEWS_SETTINGS.get('preferred_sources', 
                    ['Reuters', 'CoinDesk', 'Bloomberg'])),
                url=f"https://example.com/news/{variation}",
                published_at=datetime.now() - timedelta(
                    minutes=random.randint(0, hours * 60)
                ),
                category=template['category']
            )
            
            # Calculate sentiment
            news.sentiment = self._analyze_text_sentiment(news.title + " " + news.description)
            news.relevance = random.uniform(0.5, 1.0)
            
            news_items.append(news)
        
        # Sort by publication date
        news_items.sort(key=lambda x: x.published_at, reverse=True)
        
        return news_items

    # ==================== FREE API METHODS ====================
    
    def fetch_fear_greed_index(self) -> Dict:
        """
        Fetch real Fear & Greed Index from Alternative.me API.
        This is a free API that doesn't require authentication.
        
        Returns:
            Dict with fear_greed_index value and classification
        """
        try:
            import requests
            response = requests.get(FEAR_GREED_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    fgi = data['data'][0]
                    return {
                        'value': int(fgi.get('value', 50)),
                        'value_classification': fgi.get('value_classification', 'Neutral'),
                        'timestamp': fgi.get('time_until_update', '')
                    }
        except Exception as e:
            logger.warning(f"Error fetching Fear & Greed Index: {e}")
        
        # Fallback to simulated data
        return {
            'value': random.randint(20, 80),
            'value_classification': 'Neutral',
            'timestamp': datetime.now().isoformat()
        }

    def fetch_cryptopanic_news(self, filter_kind: str = 'hot') -> List[Dict]:
        """
        Fetch news from CryptoPanic aggregator (free, no key required).
        
        Args:
            filter_kind: Filter type - 'hot', 'latest', 'rising', 'bullish', 'bearish'
        
        Returns:
            List of news items
        """
        try:
            import requests
            params = {'filter': filter_kind, 'auth_token': 'public'}
            response = requests.get(CRYPTOPANIC_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                return [
                    {
                        'title': item.get('title', ''),
                        'domain': item.get('domain', ''),
                        'published_at': item.get('published_at', ''),
                        'votes': item.get('votes', {}).get('positive', 0)
                    }
                    for item in results[:20]
                ]
        except Exception as e:
            logger.warning(f"Error fetching CryptoPanic news: {e}")
        
        return []

    def fetch_coingecko_market_data(self, vs_currency: str = 'usd', 
                                   per_page: int = 100) -> List[Dict]:
        """
        Fetch market data from CoinGecko API (free, rate limited).
        
        Args:
            vs_currency: Currency to compare against (usd, eur, etc.)
            per_page: Number of coins to fetch
        
        Returns:
            List of market data for top coins
        """
        try:
            import requests
            url = f"{COINGECKO_API_URL}/coins/markets"
            params = {
                'vs_currency': vs_currency,
                'order': 'market_cap_desc',
                'per_page': per_page,
                'page': 1,
                'sparkline': False
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Error fetching CoinGecko data: {e}")
        
        return []

    # ==================== SENTIMENT ANALYSIS ====================
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using keyword matching.
        
        Returns:
            Sentiment score from -1 (very negative) to 1 (very positive)
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return 0.0
        
        # Calculate sentiment (-1 to 1)
        sentiment = (positive_count - negative_count) / total
        
        # Normalize to -1 to 1 range
        return max(-1, min(1, sentiment))
    
    def analyze_asset_sentiment(self, asset: str, news: List[NewsItem] = None) -> SentimentData:
        """
        Analyze sentiment for a specific asset.
        
        Args:
            asset: Asset symbol/name
            news: Optional pre-fetched news
            
        Returns:
            SentimentData object
        """
        if news is None:
            news = self.fetch_news(assets=[asset])
        
        # Filter news relevant to this asset
        relevant_news = self._filter_relevant_news(news, asset)
        
        if not relevant_news:
            return SentimentData(
                asset=asset,
                timestamp=datetime.now(),
                sentiment_score=0.0,
                confidence=0.0,
                news_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0
            )
        
        # Calculate sentiment scores
        sentiments = [n.sentiment for n in relevant_news]
        weights = [n.relevance for n in relevant_news]
        
        # Weighted average sentiment
        if sum(weights) > 0:
            weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
        else:
            weighted_sentiment = sum(sentiments) / len(sentiments)
        
        # Count categories
        positive_count = sum(1 for s in sentiments if s > 0.2)
        negative_count = sum(1 for s in sentiments if s < -0.2)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Calculate confidence based on news count
        confidence = min(1.0, len(relevant_news) / 10)
        
        # Get matched keywords
        keywords = self._extract_keywords(relevant_news)
        
        # Get sources
        sources = list(set(n.source for n in relevant_news))
        
        return SentimentData(
            asset=asset,
            timestamp=datetime.now(),
            sentiment_score=weighted_sentiment,
            confidence=confidence,
            news_count=len(relevant_news),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            keywords_matched=keywords[:5],
            sources=sources
        )
    
    def analyze_multiple_assets(self, assets: List[str]) -> Dict[str, SentimentData]:
        """
        Analyze sentiment for multiple assets.
        
        Args:
            assets: List of asset symbols/names
            
        Returns:
            Dictionary of asset -> SentimentData
        """
        results = {}
        
        # Fetch all news once
        all_news = self.fetch_news(assets=assets)
        
        for asset in assets:
            asset_sentiment = self.analyze_asset_sentiment(asset, all_news)
            results[asset] = asset_sentiment
        
        return results
    
    # ==================== HELPER METHODS ====================
    
    def _filter_relevant_news(self, news: List[NewsItem], asset: str) -> List[NewsItem]:
        """Filter news items relevant to a specific asset"""
        asset_lower = asset.lower().replace('/', ' ').replace('usdt', '').strip()
        
        relevant = []
        
        for item in news:
            # Check if asset mentioned in title or description
            text = (item.title + " " + item.description).lower()
            
            # Simple matching
            if asset_lower in text:
                relevant.append(item)
            elif any(kw in text for kw in self.keywords.get(asset_lower, [])):
                relevant.append(item)
        
        return relevant
    
    def _extract_keywords(self, news: List[NewsItem]) -> List[str]:
        """Extract mentioned keywords from news"""
        all_keywords = set()
        
        # Asset-specific keywords
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                for item in news:
                    if keyword in (item.title + " " + item.description).lower():
                        all_keywords.add(keyword)
        
        return list(all_keywords)
    
    # ==================== SOCIAL SENTIMENT ====================
    
    def get_social_sentiment(self, asset: str) -> Dict[str, Any]:
        """
        Get simulated social media sentiment for an asset.
        
        Returns:
            Dictionary with social metrics
        """
        # Simulated social metrics
        base_sentiment = random.uniform(-0.3, 0.3)
        
        # Generate realistic-looking metrics
        total_mentions = random.randint(1000, 50000)
        positive_mentions = int(total_mentions * (0.4 + base_sentiment * 0.3))
        negative_mentions = int(total_mentions * (0.3 - base_sentiment * 0.2))
        neutral_mentions = total_mentions - positive_mentions - negative_mentions
        
        return {
            'asset': asset,
            'timestamp': datetime.now(),
            'total_mentions': total_mentions,
            'positive_mentions': positive_mentions,
            'negative_mentions': negative_mentions,
            'neutral_mentions': neutral_mentions,
            'sentiment_ratio': (positive_mentions - negative_mentions) / total_mentions,
            'dominance': 'bullish' if base_sentiment > 0.1 else 'bearish' if base_sentiment < -0.1 else 'neutral',
            'fear_greed_index': random.randint(20, 80)  # 0 = extreme fear, 100 = extreme greed
        }

    def fetch_fear_greed_index(self) -> Dict:
        """
        Fetch real Fear & Greed Index from Alternative.me API.
        This is a free API that doesn't require authentication.
        
        Returns:
            Dict with fear_greed_index value and classification
        """
        try:
            import requests
            response = requests.get(FEAR_GREED_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    fgi = data['data'][0]
                    return {
                        'value': int(fgi.get('value', 50)),
                        'value_classification': fgi.get('value_classification', 'Neutral'),
                        'timestamp': fgi.get('time_until_update', '')
                    }
        except Exception as e:
            logger.warning(f"Error fetching Fear & Greed Index: {e}")
        
        # Fallback to simulated data
        return {
            'value': random.randint(20, 80),
            'value_classification': 'Neutral',
            'timestamp': datetime.now().isoformat()
        }

    def fetch_cryptopanic_news(self, filter_kind: str = 'hot') -> List[Dict]:
        """
        Fetch news from CryptoPanic aggregator (free, no key required).
        
        Args:
            filter_kind: Filter type - 'hot', 'latest', 'rising', 'bullish', 'bearish'
        
        Returns:
            List of news items
        """
        try:
            import requests
            params = {'filter': filter_kind, 'auth_token': 'public'}
            response = requests.get(CRYPTOPANIC_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                return [
                    {
                        'title': item.get('title', ''),
                        'domain': item.get('domain', ''),
                        'published_at': item.get('published_at', ''),
                        'votes': item.get('votes', {}).get('positive', 0)
                    }
                    for item in results[:20]
                ]
        except Exception as e:
            logger.warning(f"Error fetching CryptoPanic news: {e}")
        
        return []

    def fetch_coingecko_market_data(self, vs_currency: str = 'usd', 
                                   per_page: int = 100) -> List[Dict]:
        """
        Fetch market data from CoinGecko API (free, rate limited).
        
        Args:
            vs_currency: Currency to compare against (usd, eur, etc.)
            per_page: Number of coins to fetch
        
        Returns:
            List of market data for top coins
        """
        try:
            import requests
            url = f"{COINGECKO_API_URL}/coins/markets"
            params = {
                'vs_currency': vs_currency,
                'order': 'market_cap_desc',
                'per_page': per_page,
                'page': 1,
                'sparkline': False
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Error fetching CoinGecko data: {e}")
        
        return []

    # ==================== GEOPOLITICAL ANALYSIS ====================
    
    def analyze_geopolitical_events(self) -> List[GeopoliticalEvent]:
        """
        Analyze current geopolitical situation and its potential impact.
        
        Returns:
            List of GeopoliticalEvent objects
        """
        # Simulated geopolitical events
        events = [
            GeopoliticalEvent(
                title="Fed Interest Rate Decision",
                description="Federal Reserve announces interest rate decision affecting global markets",
                impact_assets={
                    'BTC': -0.1, 'ETH': -0.1, 'PAXG': 0.3, 'USDT': 0.0
                },
                severity='high',
                timestamp=datetime.now(),
                source='Economic Calendar'
            ),
            GeopoliticalEvent(
                title="European Regulatory Discussion",
                description="New crypto regulations under discussion in EU parliament",
                impact_assets={
                    'BTC': -0.2, 'ETH': -0.15, 'XRP': -0.1
                },
                severity='medium',
                timestamp=datetime.now() - timedelta(hours=6),
                source='Reuters'
            ),
            GeopoliticalEvent(
                title="Middle East Tensions",
                description="Geopolitical tensions in Middle East affecting commodity prices",
                impact_assets={
                    'PAXG': 0.25, 'XAUT': 0.25, 'BTC': 0.05
                },
                severity='medium',
                timestamp=datetime.now() - timedelta(hours=12),
                source='Bloomberg'
            ),
        ]
        
        return events
    
    def get_geopolitical_impact(self, assets: List[str]) -> Dict[str, float]:
        """
        Calculate overall geopolitical impact on specified assets.
        
        Args:
            assets: List of asset symbols
            
        Returns:
            Dictionary of asset -> impact score (-1 to 1)
        """
        events = self.analyze_geopolitical_events()
        
        impacts = {asset: 0.0 for asset in assets}
        
        for event in events:
            for asset, impact in event.impact_assets.items():
                if asset in assets:
                    impacts[asset] += impact * self._severity_weight(event.severity)
        
        # Normalize
        for asset in impacts:
            impacts[asset] = max(-1, min(1, impacts[asset]))
        
        return impacts
    
    def _severity_weight(self, severity: str) -> float:
        """Convert severity to weight"""
        weights = {
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75,
            'critical': 1.0
        }
        return weights.get(severity, 0.5)
    
    # ==================== COMBINED ANALYSIS ====================
    
    def get_combined_sentiment(self, asset: str) -> Dict[str, Any]:
        """
        Get combined sentiment from news + social + geopolitical.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Dictionary with combined sentiment data
        """
        # Get all sentiment sources
        news_sentiment = self.analyze_asset_sentiment(asset)
        social_sentiment = self.get_social_sentiment(asset)
        
        # Calculate combined score
        news_weight = 0.5
        social_weight = 0.3
        geo_weight = 0.2
        
        geo_impacts = self.get_geopolitical_impact([asset])
        geo_score = geo_impacts.get(asset, 0.0)
        
        combined_score = (
            news_sentiment.sentiment_score * news_weight +
            social_sentiment['sentiment_ratio'] * social_weight +
            geo_score * geo_weight
        )
        
        return {
            'asset': asset,
            'timestamp': datetime.now(),
            'combined_score': combined_score,
            'news_sentiment': news_sentiment.to_dict(),
            'social_sentiment': {
                'dominance': social_sentiment['dominance'],
                'fear_greed_index': social_sentiment['fear_greed_index'],
                'total_mentions': social_sentiment['total_mentions']
            },
            'geopolitical_impact': geo_score,
            'confidence': (news_sentiment.confidence + 
                          (social_sentiment['total_mentions'] / 50000)) / 2
        }
    
    # ==================== UTILITY METHODS ====================
    
    def export_sentiment(self, sentiment: SentimentData, filepath: str):
        """Export sentiment data to JSON"""
        with open(filepath, 'w') as f:
            json.dump(sentiment.to_dict(), f, indent=2, default=str)
        logger.info(f"Sentiment data exported to {filepath}")


# ==================== STANDALONE FUNCTIONS ====================

def get_sentiment(asset: str) -> SentimentData:
    """Quick sentiment analysis for an asset"""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_asset_sentiment(asset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYZER TEST")
    print("="*60)
    
    analyzer = SentimentAnalyzer()
    
    # Test single asset
    print("\n📰 Analyzing Bitcoin sentiment...")
    sentiment = analyzer.analyze_asset_sentiment('Bitcoin')
    print(f"   Score: {sentiment.sentiment_score:.2f}")
    print(f"   Label: {sentiment.get_label()}")
    print(f"   News: {sentiment.news_count} articles")
    print(f"   Confidence: {sentiment.confidence:.2%}")
    
    # Test social sentiment
    print("\n📱 Social sentiment for BTC:")
    social = analyzer.get_social_sentiment('BTC')
    print(f"   Dominance: {social['dominance']}")
    print(f"   Fear/Greed Index: {social['fear_greed_index']}")
    print(f"   Total mentions: {social['total_mentions']:,}")
    
    # Test geopolitical
    print("\n🌍 Geopolitical impact:")
    geo_impacts = analyzer.get_geopolitical_impact(['BTC', 'ETH', 'PAXG'])
    for asset, impact in geo_impacts.items():
        print(f"   {asset}: {impact:+.2f}")
    
    # Combined analysis
    print("\n🔄 Combined sentiment for ETH:")
    combined = analyzer.get_combined_sentiment('ETH')
    print(f"   Combined Score: {combined['combined_score']:.2f}")
    print(f"   Confidence: {combined['confidence']:.2%}")
    
    print("\n✅ Test complete!")

