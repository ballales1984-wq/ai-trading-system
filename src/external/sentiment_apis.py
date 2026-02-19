"""
Sentiment & News API Clients
==============================
Clients for: NewsAPI.org, Benzinga, Twitter/X, GDELT.

These provide news articles and sentiment scores that feed into
the Sentiment Analyzer and Monte Carlo Level 2 (Conditional).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

from .api_registry import (
    APICategory,
    BaseAPIClient,
    DataQuality,
    NormalizedRecord,
    RateLimitConfig,
)


# ---------------------------------------------------------------------------
# NewsAPI.org Client
# ---------------------------------------------------------------------------

class NewsAPIClient(BaseAPIClient):
    """
    NewsAPI.org — general news filtered by asset/keyword.
    Docs: https://newsapi.org/docs
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="newsapi",
            category=APICategory.SENTIMENT,
            api_key=api_key,
            base_url="https://newsapi.org/v2",
            rate_limit=RateLimitConfig(max_requests_per_minute=100, max_requests_per_second=2),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        query = kwargs.get("query", "bitcoin")
        limit = kwargs.get("limit", 50)
        sort_by = kwargs.get("sort_by", "publishedAt")

        url = f"{self.base_url}/everything"
        params = {
            "q": query,
            "sortBy": sort_by,
            "pageSize": min(limit, 100),
            "apiKey": self.api_key,
            "language": "en",
        }

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"NewsAPI HTTP {resp.status}")
                data = await resp.json()

        for article in data.get("articles", []):
            published = article.get("publishedAt", "")
            try:
                ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(timezone.utc)

            # Basic sentiment heuristic (placeholder — replace with NLP model)
            title = article.get("title", "") or ""
            description = article.get("description", "") or ""
            sentiment_score = self._basic_sentiment(title + " " + description)

            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.SENTIMENT,
                    source_api="newsapi",
                    data_type="news_sentiment",
                    payload={
                        "title": title,
                        "description": description,
                        "source": article.get("source", {}).get("name", ""),
                        "url": article.get("url", ""),
                        "sentiment_score": sentiment_score,
                    },
                    quality=DataQuality.MEDIUM,
                    confidence=0.6,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            url = f"{self.base_url}/top-headlines"
            params = {"country": "us", "pageSize": 1, "apiKey": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _basic_sentiment(text: str) -> float:
        """Very basic keyword sentiment (placeholder for real NLP)."""
        positive = ["surge", "rally", "gain", "bull", "up", "profit", "growth", "record"]
        negative = ["crash", "drop", "fall", "bear", "down", "loss", "decline", "fear"]
        text_lower = text.lower()
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total


# ---------------------------------------------------------------------------
# Benzinga API Client
# ---------------------------------------------------------------------------

class BenzingaClient(BaseAPIClient):
    """
    Benzinga API — financial news with event impact.
    Docs: https://docs.benzinga.io/
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="benzinga",
            category=APICategory.SENTIMENT,
            api_key=api_key,
            base_url="https://api.benzinga.com/api/v2",
            rate_limit=RateLimitConfig(max_requests_per_minute=30, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 50)

        url = f"{self.base_url}/news"
        params: Dict[str, Any] = {
            "token": self.api_key,
            "pageSize": min(limit, 100),
        }
        if query:
            params["tickers"] = query.upper()

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Benzinga HTTP {resp.status}")
                data = await resp.json()

        for item in data if isinstance(data, list) else []:
            created = item.get("created", "")
            try:
                ts = datetime.fromisoformat(created)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                ts = datetime.now(timezone.utc)

            title = item.get("title", "")
            body = item.get("body", "")
            sentiment_score = NewsAPIClient._basic_sentiment(title + " " + body)

            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.SENTIMENT,
                    source_api="benzinga",
                    data_type="news_sentiment",
                    payload={
                        "title": title,
                        "tickers": item.get("stocks", []),
                        "channels": item.get("channels", []),
                        "sentiment_score": sentiment_score,
                        "url": item.get("url", ""),
                    },
                    quality=DataQuality.HIGH,
                    confidence=0.7,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            url = f"{self.base_url}/news"
            params = {"token": self.api_key, "pageSize": 1}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Twitter / X API Client
# ---------------------------------------------------------------------------

class TwitterSentimentClient(BaseAPIClient):
    """
    Twitter/X API v2 — social sentiment and trending topics.
    Docs: https://developer.twitter.com/en/docs/twitter-api
    """

    def __init__(self, bearer_token: str = ""):
        super().__init__(
            name="twitter",
            category=APICategory.SENTIMENT,
            api_key=bearer_token,
            base_url="https://api.twitter.com/2",
            rate_limit=RateLimitConfig(max_requests_per_minute=15, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        query = kwargs.get("query", "bitcoin")
        limit = kwargs.get("limit", 50)

        url = f"{self.base_url}/tweets/search/recent"
        params = {
            "query": f"{query} lang:en -is:retweet",
            "max_results": min(max(limit, 10), 100),
            "tweet.fields": "created_at,public_metrics,lang",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Twitter HTTP {resp.status}")
                data = await resp.json()

        for tweet in data.get("data", []):
            created = tweet.get("created_at", "")
            try:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(timezone.utc)

            text = tweet.get("text", "")
            metrics = tweet.get("public_metrics", {})
            sentiment_score = NewsAPIClient._basic_sentiment(text)

            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.SENTIMENT,
                    source_api="twitter",
                    data_type="social_sentiment",
                    payload={
                        "text": text,
                        "sentiment_score": sentiment_score,
                        "retweet_count": metrics.get("retweet_count", 0),
                        "like_count": metrics.get("like_count", 0),
                        "reply_count": metrics.get("reply_count", 0),
                    },
                    quality=DataQuality.LOW,
                    confidence=0.4,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            url = f"{self.base_url}/tweets/search/recent"
            params = {"query": "test", "max_results": 10}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# GDELT API Client (Sentiment + Geopolitics)
# ---------------------------------------------------------------------------

class GDELTClient(BaseAPIClient):
    """
    GDELT API — global events, sentiment, and geopolitical data.
    Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    """

    def __init__(self):
        super().__init__(
            name="gdelt",
            category=APICategory.SENTIMENT,
            api_key="",  # GDELT is free, no key needed
            base_url="https://api.gdeltproject.org/api/v2",
            rate_limit=RateLimitConfig(max_requests_per_minute=60, max_requests_per_second=2),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        query = kwargs.get("query", "bitcoin")
        limit = kwargs.get("limit", 50)
        mode = kwargs.get("mode", "ArtList")  # ArtList, TimelineVol, ToneChart

        url = f"{self.base_url}/doc/doc"
        params = {
            "query": query,
            "mode": mode,
            "maxrecords": min(limit, 250),
            "format": "json",
            "sort": "DateDesc",
        }

        records: List[NormalizedRecord] = []
        if aiohttp is None:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"GDELT HTTP {resp.status}")
                data = await resp.json()

        for article in data.get("articles", []):
            date_str = article.get("seendate", "")
            try:
                ts = datetime.strptime(date_str[:14], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)

            tone = article.get("tone", 0.0)
            # GDELT tone: positive = positive sentiment, negative = negative
            sentiment_score = max(-1.0, min(1.0, tone / 10.0))

            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.SENTIMENT,
                    source_api="gdelt",
                    data_type="geopolitical_sentiment",
                    payload={
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "domain": article.get("domain", ""),
                        "language": article.get("language", ""),
                        "tone": tone,
                        "sentiment_score": sentiment_score,
                        "socialimage": article.get("socialimage", ""),
                    },
                    quality=DataQuality.MEDIUM,
                    confidence=0.5,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None:
            return False
        try:
            url = f"{self.base_url}/doc/doc"
            params = {"query": "test", "mode": "ArtList", "maxrecords": 1, "format": "json"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_sentiment_clients(
    newsapi_key: str = "",
    benzinga_key: str = "",
    twitter_bearer: str = "",
    include_gdelt: bool = True,
) -> List[BaseAPIClient]:
    """Create all sentiment clients from provided keys."""
    clients: List[BaseAPIClient] = []
    if newsapi_key:
        clients.append(NewsAPIClient(api_key=newsapi_key))
    if benzinga_key:
        clients.append(BenzingaClient(api_key=benzinga_key))
    if twitter_bearer:
        clients.append(TwitterSentimentClient(bearer_token=twitter_bearer))
    if include_gdelt:
        clients.append(GDELTClient())
    return clients


def create_sentiment_clients_from_env() -> List[BaseAPIClient]:
    """Create sentiment clients using environment variables."""
    return create_sentiment_clients(
        newsapi_key=os.getenv("NEWSAPI_KEY", ""),
        benzinga_key=os.getenv("BENZINGA_API_KEY", ""),
        twitter_bearer=os.getenv("TWITTER_BEARER_TOKEN", ""),
        include_gdelt=True,
    )
