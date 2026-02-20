"""
API Registry — Central API Factory & Dispatcher
=================================================
Manages all external API clients, handles rate limiting,
normalization, and provides a unified interface for the engine.

Usage:
    registry = APIRegistry()
    registry.configure_from_env()
    prices = await registry.fetch_market_data("BTCUSDT", "1h", limit=100)
    sentiment = await registry.fetch_sentiment("bitcoin")
    events = await registry.fetch_macro_events(region="US")
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class APICategory(Enum):
    MARKET_DATA = "market_data"
    SENTIMENT = "sentiment"
    MACRO_EVENTS = "macro_events"
    NATURAL_EVENTS = "natural_events"
    GEOPOLITICS = "geopolitics"
    INNOVATION = "innovation"
    WEATHER = "weather"


class DataQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class NormalizedRecord:
    """Unified schema for all ingested data."""
    timestamp: datetime
    category: APICategory
    source_api: str
    data_type: str          # e.g. "ohlcv", "sentiment", "event"
    payload: Dict[str, Any]
    quality: DataQuality = DataQuality.UNKNOWN
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitConfig:
    """Per-API rate limit configuration."""
    max_requests_per_minute: int = 60
    max_requests_per_second: int = 5
    burst_size: int = 10


# ---------------------------------------------------------------------------
# Base API Client
# ---------------------------------------------------------------------------

class BaseAPIClient(ABC):
    """Abstract base for all external API clients."""

    def __init__(
        self,
        name: str,
        category: APICategory,
        api_key: str = "",
        base_url: str = "",
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        self.name = name
        self.category = category
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit = rate_limit or RateLimitConfig()
        self._request_timestamps: List[float] = []
        self._enabled = True
        self._error_count = 0
        self._success_count = 0
        self._weight = 1.0  # reinforcement-learning weight

    # -- public helpers --

    @property
    def reliability(self) -> float:
        total = self._success_count + self._error_count
        return self._success_count / total if total > 0 else 0.5

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float):
        self._weight = max(0.0, min(2.0, value))

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # -- rate limiting --

    async def _wait_for_rate_limit(self):
        now = time.monotonic()
        # prune old timestamps
        cutoff = now - 60.0
        self._request_timestamps = [
            t for t in self._request_timestamps if t > cutoff
        ]
        if len(self._request_timestamps) >= self.rate_limit.max_requests_per_minute:
            wait = self._request_timestamps[0] - cutoff
            logger.debug(f"[{self.name}] rate-limit wait {wait:.2f}s")
            await asyncio.sleep(wait)
        self._request_timestamps.append(now)

    # -- abstract interface --

    @abstractmethod
    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        """Fetch data and return normalized records."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the API is reachable."""
        ...

    # -- convenience --

    async def safe_fetch(self, **kwargs) -> List[NormalizedRecord]:
        """Fetch with rate limiting, error tracking, and fallback."""
        if not self._enabled:
            return []
        try:
            await self._wait_for_rate_limit()
            records = await self.fetch(**kwargs)
            self._success_count += 1
            return records
        except Exception as exc:
            self._error_count += 1
            logger.warning(f"[{self.name}] fetch error: {exc}")
            return []

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} name={self.name} "
            f"category={self.category.value} enabled={self._enabled} "
            f"reliability={self.reliability:.2f}>"
        )


# ---------------------------------------------------------------------------
# API Registry
# ---------------------------------------------------------------------------

class APIRegistry:
    """
    Central registry that holds all API clients, dispatches requests,
    and aggregates results across multiple sources.
    """

    def __init__(self):
        self._clients: Dict[str, BaseAPIClient] = {}
        self._category_index: Dict[APICategory, List[str]] = {
            cat: [] for cat in APICategory
        }
        self._hooks: Dict[str, List[Callable]] = {
            "pre_fetch": [],
            "post_fetch": [],
            "on_error": [],
        }

    # -- registration --

    def register(self, client: BaseAPIClient):
        """Register an API client."""
        self._clients[client.name] = client
        if client.name not in self._category_index[client.category]:
            self._category_index[client.category].append(client.name)
        logger.info(f"Registered API client: {client}")

    def unregister(self, name: str):
        """Remove an API client."""
        client = self._clients.pop(name, None)
        if client:
            cat_list = self._category_index.get(client.category, [])
            if name in cat_list:
                cat_list.remove(name)

    def get_client(self, name: str) -> Optional[BaseAPIClient]:
        return self._clients.get(name)

    def get_clients_by_category(self, category: APICategory) -> List[BaseAPIClient]:
        names = self._category_index.get(category, [])
        return [self._clients[n] for n in names if n in self._clients]

    # -- hooks --

    def add_hook(self, event: str, callback: Callable):
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _fire_hooks(self, event: str, **kwargs):
        for cb in self._hooks.get(event, []):
            try:
                cb(**kwargs)
            except Exception as exc:
                logger.warning(f"Hook error ({event}): {exc}")

    # -- high-level fetch methods --

    async def fetch_category(
        self,
        category: APICategory,
        **kwargs,
    ) -> List[NormalizedRecord]:
        """Fetch from all enabled clients in a category, merge results."""
        clients = self.get_clients_by_category(category)
        enabled = [c for c in clients if c.is_enabled]
        if not enabled:
            logger.warning(f"No enabled clients for {category.value}")
            return []

        self._fire_hooks("pre_fetch", category=category, kwargs=kwargs)

        tasks = [c.safe_fetch(**kwargs) for c in enabled]
        results_nested = await asyncio.gather(*tasks)
        records: List[NormalizedRecord] = []
        for batch in results_nested:
            records.extend(batch)

        self._fire_hooks("post_fetch", category=category, records=records)
        return records

    async def fetch_market_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[NormalizedRecord]:
        return await self.fetch_category(
            APICategory.MARKET_DATA,
            symbol=symbol, interval=interval, limit=limit,
        )

    async def fetch_sentiment(self, query: str, limit: int = 50) -> List[NormalizedRecord]:
        return await self.fetch_category(
            APICategory.SENTIMENT,
            query=query, limit=limit,
        )

    async def fetch_macro_events(self, region: str = "global", days_ahead: int = 7) -> List[NormalizedRecord]:
        return await self.fetch_category(
            APICategory.MACRO_EVENTS,
            region=region, days_ahead=days_ahead,
        )

    async def fetch_natural_events(self, region: str = "global") -> List[NormalizedRecord]:
        return await self.fetch_category(
            APICategory.NATURAL_EVENTS,
            region=region,
        )

    async def fetch_geopolitics(self, query: str = "") -> List[NormalizedRecord]:
        return await self.fetch_category(
            APICategory.GEOPOLITICS,
            query=query,
        )

    async def fetch_innovation(self, sector: str = "") -> List[NormalizedRecord]:
        return await self.fetch_category(
            APICategory.INNOVATION,
            sector=sector,
        )

    async def fetch_weather(
        self,
        location: str = "52.520551,13.461804",  # Berlin default
        parameters: str = "t_2m:C,precip_1h:mm,wind_speed_10m:ms",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "PT1H",
    ) -> List[NormalizedRecord]:
        """Fetch weather data for a location and time range."""
        return await self.fetch_category(
            APICategory.WEATHER,
            location=location,
            parameters=parameters,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
        )

    # -- weight management (reinforcement learning) --

    def update_weights(self, performance: Dict[str, float]):
        """
        Update source weights based on prediction performance.
        ``performance`` maps client name → accuracy delta.
        """
        for name, delta in performance.items():
            client = self._clients.get(name)
            if client:
                client.weight = client.weight + 0.1 * delta
                logger.info(f"[{name}] weight updated to {client.weight:.3f}")

    # -- health --

    async def health_check_all(self) -> Dict[str, bool]:
        results = {}
        for name, client in self._clients.items():
            try:
                results[name] = await client.health_check()
            except Exception:
                results[name] = False
        return results

    # -- configuration helper --

    def configure_from_dict(self, config: Dict[str, Any]):
        """
        Bulk-configure clients from a dict like:
        {
            "binance": {"api_key": "...", "enabled": True},
            "newsapi": {"api_key": "...", "enabled": True},
            ...
        }
        Clients must already be registered; this just sets keys/enabled.
        """
        for name, cfg in config.items():
            client = self._clients.get(name)
            if not client:
                continue
            if "api_key" in cfg:
                client.api_key = cfg["api_key"]
            if "enabled" in cfg:
                if cfg["enabled"]:
                    client.enable()
                else:
                    client.disable()

    # -- summary --

    def summary(self) -> Dict[str, Any]:
        return {
            "total_clients": len(self._clients),
            "by_category": {
                cat.value: len(names)
                for cat, names in self._category_index.items()
            },
            "clients": {
                name: {
                    "category": c.category.value,
                    "enabled": c.is_enabled,
                    "reliability": round(c.reliability, 3),
                    "weight": round(c.weight, 3),
                }
                for name, c in self._clients.items()
            },
        }

    def __repr__(self):
        return f"<APIRegistry clients={len(self._clients)}>"
