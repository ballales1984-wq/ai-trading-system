"""
Macro Event / Economic Calendar API Clients
=============================================
Clients for: Trading Economics, EconPulse, Investing.com.

These provide economic calendar events (GDP, CPI, FOMC, etc.)
that feed into the Event Processor and Monte Carlo Level 2-3.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
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
# Trading Economics Calendar Client
# ---------------------------------------------------------------------------

class TradingEconomicsClient(BaseAPIClient):
    """
    Trading Economics — global macroeconomic calendar.
    Docs: https://docs.tradingeconomics.com/
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="trading_economics",
            category=APICategory.MACRO_EVENTS,
            api_key=api_key,
            base_url="https://api.tradingeconomics.com",
            rate_limit=RateLimitConfig(max_requests_per_minute=60, max_requests_per_second=2),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        region = kwargs.get("region", "global")
        days_ahead = kwargs.get("days_ahead", 7)

        start = datetime.now(timezone.utc)
        end = start + timedelta(days=days_ahead)

        url = f"{self.base_url}/calendar"
        params: Dict[str, Any] = {
            "c": self.api_key,
            "d1": start.strftime("%Y-%m-%d"),
            "d2": end.strftime("%Y-%m-%d"),
        }
        if region and region != "global":
            url = f"{self.base_url}/calendar/country/{region}"

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"TradingEconomics HTTP {resp.status}")
                data = await resp.json()

        for event in data if isinstance(data, list) else []:
            date_str = event.get("Date", "")
            try:
                ts = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(timezone.utc)

            importance = event.get("Importance", 0)
            impact = "high" if importance >= 3 else ("medium" if importance >= 2 else "low")

            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.MACRO_EVENTS,
                    source_api="trading_economics",
                    data_type="economic_event",
                    payload={
                        "event": event.get("Event", ""),
                        "country": event.get("Country", ""),
                        "category": event.get("Category", ""),
                        "impact": impact,
                        "actual": event.get("Actual", None),
                        "forecast": event.get("Forecast", None),
                        "previous": event.get("Previous", None),
                        "currency": event.get("Currency", ""),
                    },
                    quality=DataQuality.HIGH,
                    confidence=0.8,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            url = f"{self.base_url}/calendar"
            params = {"c": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# EconPulse Client (placeholder — custom or alternative source)
# ---------------------------------------------------------------------------

class EconPulseClient(BaseAPIClient):
    """
    EconPulse — USA/global economic pulse events.
    This is a placeholder for a custom or alternative economic data source.
    Replace base_url and parsing logic with the actual provider.
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="econpulse",
            category=APICategory.MACRO_EVENTS,
            api_key=api_key,
            base_url="https://api.econpulse.io/v1",  # placeholder URL
            rate_limit=RateLimitConfig(max_requests_per_minute=30, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        region = kwargs.get("region", "US")
        days_ahead = kwargs.get("days_ahead", 7)

        url = f"{self.base_url}/events"
        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "region": region,
            "days": days_ahead,
        }

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"EconPulse HTTP {resp.status}")
                    data = await resp.json()

            for event in data.get("events", []):
                date_str = event.get("date", "")
                try:
                    ts = datetime.fromisoformat(date_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError):
                    ts = datetime.now(timezone.utc)

                records.append(
                    NormalizedRecord(
                        timestamp=ts,
                        category=APICategory.MACRO_EVENTS,
                        source_api="econpulse",
                        data_type="economic_event",
                        payload={
                            "event": event.get("name", ""),
                            "country": event.get("country", region),
                            "impact": event.get("impact", "medium"),
                            "actual": event.get("actual"),
                            "forecast": event.get("forecast"),
                            "previous": event.get("previous"),
                        },
                        quality=DataQuality.MEDIUM,
                        confidence=0.6,
                    )
                )
        except Exception as exc:
            logger.warning(f"EconPulse fetch failed: {exc}")

        return records

    async def health_check(self) -> bool:
        return False  # placeholder — not yet available


# ---------------------------------------------------------------------------
# Investing.com Calendar Client (via scraping or partner API)
# ---------------------------------------------------------------------------

class InvestingComClient(BaseAPIClient):
    """
    Investing.com — global economic calendar events.
    Note: Investing.com doesn't have a free public API.
    This client uses their unofficial endpoints or a partner API.
    Replace with actual integration when available.
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="investing_com",
            category=APICategory.MACRO_EVENTS,
            api_key=api_key,
            base_url="https://api.investing.com/api",  # placeholder
            rate_limit=RateLimitConfig(max_requests_per_minute=10, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        # Placeholder — Investing.com requires partner access
        logger.info("Investing.com client: partner API not configured")
        return []

    async def health_check(self) -> bool:
        return False  # requires partner access


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_macro_event_clients(
    trading_economics_key: str = "",
    econpulse_key: str = "",
    investing_com_key: str = "",
) -> List[BaseAPIClient]:
    """Create all macro-event clients from provided keys."""
    clients: List[BaseAPIClient] = []
    if trading_economics_key:
        clients.append(TradingEconomicsClient(api_key=trading_economics_key))
    if econpulse_key:
        clients.append(EconPulseClient(api_key=econpulse_key))
    if investing_com_key:
        clients.append(InvestingComClient(api_key=investing_com_key))
    return clients


def create_macro_event_clients_from_env() -> List[BaseAPIClient]:
    """Create macro-event clients using environment variables."""
    return create_macro_event_clients(
        trading_economics_key=os.getenv("TRADING_ECONOMICS_API_KEY", ""),
        econpulse_key=os.getenv("ECONPULSE_API_KEY", ""),
        investing_com_key=os.getenv("INVESTING_COM_API_KEY", ""),
    )
