"""
Innovation & Energy API Clients
=================================
Clients for: EIA (Energy), Google Patents / Lens.org (Innovation).

These provide energy market data and technology innovation signals
that feed into Monte Carlo Level 5 (Semantic History) for
stress testing and long-term scenario generation.
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
# EIA (U.S. Energy Information Administration) Client
# ---------------------------------------------------------------------------

class EIAClient(BaseAPIClient):
    """
    EIA API — U.S. energy data (petroleum, natural gas, electricity).
    Docs: https://www.eia.gov/opendata/documentation.php
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="eia",
            category=APICategory.INNOVATION,
            api_key=api_key,
            base_url="https://api.eia.gov/v2",
            rate_limit=RateLimitConfig(max_requests_per_minute=100, max_requests_per_second=5),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        sector = kwargs.get("sector", "electricity")
        series_id = kwargs.get("series_id", "")

        # EIA v2 uses different endpoints and parameters
        if not series_id:
            route_map = {
                "petroleum": "/petroleum/pri/spt/data/",
                "natural-gas": "/natural-gas/pri/sum/data/",
                "electricity": "/electricity/retail-sales/data/",
            }
            route = route_map.get(sector, "/electricity/retail-sales/data/")
        else:
            route = f"/seriesid/{series_id}"

        url = f"{self.base_url}{route}"
        
        # Build params based on sector - EIA v2 format
        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "length": 100,
        }
        
        # Add sector-specific parameters
        if sector == "electricity":
            params["frequency"] = "annual"
            params["data[0]"] = "price"
            params["facets[sectorid][]"] = "RES"  # residential
        elif sector == "petroleum":
            params["frequency"] = "monthly"
            params["data[0]"] = "value"
        elif sector == "natural-gas":
            params["frequency"] = "monthly"
            params["data[0]"] = "value"

        records: List[NormalizedRecord] = []
        if not self.api_key:
            return records

        try:
            # Use requests instead of aiohttp for better DNS handling
            import requests
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"EIA HTTP {resp.status_code}")
            data = resp.json()

            response_data = data.get("response", {}).get("data", [])
            for entry in response_data:
                period = entry.get("period", "")
                
                # Find the value field - EIA v2 uses different field names
                # Check common value field names
                value = None
                for key in ['value', 'price', 'price-value', 'total']:
                    if key in entry and entry[key]:
                        try:
                            value = float(entry[key])
                            break
                        except (ValueError, TypeError):
                            pass
                
                try:
                    if len(period) == 7:  # YYYY-MM
                        ts = datetime.strptime(period, "%Y-%m").replace(tzinfo=timezone.utc)
                    elif len(period) == 10:  # YYYY-MM-DD
                        ts = datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    else:
                        ts = datetime.strptime(period[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)

                if value is not None:
                    # Get appropriate field names based on sector
                    if sector == "electricity":
                        area = entry.get("stateDescription", entry.get("stateid", ""))
                        product = entry.get("sectorName", entry.get("sectorid", ""))
                        units = entry.get("price-units", "")
                    else:
                        area = entry.get("area-name", entry.get("area", ""))
                        product = entry.get("product-name", entry.get("product", ""))
                        units = entry.get("units", "")
                    
                    records.append(
                        NormalizedRecord(
                            timestamp=ts,
                            category=APICategory.INNOVATION,
                            source_api="eia",
                            data_type="energy_price",
                            payload={
                                "sector": sector,
                                "product": product,
                                "area": area,
                                "value": value,
                                "units": units,
                            },
                            quality=DataQuality.HIGH,
                            confidence=0.9,
                        )
                    )
        except Exception as exc:
            logger.warning(f"EIA fetch failed: {exc}")

        return records

    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            # Use requests for health check (more reliable than aiohttp)
            import requests
            url = f"{self.base_url}/petroleum/pri/spt/data/"
            params = {"api_key": self.api_key, "length": 1}
            resp = requests.get(url, params=params, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Google Patents Client (via SerpAPI or BigQuery)
# ---------------------------------------------------------------------------

class GooglePatentsClient(BaseAPIClient):
    """
    Google Patents — innovation and technology patent data.
    Uses SerpAPI as a proxy (https://serpapi.com/google-patents-api).
    Alternative: Google BigQuery Patents Public Dataset.
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="google_patents",
            category=APICategory.INNOVATION,
            api_key=api_key,
            base_url="https://serpapi.com/search",
            rate_limit=RateLimitConfig(max_requests_per_minute=30, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        sector = kwargs.get("sector", "artificial intelligence")
        query = kwargs.get("query", sector)
        limit = kwargs.get("limit", 20)

        params = {
            "engine": "google_patents",
            "q": query,
            "api_key": self.api_key,
            "num": min(limit, 100),
        }

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"GooglePatents HTTP {resp.status}")
                    data = await resp.json()

            for result in data.get("organic_results", []):
                date_str = result.get("filing_date", result.get("publication_date", ""))
                try:
                    ts = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)

                # Score innovation impact based on citations
                citations = result.get("cited_by", {}).get("total", 0)
                impact_score = min(citations / 100.0, 1.0)

                records.append(
                    NormalizedRecord(
                        timestamp=ts,
                        category=APICategory.INNOVATION,
                        source_api="google_patents",
                        data_type="patent",
                        payload={
                            "title": result.get("title", ""),
                            "patent_id": result.get("patent_id", ""),
                            "assignee": result.get("assignee", ""),
                            "sector": sector,
                            "citations": citations,
                            "impact_score": round(impact_score, 3),
                            "abstract": result.get("snippet", ""),
                        },
                        quality=DataQuality.MEDIUM,
                        confidence=0.5,
                    )
                )
        except Exception as exc:
            logger.warning(f"GooglePatents fetch failed: {exc}")

        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            params = {"engine": "google_patents", "q": "test", "api_key": self.api_key, "num": 1}
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Lens.org Client
# ---------------------------------------------------------------------------

class LensOrgClient(BaseAPIClient):
    """
    Lens.org — scholarly and patent data for innovation tracking.
    Docs: https://docs.api.lens.org/
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="lens_org",
            category=APICategory.INNOVATION,
            api_key=api_key,
            base_url="https://api.lens.org/patent/search",
            rate_limit=RateLimitConfig(max_requests_per_minute=10, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        sector = kwargs.get("sector", "renewable energy")
        query = kwargs.get("query", sector)
        limit = kwargs.get("limit", 20)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "query": {"match": {"title": query}},
            "size": min(limit, 100),
            "sort": [{"date_published": "desc"}],
        }

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Lens.org HTTP {resp.status}")
                    data = await resp.json()

            for item in data.get("data", []):
                date_str = item.get("date_published", "")
                try:
                    ts = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)

                families = item.get("families", {}).get("simple_family", {}).get("size", 0)
                impact_score = min(families / 50.0, 1.0)

                records.append(
                    NormalizedRecord(
                        timestamp=ts,
                        category=APICategory.INNOVATION,
                        source_api="lens_org",
                        data_type="patent",
                        payload={
                            "title": item.get("title", ""),
                            "lens_id": item.get("lens_id", ""),
                            "applicants": item.get("applicants", []),
                            "sector": sector,
                            "family_size": families,
                            "impact_score": round(impact_score, 3),
                        },
                        quality=DataQuality.MEDIUM,
                        confidence=0.5,
                    )
                )
        except Exception as exc:
            logger.warning(f"Lens.org fetch failed: {exc}")

        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            body = {"query": {"match": {"title": "test"}}, "size": 1}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=body, headers=headers) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_innovation_clients(
    eia_key: str = "",
    serpapi_key: str = "",
    lens_key: str = "",
) -> List[BaseAPIClient]:
    """Create all innovation/energy clients from provided keys."""
    clients: List[BaseAPIClient] = []
    if eia_key:
        clients.append(EIAClient(api_key=eia_key))
    if serpapi_key:
        clients.append(GooglePatentsClient(api_key=serpapi_key))
    if lens_key:
        clients.append(LensOrgClient(api_key=lens_key))
    return clients


def create_innovation_clients_from_env() -> List[BaseAPIClient]:
    """Create innovation clients using environment variables."""
    return create_innovation_clients(
        eia_key=os.getenv("EIA_API_KEY", ""),
        serpapi_key=os.getenv("SERPAPI_KEY", ""),
        lens_key=os.getenv("LENS_ORG_API_KEY", ""),
    )
