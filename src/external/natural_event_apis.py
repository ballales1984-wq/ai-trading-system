"""
Natural Events & Climate API Clients
======================================
Clients for: Open-Meteo, Climate TRACE, USGS Water Data.

These provide weather, climate, and hydrological data that feed into
Feature Engineering and Monte Carlo Level 4 (Multi-Factor) for
commodity and energy price correlations.
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
# Open-Meteo Client
# ---------------------------------------------------------------------------

class OpenMeteoClient(BaseAPIClient):
    """
    Open-Meteo — free weather and climate data.
    Docs: https://open-meteo.com/en/docs
    No API key required.
    """

    def __init__(self):
        super().__init__(
            name="open_meteo",
            category=APICategory.NATURAL_EVENTS,
            api_key="",
            base_url="https://api.open-meteo.com/v1",
            rate_limit=RateLimitConfig(max_requests_per_minute=600, max_requests_per_second=10),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        latitude = kwargs.get("latitude", 40.71)   # default: New York
        longitude = kwargs.get("longitude", -74.01)
        region = kwargs.get("region", "US")
        days_back = kwargs.get("days_back", 30)

        start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        url = f"{self.base_url}/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
        }

        records: List[NormalizedRecord] = []
        if aiohttp is None:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Open-Meteo HTTP {resp.status}")
                data = await resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        wind = daily.get("windspeed_10m_max", [])

        for i, date_str in enumerate(dates):
            try:
                ts = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            t_max = temp_max[i] if i < len(temp_max) else None
            t_min = temp_min[i] if i < len(temp_min) else None
            p = precip[i] if i < len(precip) else None
            w = wind[i] if i < len(wind) else None

            # Classify extreme events
            event_type = "normal"
            intensity = 0.0
            if t_max is not None and t_max > 40:
                event_type = "heatwave"
                intensity = (t_max - 40) / 10.0
            elif t_min is not None and t_min < -20:
                event_type = "cold_snap"
                intensity = (-20 - t_min) / 10.0
            elif p is not None and p > 50:
                event_type = "heavy_rain"
                intensity = p / 100.0
            elif w is not None and w > 100:
                event_type = "storm"
                intensity = w / 150.0

            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.NATURAL_EVENTS,
                    source_api="open_meteo",
                    data_type="weather",
                    payload={
                        "region": region,
                        "temperature_max": t_max,
                        "temperature_min": t_min,
                        "precipitation_mm": p,
                        "windspeed_max_kmh": w,
                        "event_type": event_type,
                        "intensity": round(min(intensity, 1.0), 3),
                    },
                    quality=DataQuality.HIGH,
                    confidence=0.9,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None:
            return False
        try:
            url = f"{self.base_url}/forecast"
            params = {"latitude": 52.52, "longitude": 13.41, "daily": "temperature_2m_max", "timezone": "UTC"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Climate TRACE Client
# ---------------------------------------------------------------------------

class ClimateTRACEClient(BaseAPIClient):
    """
    Climate TRACE — emissions and climate change data.
    Docs: https://climatetrace.org/data
    """

    def __init__(self):
        super().__init__(
            name="climate_trace",
            category=APICategory.NATURAL_EVENTS,
            api_key="",
            base_url="https://api.climatetrace.org/v4",
            rate_limit=RateLimitConfig(max_requests_per_minute=30, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        region = kwargs.get("region", "global")
        sector = kwargs.get("sector", "power")  # power, transportation, etc.

        url = f"{self.base_url}/country/emissions"
        params: Dict[str, Any] = {
            "since": 2020,
            "to": 2026,
        }
        if region and region != "global":
            params["country"] = region.upper()

        records: List[NormalizedRecord] = []
        if aiohttp is None:
            return records

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"ClimateTRACE HTTP {resp.status}")
                    data = await resp.json()

            for entry in data if isinstance(data, list) else []:
                year = entry.get("year", 2024)
                ts = datetime(year, 1, 1, tzinfo=timezone.utc)
                records.append(
                    NormalizedRecord(
                        timestamp=ts,
                        category=APICategory.NATURAL_EVENTS,
                        source_api="climate_trace",
                        data_type="emissions",
                        payload={
                            "country": entry.get("country", region),
                            "sector": entry.get("sector", sector),
                            "emissions_co2_tonnes": entry.get("co2", 0),
                            "emissions_ch4_tonnes": entry.get("ch4", 0),
                            "year": year,
                        },
                        quality=DataQuality.MEDIUM,
                        confidence=0.7,
                    )
                )
        except Exception as exc:
            logger.warning(f"ClimateTRACE fetch failed: {exc}")

        return records

    async def health_check(self) -> bool:
        if aiohttp is None:
            return False
        try:
            url = f"{self.base_url}/country/emissions"
            params = {"since": 2023, "to": 2024}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# USGS Water Data Client
# ---------------------------------------------------------------------------

class USGSWaterClient(BaseAPIClient):
    """
    USGS Water Data — hydrological data (streamflow, water levels).
    Docs: https://waterservices.usgs.gov/
    Free, no API key required.
    """

    def __init__(self):
        super().__init__(
            name="usgs_water",
            category=APICategory.NATURAL_EVENTS,
            api_key="",
            base_url="https://waterservices.usgs.gov/nwis",
            rate_limit=RateLimitConfig(max_requests_per_minute=30, max_requests_per_second=1),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        # Default: Colorado River at Lees Ferry (important for water/energy)
        site_no = kwargs.get("site_no", "09380000")
        days_back = kwargs.get("days_back", 30)
        region = kwargs.get("region", "US")

        url = f"{self.base_url}/iv/"
        params = {
            "format": "json",
            "sites": site_no,
            "period": f"P{days_back}D",
            "parameterCd": "00060",  # discharge (cubic feet/sec)
            "siteStatus": "active",
        }

        records: List[NormalizedRecord] = []
        if aiohttp is None:
            return records

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"USGS HTTP {resp.status}")
                    data = await resp.json()

            time_series = data.get("value", {}).get("timeSeries", [])
            for series in time_series:
                site_name = series.get("sourceInfo", {}).get("siteName", "")
                values = series.get("values", [{}])[0].get("value", [])

                for val in values:
                    date_str = val.get("dateTime", "")
                    try:
                        ts = datetime.fromisoformat(date_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except (ValueError, AttributeError):
                        continue

                    discharge = float(val.get("value", 0))
                    # Classify drought/flood conditions
                    event_type = "normal"
                    intensity = 0.0
                    if discharge < 100:
                        event_type = "drought"
                        intensity = 1.0 - (discharge / 100.0)
                    elif discharge > 10000:
                        event_type = "flood"
                        intensity = min(discharge / 50000.0, 1.0)

                    records.append(
                        NormalizedRecord(
                            timestamp=ts,
                            category=APICategory.NATURAL_EVENTS,
                            source_api="usgs_water",
                            data_type="hydrology",
                            payload={
                                "site": site_name,
                                "site_no": site_no,
                                "discharge_cfs": discharge,
                                "event_type": event_type,
                                "intensity": round(intensity, 3),
                                "region": region,
                            },
                            quality=DataQuality.HIGH,
                            confidence=0.85,
                        )
                    )
        except Exception as exc:
            logger.warning(f"USGS Water fetch failed: {exc}")

        return records

    async def health_check(self) -> bool:
        if aiohttp is None:
            return False
        try:
            url = f"{self.base_url}/iv/"
            params = {"format": "json", "sites": "09380000", "period": "P1D", "parameterCd": "00060"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_natural_event_clients(
    include_open_meteo: bool = True,
    include_climate_trace: bool = True,
    include_usgs: bool = True,
) -> List[BaseAPIClient]:
    """Create all natural-event clients (all free, no keys needed)."""
    clients: List[BaseAPIClient] = []
    if include_open_meteo:
        clients.append(OpenMeteoClient())
    if include_climate_trace:
        clients.append(ClimateTRACEClient())
    if include_usgs:
        clients.append(USGSWaterClient())
    return clients


def create_natural_event_clients_from_env() -> List[BaseAPIClient]:
    """Create natural-event clients (all free)."""
    return create_natural_event_clients()
