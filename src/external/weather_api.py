"""
Weather API Client — Meteomatics Integration
=============================================
Provides weather data for trading decisions, including:
- Temperature, precipitation, wind speed
- Historical, current, and forecast data
- Grid-based and point-based queries

API Documentation: https://www.meteomatics.com/en/api/getting-started/

Usage:
    client = MeteomaticsClient(username="user", password="pass")
    data = await client.fetch_weather(
        location="52.520551,13.461804",  # Berlin
        parameters="t_2m:C,precip_1h:mm,wind_speed_10m:ms",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc) + timedelta(days=3),
    )
"""

import asyncio
import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .api_registry import (
    APICategory,
    BaseAPIClient,
    DataQuality,
    NormalizedRecord,
    RateLimitConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weather Parameters
# ---------------------------------------------------------------------------

class WeatherParameter:
    """Common weather parameters available from Meteomatics API."""
    
    # Temperature
    TEMPERATURE_2M = "t_2m:C"  # Temperature at 2m in Celsius
    TEMPERATURE_MAX = "t_max_2m_24h:C"
    TEMPERATURE_MIN = "t_min_2m_24h:C"
    TEMPERATURE_FEELS_LIKE = "t_feel:C"
    
    # Precipitation
    PRECIP_1H = "precip_1h:mm"  # Precipitation in last hour (mm)
    PRECIP_24H = "precip_24h:mm"
    PRECIP_PROBABILITY = "prob_precip_1h:p"
    
    # Wind
    WIND_SPEED_10M = "wind_speed_10m:ms"  # Wind speed at 10m (m/s)
    WIND_SPEED_100M = "wind_speed_100m:ms"
    WIND_DIRECTION_10M = "wind_dir_10m:d"
    WIND_GUST_10M = "wind_gusts_10m_1h:ms"
    
    # Pressure & Humidity
    PRESSURE_MSL = "msl_pressure:hPa"
    RELATIVE_HUMIDITY = "relative_humidity_2m:p"
    DEW_POINT = "dew_point_2m:C"
    
    # Cloud & Visibility
    CLOUD_COVER = "total_cloud_cover:p"
    VISIBILITY = "visibility:m"
    
    # Solar
    SUNSHINE_DURATION = "sunshine_duration_1h:min"
    SOLAR_RADIATION = "global_rad:W"
    
    # Severe Weather
    CAPE = "cape:J"  # Convective Available Potential Energy
    LIGHTNING_PROB = "prob_lightning_1h:p"
    
    # Agricultural
    SOIL_TEMPERATURE = "soil_temperature_0cm:C"
    SOIL_MOISTURE = "soil_moisture_index_0cm:p"
    EVAPOTRANSPIRATION = "evapotranspiration_24h:mm"


# ---------------------------------------------------------------------------
# Location Helpers
# ---------------------------------------------------------------------------

class WeatherLocation:
    """Predefined locations for common trading hubs."""
    
    # Major Financial Centers
    NEW_YORK = "40.7128,-74.0060"
    LONDON = "51.5074,-0.1278"
    TOKYO = "35.6762,139.6503"
    FRANKFURT = "50.1109,8.6821"
    HONG_KONG = "22.3193,114.1694"
    SINGAPORE = "1.3521,103.8198"
    SYDNEY = "-33.8688,151.2093"
    DUBAI = "25.2048,55.2708"
    
    # Commodity Production Regions
    # Oil
    HOUSTON = "29.7604,-95.3698"
    RIYADH = "24.7136,46.6753"
    MOSCOW = "55.7558,37.6173"
    
    # Agriculture
    CHICAGO = "41.8781,-87.6298"
    SAO_PAULO = "-23.5505,-46.6333"
    BUENOS_AIRES = "-34.6037,-58.3816"
    
    # Metals Mining
    PERTH = "-31.9505,115.8605"
    JOHANNESBURG = "-26.2041,28.0473"
    SANTIAGO = "-33.4489,-70.6693"
    
    @staticmethod
    def from_lat_lon(lat: float, lon: float) -> str:
        """Create location string from latitude and longitude."""
        return f"{lat},{lon}"
    
    @staticmethod
    def bounding_box(
        lat_min: float, lon