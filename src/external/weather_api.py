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
    BERLIN = "52.520551,13.461804"
    
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
        lat_min: float, lon_min: float,
        lat_max: float, lon_max: float,
        resolution_lat: float = 0.5,
        resolution_lon: float = 0.5,
    ) -> str:
        """Create a grid bounding box for area queries."""
        return f"{lat_min},{lon_min}_{lat_max},{lon_max}:{resolution_lat},{resolution_lon}"


# ---------------------------------------------------------------------------
# Weather Impact Scoring
# ---------------------------------------------------------------------------

class WeatherImpactScorer:
    """
    Scores weather data for commodity trading impact.
    
    Different commodities are affected by different weather conditions:
    - Agriculture: precipitation, temperature, soil moisture
    - Energy: temperature (heating/cooling demand), wind (renewables)
    - Metals: minimal direct impact, but mining operations can be affected
    """
    
    # Commodity sensitivity weights
    COMMODITY_WEIGHTS = {
        "agriculture": {
            "precipitation": 0.4,
            "temperature": 0.3,
            "soil_moisture": 0.2,
            "extreme_events": 0.1,
        },
        "energy": {
            "temperature": 0.5,
            "wind": 0.3,
            "solar": 0.2,
        },
        "metals": {
            "precipitation": 0.3,  # Flooding can disrupt mining
            "extreme_events": 0.7,
        },
    }
    
    @staticmethod
    def score_temperature(temp_c: float, commodity_type: str = "energy") -> float:
        """
        Score temperature impact on commodity.
        
        For energy:
        - Extreme cold (heating demand): positive score
        - Extreme heat (cooling demand): positive score
        - Mild temperatures: negative score (low demand)
        
        Returns score from -1 (bearish) to 1 (bullish)
        """
        if commodity_type == "energy":
            # Heating Degree Days effect
            if temp_c < 10:
                # Cold weather increases heating demand
                return min(1.0, (10 - temp_c) / 20)
            elif temp_c > 25:
                # Hot weather increases cooling demand
                return min(1.0, (temp_c - 25) / 15)
            else:
                # Mild weather, low demand
                return -0.3
        elif commodity_type == "agriculture":
            # Optimal growing temperatures
            if 15 <= temp_c <= 30:
                return 0.2  # Good for crops
            elif temp_c < 0 or temp_c > 40:
                return -0.8  # Crop damage
            else:
                return 0.0
        return 0.0
    
    @staticmethod
    def score_precipitation(
        precip_mm: float,
        commodity_type: str = "agriculture",
        region_type: str = "temperate",
    ) -> float:
        """
        Score precipitation impact.
        
        For agriculture:
        - Moderate rain: positive
        - Drought or flood: negative
        
        For energy:
        - Hydro: positive correlation
        - Other: minimal impact
        """
        if commodity_type == "agriculture":
            if region_type == "temperate":
                if 5 <= precip_mm <= 25:  # Good rainfall per day
                    return 0.3
                elif precip_mm > 50:  # Flooding risk
                    return -0.7
                elif precip_mm < 2:  # Drought risk
                    return -0.5
            elif region_type == "tropical":
                if 10 <= precip_mm <= 50:
                    return 0.2
                elif precip_mm > 100:
                    return -0.6
        return 0.0
    
    @staticmethod
    def score_wind(wind_speed_ms: float) -> float:
        """
        Score wind speed for renewable energy.
        
        - Optimal wind range: positive
        - Too low or too high: reduced output
        """
        # Wind turbines operate optimally at 12-25 m/s
        if 12 <= wind_speed_ms <= 25:
            return 0.5  # Good wind power generation
        elif 6 <= wind_speed_ms < 12:
            return 0.2  # Moderate generation
        elif wind_speed_ms > 25:
            return -0.3  # Turbines may shut down
        else:
            return -0.2  # Low generation


# ---------------------------------------------------------------------------
# Meteomatics API Client
# ---------------------------------------------------------------------------

class MeteomaticsClient(BaseAPIClient):
    """
    Client for Meteomatics Weather API.
    
    API URL format:
        api.meteomatics.com/validdatetime/parameters/locations/format
    
    Authentication: HTTP Basic Auth (username:password)
    """
    
    BASE_URL = "https://api.meteomatics.com"
    
    def __init__(
        self,
        username: str = "",
        password: str = "",
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        super().__init__(
            name="meteomatics",
            category=APICategory.WEATHER,
            api_key=username,  # Store username in api_key field
            base_url=self.BASE_URL,
            rate_limit=rate_limit or RateLimitConfig(
                max_requests_per_minute=60,
                max_requests_per_second=5,
            ),
        )
        self.password = password
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _get_auth_header(self) -> str:
        """Generate Basic Auth header."""
        credentials = f"{self.api_key}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": self._get_auth_header(),
                "Accept": "application/json",
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_url(
        self,
        valid_datetime: str,
        parameters: str,
        location: str,
        format: str = "json",
    ) -> str:
        """
        Build the API URL.
        
        Args:
            valid_datetime: ISO datetime or range (e.g., "2026-02-20T00:00:00Z" 
                          or "2026-02-20T00:00:00Z--2026-02-23T00:00:00Z:PT1H")
            parameters: Comma-separated parameters (e.g., "t_2m:C,precip_1h:mm")
            location: Location string (e.g., "52.520551,13.461804")
            format: Output format (json, xml, html, png, netcdf)
        """
        return f"{self.BASE_URL}/{valid_datetime}/{parameters}/{location}/{format}"
    
    async def fetch(
        self,
        location: str = "52.520551,13.461804",
        parameters: str = "t_2m:C,precip_1h:mm,wind_speed_10m:ms",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "PT1H",
        **kwargs,
    ) -> List[NormalizedRecord]:
        """
        Fetch weather data from Meteomatics API.
        
        Args:
            location: Location string (lat,lon) or bounding box
            parameters: Comma-separated weather parameters
            start_time: Start datetime (defaults to now)
            end_time: End datetime (defaults to 24h from start)
            interval: Time interval (PT1H = hourly, PT15M = 15 min, P1D = daily)
        
        Returns:
            List of NormalizedRecord with weather data
        """
        now = datetime.now(timezone.utc)
        start = start_time or now
        end = end_time or (start + timedelta(hours=24))
        
        # Format datetime range
        datetime_str = f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}--{end.strftime('%Y-%m-%dT%H:%M:%SZ')}:{interval}"
        
        url = self._build_url(datetime_str, parameters, location)
        
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_response(data, parameters)
                elif response.status == 401:
                    logger.error(f"[{self.name}] Authentication failed")
                    self.disable()
                    return []
                else:
                    text = await response.text()
                    logger.warning(f"[{self.name}] API error {response.status}: {text}")
                    return []
        except aiohttp.ClientError as e:
            logger.error(f"[{self.name}] HTTP error: {e}")
            return []
    
    def _parse_response(
        self,
        data: Dict[str, Any],
        parameters: str,
    ) -> List[NormalizedRecord]:
        """Parse Meteomatics JSON response into NormalizedRecords."""
        records = []
        
        # Meteomatics response format:
        # {
        #   "data": [
        #     {
        #       "parameter": "t_2m:C",
        #       "coordinates": [{"lat": 52.5, "lon": 13.4, "dates": [...]}]
        #     }
        #   ]
        # }
        
        data_list = data.get("data", [])
        if not data_list:
            return records
        
        # Group by timestamp
        timestamp_data: Dict[str, Dict[str, Any]] = {}
        
        for param_data in data_list:
            parameter = param_data.get("parameter", "unknown")
            for coord in param_data.get("coordinates", []):
                lat = coord.get("lat", 0)
                lon = coord.get("lon", 0)
                for date_entry in coord.get("dates", []):
                    date_str = date_entry.get("date", "")
                    value = date_entry.get("value")
                    
                    if date_str not in timestamp_data:
                        timestamp_data[date_str] = {
                            "location": {"lat": lat, "lon": lon},
                            "parameters": {},
                        }
                    timestamp_data[date_str]["parameters"][parameter] = value
        
        # Create NormalizedRecords
        for date_str, entry in timestamp_data.items():
            try:
                timestamp = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now(timezone.utc)
            
            record = NormalizedRecord(
                timestamp=timestamp,
                category=APICategory.WEATHER,
                source_api=self.name,
                data_type="weather",
                payload={
                    "location": entry["location"],
                    "parameters": entry["parameters"],
                },
                quality=DataQuality.HIGH,
                confidence=1.0,
                metadata={
                    "raw_date": date_str,
                },
            )
            records.append(record)
        
        return records
    
    async def fetch_current(
        self,
        location: str,
        parameters: str = "t_2m:C,precip_1h:mm,wind_speed_10m:ms",
    ) -> List[NormalizedRecord]:
        """Fetch current weather for a location."""
        now = datetime.now(timezone.utc)
        return await self.fetch(
            location=location,
            parameters=parameters,
            start_time=now,
            end_time=now,
        )
    
    async def fetch_forecast(
        self,
        location: str,
        days: int = 7,
        parameters: str = "t_2m:C,precip_1h:mm,wind_speed_10m:ms",
        interval: str = "PT1H",
    ) -> List[NormalizedRecord]:
        """Fetch weather forecast for a location."""
        now = datetime.now(timezone.utc)
        return await self.fetch(
            location=location,
            parameters=parameters,
            start_time=now,
            end_time=now + timedelta(days=days),
            interval=interval,
        )
    
    async def fetch_agricultural(
        self,
        location: str,
        days: int = 7,
    ) -> List[NormalizedRecord]:
        """Fetch agricultural-relevant weather parameters."""
        params = ",".join([
            WeatherParameter.TEMPERATURE_2M,
            WeatherParameter.TEMPERATURE_MAX,
            WeatherParameter.TEMPERATURE_MIN,
            WeatherParameter.PRECIP_24H,
            WeatherParameter.SOIL_MOISTURE,
            WeatherParameter.SOIL_TEMPERATURE,
            WeatherParameter.EVAPOTRANSPIRATION,
        ])
        return await self.fetch_forecast(location, days, params)
    
    async def fetch_energy_demand(
        self,
        location: str,
        days: int = 7,
    ) -> List[NormalizedRecord]:
        """Fetch energy demand-relevant weather parameters."""
        params = ",".join([
            WeatherParameter.TEMPERATURE_2M,
            WeatherParameter.TEMPERATURE_FEELS_LIKE,
            WeatherParameter.WIND_SPEED_10M,
            WeatherParameter.WIND_SPEED_100M,
            WeatherParameter.SOLAR_RADIATION,
            WeatherParameter.CLOUD_COVER,
        ])
        return await self.fetch_forecast(location, days, params)
    
    async def health_check(self) -> bool:
        """Check if the API is accessible."""
        try:
            # Simple query for current temperature in Berlin
            records = await self.fetch_current(
                location=WeatherLocation.BERLIN,
                parameters=WeatherParameter.TEMPERATURE_2M,
            )
            return len(records) > 0
        except Exception as e:
            logger.error(f"[{self.name}] Health check failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Mock Client for Testing
# ---------------------------------------------------------------------------

class MockMeteomaticsClient(BaseAPIClient):
    """Mock weather client for testing without API credentials."""
    
    def __init__(self):
        super().__init__(
            name="meteomatics_mock",
            category=APICategory.WEATHER,
            rate_limit=RateLimitConfig(max_requests_per_minute=1000),
        )
    
    async def fetch(
        self,
        location: str = "52.520551,13.461804",
        parameters: str = "t_2m:C,precip_1h:mm,wind_speed_10m:ms",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "PT1H",
        **kwargs,
    ) -> List[NormalizedRecord]:
        """Generate mock weather data."""
        import random
        
        now = datetime.now(timezone.utc)
        start = start_time or now
        end = end_time or (start + timedelta(hours=24))
        
        records = []
        current = start
        
        # Parse location
        try:
            lat, lon = map(float, location.split(","))
        except ValueError:
            lat, lon = 52.5, 13.4
        
        while current <= end:
            # Generate realistic mock data
            temp = 15 + 10 * random.random()  # 15-25°C
            precip = random.random() * 5  # 0-5mm
            wind = 3 + 7 * random.random()  # 3-10 m/s
            
            records.append(NormalizedRecord(
                timestamp=current,
                category=APICategory.WEATHER,
                source_api=self.name,
                data_type="weather",
                payload={
                    "location": {"lat": lat, "lon": lon},
                    "parameters": {
                        "t_2m:C": round(temp, 1),
                        "precip_1h:mm": round(precip, 2),
                        "wind_speed_10m:ms": round(wind, 1),
                    },
                },
                quality=DataQuality.MEDIUM,
                confidence=0.8,
                metadata={"mock": True},
            ))
            
            # Increment by interval
            if interval == "PT1H":
                current += timedelta(hours=1)
            elif interval == "PT15M":
                current += timedelta(minutes=15)
            elif interval == "P1D":
                current += timedelta(days=1)
            else:
                current += timedelta(hours=1)
        
        return records
    
    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def create_weather_client(
    username: str = "",
    password: str = "",
    use_mock: bool = False,
) -> BaseAPIClient:
    """
    Create a weather API client.
    
    Args:
        username: Meteomatics API username
        password: Meteomatics API password
        use_mock: If True, return a mock client for testing
    
    Returns:
        MeteomaticsClient or MockMeteomaticsClient
    """
    if use_mock or not username or not password:
        logger.info("Using mock weather client")
        return MockMeteomaticsClient()
    return MeteomaticsClient(username=username, password=password)
