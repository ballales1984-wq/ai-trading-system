"""
Modular Predictive Multi-Source Engine
======================================

A comprehensive modular engine for generating composite signals from multiple
data sources including weather, crypto, energy, traffic, events, and crime data.

Architecture:
- Temporal Modules: Day/Night, Week, Season, Year, Weather
- API Modules: Open-Meteo, CoinGecko, EIA, Traffic APIs
- Composite Engine: Weighted signal aggregation
- Timeline Generation: Daily signals for 50 world cities

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import requests
from abc import ABC, abstractmethod
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class City:
    """Represents a city with geographic and economic data."""
    name: str
    country: str
    lat: float
    lon: float
    population_m: float  # millions
    gdp_billion_usd: float
    economic_importance: float  # 0-1 normalized
    electricity_consumption_per_capita_kwh: float = 5000.0
    cash_circulating_billion_usd: float = 0.0
    electronic_money_billion_usd: float = 0.0
    tax_evasion_factor: float = 0.1
    criminal_low_income_factor: float = 0.02
    criminal_high_income_factor: float = 0.01


@dataclass
class SignalResult:
    """Result of a signal calculation."""
    timestamp: datetime
    city: str
    composite_signal: float
    time_signal: float = 0.0
    weather_signal: float = 0.0
    economic_signal: float = 0.0
    social_signal: float = 0.0
    crime_signal: float = 0.0
    crypto_signal: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TOP 50 WORLD CITIES DATA
# =============================================================================

TOP_50_CITIES = [
    # North America
    City("New York", "USA", 40.7128, -74.0060, 8.5, 1700, 0.95, 4800, 2000, 5000, 0.08, 0.015, 0.01),
    City("Los Angeles", "USA", 34.0522, -118.2437, 4.0, 1000, 0.85, 4500, 800, 2000, 0.07, 0.02, 0.012),
    City("Chicago", "USA", 41.8781, -87.6298, 2.7, 700, 0.80, 5200, 500, 1200, 0.07, 0.025, 0.015),
    City("Houston", "USA", 29.7604, -95.3698, 2.3, 550, 0.75, 6000, 400, 1000, 0.06, 0.018, 0.01),
    City("Miami", "USA", 25.7617, -80.1918, 2.1, 400, 0.70, 5500, 350, 900, 0.09, 0.022, 0.014),
    City("Toronto", "Canada", 43.6532, -79.3832, 6.0, 380, 0.78, 4600, 300, 800, 0.06, 0.012, 0.008),
    City("Mexico City", "Mexico", 19.4326, -99.1332, 21.0, 250, 0.65, 2200, 150, 400, 0.25, 0.05, 0.03),
    
    # South America
    City("São Paulo", "Brazil", -23.5505, -46.6333, 12.3, 350, 0.70, 2800, 120, 350, 0.28, 0.06, 0.035),
    City("Buenos Aires", "Argentina", -34.6037, -58.3816, 15.0, 180, 0.60, 3200, 80, 250, 0.30, 0.055, 0.032),
    City("Rio de Janeiro", "Brazil", -22.9068, -43.1729, 6.7, 150, 0.55, 2600, 70, 200, 0.30, 0.07, 0.04),
    
    # Europe
    City("London", "UK", 51.5074, -0.1278, 9.0, 1100, 0.92, 4200, 1800, 4500, 0.05, 0.012, 0.008),
    City("Paris", "France", 48.8566, 2.3522, 2.2, 850, 0.90, 4100, 600, 1500, 0.06, 0.01, 0.007),
    City("Berlin", "Germany", 52.5200, 13.4050, 3.6, 450, 0.82, 4400, 400, 1000, 0.05, 0.008, 0.006),
    City("Madrid", "Spain", 40.4168, -3.7038, 3.2, 300, 0.75, 3800, 250, 700, 0.12, 0.018, 0.012),
    City("Rome", "Italy", 41.9028, 12.4964, 2.8, 200, 0.72, 3600, 200, 550, 0.18, 0.025, 0.018),
    City("Milan", "Italy", 45.4642, 9.1900, 3.2, 350, 0.78, 4000, 280, 750, 0.16, 0.02, 0.015),
    City("Amsterdam", "Netherlands", 52.3676, 4.9041, 1.1, 250, 0.80, 4500, 180, 500, 0.04, 0.008, 0.006),
    City("Frankfurt", "Germany", 50.1109, 8.6821, 1.8, 400, 0.85, 4800, 350, 900, 0.04, 0.007, 0.005),
    City("Moscow", "Russia", 55.7558, 37.6173, 12.5, 300, 0.70, 3800, 150, 400, 0.20, 0.06, 0.04),
    
    # Asia
    City("Tokyo", "Japan", 35.6762, 139.6503, 14.0, 1800, 0.95, 5200, 2500, 6000, 0.03, 0.005, 0.004),
    City("Osaka", "Japan", 34.6937, 135.5023, 8.8, 600, 0.80, 4800, 800, 2000, 0.03, 0.006, 0.005),
    City("Shanghai", "China", 31.2304, 121.4737, 24.0, 1200, 0.88, 4200, 500, 1500, 0.15, 0.03, 0.02),
    City("Beijing", "China", 39.9042, 116.4074, 21.5, 900, 0.85, 4000, 400, 1200, 0.15, 0.028, 0.018),
    City("Shenzhen", "China", 22.5431, 114.0579, 12.5, 550, 0.80, 4500, 300, 1000, 0.12, 0.025, 0.016),
    City("Hong Kong", "China", 22.3193, 114.1694, 7.5, 450, 0.90, 5500, 600, 1800, 0.05, 0.012, 0.008),
    City("Singapore", "Singapore", 1.3521, 103.8198, 5.7, 400, 0.88, 5800, 500, 1500, 0.03, 0.008, 0.005),
    City("Seoul", "South Korea", 37.5665, 126.9780, 9.7, 600, 0.82, 5000, 400, 1200, 0.06, 0.015, 0.01),
    City("Mumbai", "India", 19.0760, 72.8777, 20.7, 200, 0.65, 1200, 80, 250, 0.30, 0.08, 0.05),
    City("Delhi", "India", 28.7041, 77.1025, 16.8, 180, 0.62, 1100, 70, 200, 0.32, 0.085, 0.055),
    City("Bangalore", "India", 12.9716, 77.5946, 8.4, 120, 0.58, 1400, 50, 150, 0.25, 0.06, 0.04),
    City("Bangkok", "Thailand", 13.7563, 100.5018, 10.7, 150, 0.60, 2800, 100, 350, 0.18, 0.04, 0.028),
    City("Jakarta", "Indonesia", -6.2088, 106.8456, 10.5, 180, 0.58, 1500, 80, 250, 0.25, 0.055, 0.035),
    City("Manila", "Philippines", 14.5995, 120.9842, 13.9, 120, 0.52, 1200, 60, 180, 0.28, 0.06, 0.04),
    City("Kuala Lumpur", "Malaysia", 3.1390, 101.6869, 7.6, 150, 0.62, 3500, 120, 400, 0.15, 0.03, 0.02),
    City("Taipei", "Taiwan", 25.0330, 121.5654, 7.0, 200, 0.72, 4800, 180, 550, 0.06, 0.015, 0.01),
    
    # Middle East
    City("Dubai", "UAE", 25.2048, 55.2708, 3.4, 300, 0.80, 7500, 200, 800, 0.02, 0.01, 0.008),
    City("Tel Aviv", "Israel", 32.0853, 34.7818, 2.5, 150, 0.72, 4200, 100, 350, 0.05, 0.012, 0.008),
    City("Istanbul", "Turkey", 41.0082, 28.9784, 15.5, 200, 0.65, 2800, 120, 400, 0.18, 0.04, 0.028),
    
    # Africa
    City("Cairo", "Egypt", 30.0444, 31.2357, 20.5, 100, 0.50, 1500, 40, 150, 0.35, 0.07, 0.045),
    City("Johannesburg", "South Africa", -26.2041, 28.0473, 5.8, 120, 0.55, 3200, 50, 200, 0.25, 0.06, 0.04),
    City("Lagos", "Nigeria", 6.5244, 3.3792, 15.4, 80, 0.45, 800, 30, 100, 0.40, 0.09, 0.06),
    
    # Oceania
    City("Sydney", "Australia", -33.8688, 151.2093, 5.3, 400, 0.82, 5800, 350, 1100, 0.04, 0.01, 0.007),
    City("Melbourne", "Australia", -37.8136, 144.9631, 5.0, 350, 0.78, 5500, 300, 950, 0.04, 0.01, 0.007),
    
    # Additional major cities
    City("San Francisco", "USA", 37.7749, -122.4194, 0.9, 600, 0.88, 4500, 500, 1500, 0.06, 0.015, 0.01),
    City("Seattle", "USA", 47.6062, -122.3321, 0.8, 400, 0.80, 4800, 300, 900, 0.05, 0.014, 0.009),
    City("Boston", "USA", 42.3601, -71.0589, 0.7, 380, 0.82, 4600, 280, 850, 0.05, 0.012, 0.008),
    City("Philadelphia", "USA", 39.9526, -75.1652, 1.6, 320, 0.72, 4400, 250, 750, 0.06, 0.018, 0.012),
    City("Atlanta", "USA", 33.7490, -84.3880, 2.2, 350, 0.70, 5200, 200, 650, 0.07, 0.02, 0.014),
]


# =============================================================================
# TEMPORAL MODULES
# =============================================================================

class TemporalModule(ABC):
    """Abstract base class for temporal modules."""
    
    @abstractmethod
    def calculate(self, dt: datetime) -> float:
        """Calculate the temporal factor for a given datetime."""
        pass


class DayNightModule(TemporalModule):
    """
    Day/Night factor module.
    
    Determines activity levels based on time of day:
    - Day (6-18): Normal activity (1.0)
    - Evening (18-22): Peak activity (1.2)
    - Night (22-6): Low activity (0.8)
    """
    
    def calculate(self, dt: datetime) -> float:
        hour = dt.hour
        if 6 <= hour < 18:
            return 1.0  # Day - normal
        elif 18 <= hour < 22:
            return 1.2  # Evening - peak
        else:
            return 0.8  # Night - low


class WeekModule(TemporalModule):
    """
    Week day factor module.
    
    Weekend days have higher activity:
    - Monday: 0.95
    - Tuesday: 0.90
    - Wednesday: 0.90
    - Thursday: 0.95
    - Friday: 1.10
    - Saturday: 1.40
    - Sunday: 1.30
    """
    
    WEEKDAY_FACTORS = {
        "Monday": 0.95,
        "Tuesday": 0.90,
        "Wednesday": 0.90,
        "Thursday": 0.95,
        "Friday": 1.10,
        "Saturday": 1.40,
        "Sunday": 1.30
    }
    
    def calculate(self, dt: datetime) -> float:
        day_name = dt.strftime("%A")
        return self.WEEKDAY_FACTORS.get(day_name, 1.0)


class SeasonModule(TemporalModule):
    """
    Season factor module.
    
    Affects consumption patterns:
    - Winter (Dec-Feb): 0.9 (lower retail, higher energy)
    - Spring (Mar-May): 1.1 (increased activity)
    - Summer (Jun-Aug): 1.2 (peak activity)
    - Autumn (Sep-Nov): 1.0 (normal)
    """
    
    def calculate(self, dt: datetime) -> float:
        month = dt.month
        if month in [12, 1, 2]:
            return 0.9  # Winter
        elif month in [3, 4, 5]:
            return 1.1  # Spring
        elif month in [6, 7, 8]:
            return 1.2  # Summer
        else:
            return 1.0  # Autumn


class YearModule(TemporalModule):
    """
    Year factor module for special events and holidays.
    
    Major holidays have significant impact on consumption.
    """
    
    SPECIAL_EVENTS = {
        # Format: "MM-DD": factor
        "01-01": 1.3,   # New Year
        "02-14": 1.2,   # Valentine's Day
        "03-08": 1.1,   # International Women's Day
        "04-01": 1.05,  # April Fools
        "05-01": 1.1,   # Labor Day
        "06-21": 1.1,   # Summer Solstice
        "10-31": 1.25,  # Halloween
        "11-11": 1.3,   # Singles Day (big in Asia)
        "11-26": 1.4,   # Thanksgiving (US)
        "12-24": 1.5,   # Christmas Eve
        "12-25": 1.6,   # Christmas
        "12-31": 1.4,   # New Year's Eve
    }
    
    def calculate(self, dt: datetime) -> float:
        key = dt.strftime("%m-%d")
        return self.SPECIAL_EVENTS.get(key, 1.0)


class WeatherModule:
    """
    Weather factor module.
    
    Calculates impact based on temperature, rain, and wind.
    """
    
    def calculate(self, temp: float, rain: bool, wind: bool) -> float:
        """
        Calculate weather factor.
        
        Args:
            temp: Temperature in Celsius
            rain: Whether it's raining
            wind: Whether it's windy (>20 m/s)
        
        Returns:
            Weather factor (0.5 - 1.3)
        """
        factor = 1.0
        
        # Temperature impact
        if temp < 5:
            factor *= 0.9  # Cold - less outdoor activity
        elif temp > 35:
            factor *= 0.85  # Extreme heat
        elif temp > 25:
            factor *= 1.15  # Warm - more activity
        elif 15 <= temp <= 25:
            factor *= 1.1  # Pleasant weather
        
        # Rain impact
        if rain:
            factor *= 0.85  # Rain reduces physical activity
        
        # Wind impact
        if wind:
            factor *= 0.9  # Wind reduces activity
        
        return factor


# =============================================================================
# API MODULES
# =============================================================================

class APIModule(ABC):
    """Abstract base class for API modules."""
    
    @abstractmethod
    def fetch(self, city: City, dt: datetime) -> Dict[str, Any]:
        """Fetch data from API."""
        pass
    
    @abstractmethod
    def normalize(self, data: Dict[str, Any]) -> float:
        """Normalize API data to 0-1 scale."""
        pass


class OpenMeteoModule(APIModule):
    """
    Open-Meteo API module for weather data.
    
    Free API, no key required.
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def fetch(self, city: City, dt: datetime) -> Dict[str, Any]:
        """Fetch weather data from Open-Meteo."""
        try:
            url = f"{self.BASE_URL}?latitude={city.lat}&longitude={city.lon}&daily=temperature_2m_max,precipitation_sum,windspeed_10m_max&timezone=auto"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get('daily', {})
            return {
                "temp": daily.get('temperature_2m_max', [20])[0],
                "precipitation": daily.get('precipitation_sum', [0])[0],
                "windspeed": daily.get('windspeed_10m_max', [0])[0],
                "success": True
            }
        except Exception as e:
            logger.warning(f"Open-Meteo API error for {city.name}: {e}")
            return {"temp": 20, "precipitation": 0, "windspeed": 5, "success": False}
    
    def normalize(self, data: Dict[str, Any]) -> float:
        """Normalize weather data to 0-1 scale."""
        temp = data.get("temp", 20)
        precip = data.get("precipitation", 0)
        wind = data.get("windspeed", 5)
        
        # Temperature normalization (-30 to 50 range)
        temp_norm = np.clip((temp + 30) / 80, 0, 1)
        
        # Precipitation normalization (0 to 50mm)
        precip_norm = np.clip(precip / 50, 0, 1)
        
        # Wind normalization (0 to 30 m/s)
        wind_norm = np.clip(wind / 30, 0, 1)
        
        # Weighted average
        return 0.5 * temp_norm + 0.3 * precip_norm + 0.2 * wind_norm


class CoinGeckoModule(APIModule):
    """
    CoinGecko API module for crypto data.
    
    Free API with rate limits.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self._cache = {}
        self._cache_time = 0
    
    def fetch(self, city: City, dt: datetime) -> Dict[str, Any]:
        """Fetch crypto data from CoinGecko."""
        # Use cache if less than 1 hour old
        current_time = time.time()
        if self._cache and (current_time - self._cache_time) < 3600:
            return self._cache
        
        try:
            url = f"{self.BASE_URL}/coins/bitcoin/market_chart?vs_currency=usd&days=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if prices and volumes:
                price_change = prices[-1][1] - prices[0][1] if len(prices) > 1 else 0
                total_volume = sum(v[1] for v in volumes)
                
                result = {
                    "price_change": price_change,
                    "total_volume": total_volume,
                    "success": True
                }
                self._cache = result
                self._cache_time = current_time
                return result
            
        except Exception as e:
            logger.warning(f"CoinGecko API error: {e}")
        
        return {"price_change": 0, "total_volume": 1e9, "success": False}
    
    def normalize(self, data: Dict[str, Any]) -> float:
        """Normalize crypto data to 0-1 scale."""
        price_change = data.get("price_change", 0)
        volume = data.get("total_volume", 1e9)
        
        # Price change normalization (tanh for bounded output)
        change_norm = (np.tanh(price_change / 1000) + 1) / 2
        
        # Volume normalization (0 to 50 billion)
        volume_norm = np.clip(volume / 5e10, 0, 1)
        
        return 0.6 * change_norm + 0.4 * volume_norm


class EnergyModule(APIModule):
    """
    Energy data module.
    
    Uses EIA API for US data, simulated for others.
    """
    
    def __init__(self, eia_api_key: Optional[str] = None):
        self.eia_api_key = eia_api_key
    
    def fetch(self, city: City, dt: datetime) -> Dict[str, Any]:
        """Fetch energy data."""
        # Simulated data for now
        # In production, would call EIA API for US cities
        base_consumption = city.electricity_consumption_per_capita_kwh
        
        # Seasonal variation
        month = dt.month
        if month in [12, 1, 2, 6, 7, 8]:  # Winter and Summer peaks
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        return {
            "consumption": base_consumption * multiplier,
            "price": np.random.uniform(40, 120),  # Simulated price
            "success": True
        }
    
    def normalize(self, data: Dict[str, Any]) -> float:
        """Normalize energy data to 0-1 scale."""
        consumption = data.get("consumption", 5000)
        price = data.get("price", 80)
        
        # Consumption normalization (0 to 10000 kWh)
        consumption_norm = np.clip(consumption / 10000, 0, 1)
        
        # Price normalization (20 to 200)
        price_norm = np.clip((price - 20) / 180, 0, 1)
        
        return 0.6 * consumption_norm + 0.4 * price_norm


class TrafficModule(APIModule):
    """
    Traffic data module.
    
    Simulated traffic index (placeholder for Google Maps/TomTom API).
    """
    
    def fetch(self, city: City, dt: datetime) -> Dict[str, Any]:
        """Fetch traffic data (simulated)."""
        # Base traffic based on population
        base_traffic = city.population_m / 25  # Normalize by max population
        
        # Day of week effect
        weekday = dt.weekday()
        if weekday < 5:
            day_factor = 0.8  # Weekday
        else:
            day_factor = 1.2  # Weekend
        
        # Hour effect
        hour = dt.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            hour_factor = 1.3  # Rush hour
        elif 22 <= hour or hour <= 5:
            hour_factor = 0.3  # Night
        else:
            hour_factor = 1.0
        
        traffic_index = base_traffic * day_factor * hour_factor
        
        return {
            "traffic_index": np.clip(traffic_index, 0, 1),
            "success": True
        }
    
    def normalize(self, data: Dict[str, Any]) -> float:
        """Normalize traffic data."""
        return data.get("traffic_index", 0.5)


class EventsModule(APIModule):
    """
    Events data module.
    
    Simulated events index based on calendar.
    """
    
    def fetch(self, city: City, dt: datetime) -> Dict[str, Any]:
        """Fetch events data (simulated)."""
        weekday = dt.weekday()
        
        # Weekend has more events
        if weekday >= 5:
            base = 0.7
        else:
            base = 0.3
        
        # Add some randomness
        events_index = base + np.random.uniform(-0.1, 0.1)
        
        return {
            "events_index": np.clip(events_index, 0, 1),
            "success": True
        }
    
    def normalize(self, data: Dict[str, Any]) -> float:
        """Normalize events data."""
        return data.get("events_index", 0.5)


class CrimeModule(APIModule):
    """
    Crime data module.
    
    Simulated crime index based on city factors.
    """
    
    def fetch(self, city: City, dt: datetime) -> Dict[str, Any]:
        """Fetch crime data (simulated)."""
        # Base crime from city factors
        low_income_crime = city.criminal_low_income_factor
        high_income_crime = city.criminal_high_income_factor
        
        # Weighted average
        crime_index = 0.6 * low_income_crime + 0.4 * high_income_crime
        
        # Add temporal variation
        hour = dt.hour
        if 22 <= hour or hour <= 5:
            crime_index *= 1.3  # More crime at night
        elif 6 <= hour <= 18:
            crime_index *= 0.8  # Less crime during day
        
        return {
            "crime_index": np.clip(crime_index, 0, 1),
            "success": True
        }
    
    def normalize(self, data: Dict[str, Any]) -> float:
        """Normalize crime data (inverse - lower crime = higher signal)."""
        crime = data.get("crime_index", 0.5)
        return max(0.5, 1 - crime)


# =============================================================================
# COMPOSITE SIGNAL ENGINE
# =============================================================================

class CompositeSignalEngine:
    """
    Composite Signal Engine.
    
    Combines multiple signal sources into a unified composite signal.
    """
    
    DEFAULT_WEIGHTS = {
        "time": 0.25,
        "weather": 0.15,
        "economic": 0.20,
        "social": 0.15,
        "crime": 0.10,
        "crypto": 0.15
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_real_apis: bool = True,
        eia_api_key: Optional[str] = None
    ):
        """
        Initialize the composite signal engine.
        
        Args:
            weights: Custom weights for signal components
            use_real_apis: Whether to use real API calls
            eia_api_key: EIA API key for energy data
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.use_real_apis = use_real_apis
        
        # Initialize temporal modules
        self.day_night = DayNightModule()
        self.week = WeekModule()
        self.season = SeasonModule()
        self.year = YearModule()
        self.weather = WeatherModule()
        
        # Initialize API modules
        if use_real_apis:
            self.weather_api = OpenMeteoModule()
            self.crypto_api = CoinGeckoModule()
        else:
            self.weather_api = None
            self.crypto_api = None
        
        self.energy_api = EnergyModule(eia_api_key)
        self.traffic_api = TrafficModule()
        self.events_api = EventsModule()
        self.crime_api = CrimeModule()
    
    def calculate_time_signal(self, dt: datetime) -> float:
        """Calculate combined temporal signal."""
        return (
            self.day_night.calculate(dt) *
            self.week.calculate(dt) *
            self.season.calculate(dt) *
            self.year.calculate(dt)
        )
    
    def calculate_weather_signal(
        self,
        city: City,
        dt: datetime,
        use_api: bool = True
    ) -> tuple:
        """Calculate weather signal."""
        if use_api and self.weather_api:
            data = self.weather_api.fetch(city, dt)
            signal = self.weather_api.normalize(data)
            raw = data
        else:
            # Simulated weather
            temp = np.random.uniform(-5, 35)
            rain = np.random.random() > 0.8
            wind = np.random.random() > 0.9
            signal = self.weather.calculate(temp, rain, wind)
            raw = {"temp": temp, "rain": rain, "wind": wind}
        
        return signal, raw
    
    def calculate_economic_signal(self, city: City, dt: datetime) -> tuple:
        """Calculate economic signal."""
        energy_data = self.energy_api.fetch(city, dt)
        energy_signal = self.energy_api.normalize(energy_data)
        
        # Commodity placeholder
        commodity_signal = np.random.uniform(0.3, 0.7)
        
        signal = (energy_signal + commodity_signal) / 2
        return signal, energy_data
    
    def calculate_social_signal(self, city: City, dt: datetime) -> tuple:
        """Calculate social signal."""
        traffic_data = self.traffic_api.fetch(city, dt)
        traffic_signal = self.traffic_api.normalize(traffic_data)
        
        events_data = self.events_api.fetch(city, dt)
        events_signal = self.events_api.normalize(events_data)
        
        signal = (traffic_signal + events_signal) / 2
        return signal, {"traffic": traffic_data, "events": events_data}
    
    def calculate_crime_signal(self, city: City, dt: datetime) -> tuple:
        """Calculate crime signal."""
        data = self.crime_api.fetch(city, dt)
        signal = self.crime_api.normalize(data)
        return signal, data
    
    def calculate_crypto_signal(self, city: City, dt: datetime, use_api: bool = True) -> tuple:
        """Calculate crypto signal."""
        if use_api and self.crypto_api:
            data = self.crypto_api.fetch(city, dt)
            signal = self.crypto_api.normalize(data)
        else:
            signal = np.random.uniform(0.3, 0.7)
            data = {"simulated": True}
        
        return signal, data
    
    def calculate_composite(
        self,
        city: City,
        dt: datetime,
        use_api: bool = True
    ) -> SignalResult:
        """
        Calculate composite signal for a city at a given time.
        
        Args:
            city: City object
            dt: Datetime
            use_api: Whether to use real API calls
        
        Returns:
            SignalResult with all signal components
        """
        # Calculate individual signals
        time_signal = self.calculate_time_signal(dt)
        weather_signal, weather_raw = self.calculate_weather_signal(city, dt, use_api)
        economic_signal, economic_raw = self.calculate_economic_signal(city, dt)
        social_signal, social_raw = self.calculate_social_signal(city, dt)
        crime_signal, crime_raw = self.calculate_crime_signal(city, dt)
        crypto_signal, crypto_raw = self.calculate_crypto_signal(city, dt, use_api)
        
        # Calculate weighted composite
        composite = (
            self.weights["time"] * time_signal +
            self.weights["weather"] * weather_signal +
            self.weights["economic"] * economic_signal +
            self.weights["social"] * social_signal +
            self.weights["crime"] * crime_signal +
            self.weights["crypto"] * crypto_signal
        )
        
        # Normalize to 0-1
        composite = np.clip(composite / 1.5, 0, 1)  # Max theoretical is ~1.5
        
        return SignalResult(
            timestamp=dt,
            city=city.name,
            composite_signal=round(composite, 4),
            time_signal=round(time_signal, 4),
            weather_signal=round(weather_signal, 4),
            economic_signal=round(economic_signal, 4),
            social_signal=round(social_signal, 4),
            crime_signal=round(crime_signal, 4),
            crypto_signal=round(crypto_signal, 4),
            raw_data={
                "weather": weather_raw,
                "economic": economic_raw,
                "social": social_raw,
                "crime": crime_raw,
                "crypto": crypto_raw
            }
        )


# =============================================================================
# TIMELINE GENERATOR
# =============================================================================

class TimelineGenerator:
    """
    Timeline Generator.
    
    Generates daily signals for multiple cities over a time period.
    """
    
    def __init__(
        self,
        engine: Optional[CompositeSignalEngine] = None,
        cities: Optional[List[City]] = None
    ):
        """
        Initialize timeline generator.
        
        Args:
            engine: CompositeSignalEngine instance
            cities: List of cities to process
        """
        self.engine = engine or CompositeSignalEngine()
        self.cities = cities or TOP_50_CITIES
    
    def generate(
        self,
        start_date: datetime,
        end_date: datetime,
        use_api: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Generate timeline of signals.
        
        Args:
            start_date: Start date
            end_date: End date
            use_api: Whether to use real API calls
            progress_callback: Optional callback for progress updates
        
        Returns:
            DataFrame with timeline data
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        results = []
        total = len(self.cities) * len(dates)
        count = 0
        
        for city in self.cities:
            for date in dates:
                result = self.engine.calculate_composite(city, date, use_api)
                
                results.append({
                    "date": date,
                    "city": city.name,
                    "country": city.country,
                    "population_m": city.population_m,
                    "gdp_billion_usd": city.gdp_billion_usd,
                    "economic_importance": city.economic_importance,
                    "composite_signal": result.composite_signal,
                    "time_signal": result.time_signal,
                    "weather_signal": result.weather_signal,
                    "economic_signal": result.economic_signal,
                    "social_signal": result.social_signal,
                    "crime_signal": result.crime_signal,
                    "crypto_signal": result.crypto_signal
                })
                
                count += 1
                if progress_callback and count % 10 == 0:
                    progress_callback(count, total)
        
        df = pd.DataFrame(results)
        return df
    
    def generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics by city."""
        summary = df.groupby('city').agg({
            'composite_signal': ['mean', 'std', 'min', 'max'],
            'weather_signal': 'mean',
            'economic_signal': 'mean',
            'social_signal': 'mean',
            'crime_signal': 'mean',
            'crypto_signal': 'mean',
            'population_m': 'first',
            'gdp_billion_usd': 'first',
            'economic_importance': 'first'
        }).round(4)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        return summary.reset_index()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("Modular Predictive Multi-Source Engine")
    print("=" * 60)
    
    # Initialize engine
    engine = CompositeSignalEngine(use_real_apis=True)
    
    # Initialize generator
    generator = TimelineGenerator(engine=engine, cities=TOP_50_CITIES[:10])  # First 10 cities for demo
    
    # Generate timeline for 1 week
    start_date = datetime(2026, 2, 1)
    end_date = datetime(2026, 2, 7)
    
    print(f"\nGenerating timeline from {start_date.date()} to {end_date.date()}...")
    print(f"Cities: {len(generator.cities)}")
    
    def progress(current, total):
        pct = (current / total) * 100
        print(f"\rProgress: {pct:.1f}% ({current}/{total})", end="", flush=True)
    
    df = generator.generate(start_date, end_date, use_api=True, progress_callback=progress)
    print("\n")
    
    # Display sample
    print("\nSample Data:")
    print(df.head(10).to_string())
    
    # Generate summary
    print("\n\nSummary Statistics:")
    summary = generator.generate_summary(df)
    print(summary.to_string())
    
    # Save to CSV
    output_file = "city_signals_timeline.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nTimeline saved to: {output_file}")
    
    # Save summary
    summary_file = "city_signals_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    return df, summary


if __name__ == "__main__":
    main()
