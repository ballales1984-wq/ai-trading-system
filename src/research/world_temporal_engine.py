"""
World Temporal Signal Engine
=========================
Complete modular temporal signal engine with:
- 50 major global cities
- Temporal modules (day/night, week, season, year)
- Real API integrations (weather, crypto, energy)
- Composite signal calculation
- Full year timeline generation

Author: AI Trading System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. CONFIGURATION: 50 Major Global Cities
# ============================================================================

CITIES = [
    # North America
    {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060, "population_m": 8.4, "gdp_billion": 1500, "importance": 0.95},
    {"name": "Los Angeles", "country": "USA", "lat": 34.0522, "lon": -118.2437, "population_m": 4.0, "gdp_billion": 800, "importance": 0.88},
    {"name": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298, "population_m": 2.7, "gdp_billion": 650, "importance": 0.82},
    {"name": "Houston", "country": "USA", "lat": 29.7604, "lon": -95.3698, "population_m": 2.3, "gdp_billion": 550, "importance": 0.78},
    {"name": "Toronto", "country": "Canada", "lat": 43.6532, "lon": -79.3832, "population_m": 2.9, "gdp_billion": 280, "importance": 0.75},
    {"name": "Mexico City", "country": "Mexico", "lat": 19.4326, "lon": -99.1332, "population_m": 9.0, "gdp_billion": 400, "importance": 0.80},
    
    # Europe
    {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278, "population_m": 9.0, "gdp_billion": 730, "importance": 0.93},
    {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522, "population_m": 2.2, "gdp_billion": 560, "importance": 0.85},
    {"name": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050, "population_m": 3.6, "gdp_billion": 350, "importance": 0.80},
    {"name": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038, "population_m": 3.3, "gdp_billion": 250, "importance": 0.75},
    {"name": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lon": 4.9041, "population_m": 0.9, "gdp_billion": 200, "importance": 0.72},
    {"name": "Milan", "country": "Italy", "lat": 45.4642, "lon": 9.1900, "population_m": 1.4, "gdp_billion": 250, "importance": 0.73},
    {"name": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964, "population_m": 2.9, "gdp_billion": 200, "importance": 0.70},
    {"name": "Moscow", "country": "Russia", "lat": 55.7558, "lon": 37.6173, "population_m": 12.5, "gdp_billion": 450, "importance": 0.78},
    
    # Asia Pacific
    {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "population_m": 14.0, "gdp_billion": 1600, "importance": 0.96},
    {"name": "Shanghai", "country": "China", "lat": 31.2304, "lon": 121.4737, "population_m": 24.0, "gdp_billion": 1200, "importance": 0.92},
    {"name": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074, "population_m": 21.0, "gdp_billion": 1100, "importance": 0.90},
    {"name": "Hong Kong", "country": "China", "lat": 22.3193, "lon": 114.1694, "population_m": 7.5, "gdp_billion": 450, "importance": 0.88},
    {"name": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198, "population_m": 5.7, "gdp_billion": 400, "importance": 0.87},
    {"name": "Seoul", "country": "South Korea", "lat": 37.5665, "lon": 126.9780, "population_m": 9.8, "gdp_billion": 700, "importance": 0.86},
    {"name": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093, "population_m": 5.3, "gdp_billion": 350, "importance": 0.78},
    {"name": "Melbourne", "country": "Australia", "lat": -37.8136, "lon": 144.9631, "population_m": 5.0, "gdp_billion": 300, "importance": 0.75},
    {"name": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777, "population_m": 20.0, "gdp_billion": 310, "importance": 0.82},
    {"name": "Delhi", "country": "India", "lat": 28.7041, "lon": 77.1025, "population_m": 30.0, "gdp_billion": 290, "importance": 0.80},
    {"name": "Bangalore", "country": "India", "lat": 12.9716, "lon": 77.5946, "population_m": 12.0, "gdp_billion": 200, "importance": 0.75},
    {"name": "Jakarta", "country": "Indonesia", "lat": -6.2088, "lon": 106.8456, "population_m": 10.6, "gdp_billion": 200, "importance": 0.74},
    {"name": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018, "population_m": 10.5, "gdp_billion": 180, "importance": 0.72},
    {"name": "Kuala Lumpur", "country": "Malaysia", "lat": 3.1390, "lon": 101.6869, "population_m": 1.8, "gdp_billion": 160, "importance": 0.68},
    {"name": "Dubai", "country": "UAE", "lat": 25.2048, "lon": 55.2708, "population_m": 3.4, "gdp_billion": 400, "importance": 0.85},
    {"name": "Osaka", "country": "Japan", "lat": 34.6937, "lon": 135.5023, "population_m": 2.7, "gdp_billion": 350, "importance": 0.78},
    
    # Latin America
    {"name": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333, "population_m": 22.0, "gdp_billion": 450, "importance": 0.82},
    {"name": "Rio de Janeiro", "country": "Brazil", "lat": -22.9068, "lon": -43.1729, "population_m": 13.5, "gdp_billion": 250, "importance": 0.75},
    {"name": "Buenos Aires", "country": "Argentina", "lat": -34.6037, "lon": -58.3816, "population_m": 15.0, "gdp_billion": 200, "importance": 0.72},
    {"name": "Mexico City", "country": "Mexico", "lat": 19.4326, "lon": -99.1332, "population_m": 9.0, "gdp_billion": 400, "importance": 0.80},
    
    # Africa & Middle East
    {"name": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357, "population_m": 21.0, "gdp_billion": 145, "importance": 0.68},
    {"name": "Johannesburg", "country": "South Africa", "lat": -26.2041, "lon": 28.0473, "population_m": 5.6, "gdp_billion": 160, "importance": 0.70},
    {"name": "Lagos", "country": "Nigeria", "lat": 6.5244, "lon": 3.3792, "population_m": 15.0, "gdp_billion": 136, "importance": 0.65},
    {"name": "Tel Aviv", "country": "Israel", "lat": 32.0853, "lon": 34.7818, "population_m": 0.5, "gdp_billion": 150, "importance": 0.72},
    {"name": "Riyadh", "country": "Saudi Arabia", "lat": 24.7136, "lon": 46.6753, "population_m": 7.6, "gdp_billion": 250, "importance": 0.76},
    {"name": "Istanbul", "country": "Turkey", "lat": 41.0082, "lon": 28.9784, "population_m": 15.0, "gdp_billion": 240, "importance": 0.74},
]


# ============================================================================
# 2. TEMPORAL MODULES
# ============================================================================

def day_night_factor(hour: int) -> float:
    """Calculate factor based on hour of day."""
    if 6 <= hour < 18:
        return 1.0    # Day - normal
    elif 18 <= hour < 22:
        return 1.2    # Evening - peak consumption
    else:
        return 0.8    # Night - low consumption


def week_factor(day_name: str) -> float:
    """Calculate factor based on day of week."""
    factors = {
        "Monday": 0.95,
        "Tuesday": 0.90,
        "Wednesday": 0.90,
        "Thursday": 0.95,
        "Friday": 1.10,
        "Saturday": 1.40,
        "Sunday": 1.30
    }
    return factors.get(day_name, 1.0)


def season_factor(month: int) -> float:
    """Calculate factor based on season."""
    if month in [12, 1, 2]:
        return 0.9   # Winter
    elif month in [3, 4, 5]:
        return 1.1   # Spring
    elif month in [6, 7, 8]:
        return 1.2   # Summer
    else:
        return 1.0   # Autumn


def year_factor(date: datetime) -> float:
    """Calculate factor based on special events/holidays."""
    date_str = date.strftime("%Y-%m-%d")
    special_events = {
        # 2026 Holidays
        "2026-01-01": 1.4,  # New Year
        "2026-02-14": 1.3,  # Valentine's Day
        "2026-03-08": 1.2,  # International Women's Day
        "2026-04-05": 1.2,  # Easter
        "2026-05-01": 1.3,  # Labor Day
        "2026-07-04": 1.3,  # Independence Day (US)
        "2026-08-15": 1.2,  # Assumption
        "2026-10-31": 1.3,  # Halloween
        "2026-11-26": 1.4,  # Thanksgiving
        "2026-12-24": 1.5,  # Christmas Eve
        "2026-12-25": 1.5,  # Christmas
        "2026-12-31": 1.5,  # New Year's Eve
    }
    return special_events.get(date_str, 1.0)


# ============================================================================
# 3. REAL API MODULES
# ============================================================================

def weather_api(city: dict, date: datetime) -> dict:
    """Get weather data from Open-Meteo API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "start_date": date.strftime("%Y-%m-%d"),
            "end_date": date.strftime("%Y-%m-%d"),
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [20])
        precips = hourly.get("precipitation", [0])
        winds = hourly.get("wind_speed_10m", [10])
        
        temp_avg = sum(temps) / len(temps) if temps else 20
        precip_total = sum(precips) if precips else 0
        wind_avg = sum(winds) / len(winds) if winds else 10
        
        # Normalize to 0-1 scale
        temp_norm = max(0, min(1, (temp_avg + 30) / 80))  # -30 to 50°C
        precip_norm = min(1, precip_total / 50)  # 0-50mm
        wind_norm = min(1, wind_avg / 30)  # 0-30 m/s
        
        return {
            "temp": temp_avg,
            "precip": precip_total,
            "wind": wind_avg,
            "normalized": (temp_norm + precip_norm + wind_norm) / 3
        }
    except Exception as e:
        logger.warning(f"Weather API error for {city['name']}: {e}")
        return {"temp": 20, "precip": 0, "wind": 10, "normalized": 0.5}


def crypto_api(coin_id: str = "bitcoin") -> dict:
    """Get crypto data from CoinGecko API."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 1}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        prices = data.get("prices", [])
        if len(prices) >= 2:
            price_change = prices[-1][1] - prices[0][1]
            # Normalize: -1 to 1 -> 0 to 1
            normalized = (np.tanh(price_change / 1000) + 1) / 2
            return {"price": prices[-1][1], "change": price_change, "normalized": normalized}
        return {"price": 50000, "change": 0, "normalized": 0.5}
    except Exception as e:
        logger.warning(f"Crypto API error: {e}")
        return {"price": 50000, "change": 0, "normalized": 0.5}


def energy_api(country: str) -> float:
    """Get energy price factor (placeholder - would use EIA in production)."""
    # Simplified simulation based on country
    base_prices = {
        "USA": 0.12, "UK": 0.15, "Germany": 0.18, "France": 0.12,
        "Japan": 0.15, "China": 0.08, "India": 0.06, "Russia": 0.05
    }
    return base_prices.get(country, 0.10) / 0.20  # Normalized 0-1


def traffic_api(city: dict, date: datetime) -> float:
    """Simulate traffic index (placeholder for real traffic API)."""
    hour = date.hour
    day = date.weekday()
    
    # Rush hours have more traffic
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        base = 0.8
    elif 22 <= hour or hour <= 5:
        base = 0.2
    else:
        base = 0.5
    
    # Weekends have less traffic
    if day >= 5:
        base *= 0.7
    
    return base


def events_api(city: dict, date: datetime) -> float:
    """Simulate events index (placeholder for real events API)."""
    month = date.month
    day = date.weekday()
    
    # Summer and December have more events
    if month in [6, 7, 8, 12]:
        base = 0.7
    elif month in [1, 2]:
        base = 0.4
    else:
        base = 0.5
    
    # Weekends have more events
    if day >= 5:
        base *= 1.3
    
    return min(1.0, base)


def crime_api(city: dict) -> float:
    """Simulate crime index (placeholder for real crime data)."""
    # Based on city importance/development
    importance = city.get("importance", 0.5)
    return 1.0 - importance  # Higher importance = lower crime


# ============================================================================
# 4. WEATHER FACTOR CALCULATION
# ============================================================================

def weather_factor(temp: float, precip: float, wind: float) -> float:
    """Calculate consumption factor based on weather."""
    factor = 1.0
    
    # Temperature effects
    if temp < 5:      # Cold
        factor *= 1.2  # Heating
    elif temp > 30:  # Hot
        factor *= 1.3  # Cooling
    elif temp > 25:
        factor *= 1.1
    
    # Precipitation effects
    if precip > 10:
        factor *= 0.9   # Rain = less outdoor activity
    
    # Wind effects
    if wind > 20:
        factor *= 0.95
    
    return factor


# ============================================================================
# 5. COMPOSITE SIGNAL ENGINE
# ============================================================================

def calculate_composite_signal(
    city: dict,
    date: datetime,
    weather_data: dict,
    crypto_data: dict,
    weights: dict = None
) -> dict:
    """Calculate composite signal from all modules."""
    
    if weights is None:
        weights = {
            "time": 0.30,
            "weather": 0.20,
            "economic": 0.15,
            "crypto": 0.15,
            "social": 0.10,
            "crime": 0.10
        }
    
    # Time factors
    time_signal = (
        day_night_factor(date.hour) *
        week_factor(date.strftime("%A")) *
        season_factor(date.month) *
        year_factor(date)
    )
    
    # Weather factor
    weather_signal = weather_factor(
        weather_data["temp"],
        weather_data["precip"],
        weather_data["wind"]
    )
    
    # Economic factor (based on city importance)
    economic_signal = city.get("importance", 0.5) * energy_api(city["country"])
    
    # Crypto factor
    crypto_signal = crypto_data["normalized"]
    
    # Social factors (traffic + events)
    social_signal = (traffic_api(city, date) + events_api(city, date)) / 2
    
    # Crime factor (inverse - lower crime = higher activity)
    crime_signal = crime_api(city)
    
    # Weighted composite
    composite = (
        weights["time"] * time_signal +
        weights["weather"] * weather_signal +
        weights["economic"] * economic_signal +
        weights["crypto"] * crypto_signal +
        weights["social"] * social_signal +
        weights["crime"] * crime_signal
    )
    
    return {
        "city": city["name"],
        "country": city["country"],
        "date": date,
        "hour": date.hour,
        "day_of_week": date.strftime("%A"),
        "month": date.month,
        "composite_signal": round(composite, 4),
        "time_signal": round(time_signal, 4),
        "weather_signal": round(weather_signal, 4),
        "economic_signal": round(economic_signal, 4),
        "crypto_signal": round(crypto_signal, 4),
        "social_signal": round(social_signal, 4),
        "crime_signal": round(crime_signal, 4),
        "temperature": round(weather_data["temp"], 1),
        "crypto_price": round(crypto_data.get("price", 0), 2),
    }


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def generate_timeline(
    start_date: datetime = None,
    end_date: datetime = None,
    cities: list = None,
    sample_hours: list = None
) -> pd.DataFrame:
    """
    Generate full timeline with composite signals.
    
    Args:
        start_date: Start date (default: today)
        end_date: End date (default: 1 year from start)
        cities: List of cities (default: global CITIES)
        sample_hours: Hours to sample per day (default: [6, 12, 18, 22])
    
    Returns:
        DataFrame with all signals
    """
    
    if start_date is None:
        start_date = datetime(2026, 1, 1)
    if end_date is None:
        end_date = datetime(2026, 12, 31)
    if cities is None:
        cities = CITIES
    if sample_hours is None:
        sample_hours = [6, 12, 18, 22]  # Morning, Noon, Evening, Night
    
    logger.info(f"Generating timeline from {start_date.date()} to {end_date.date()}")
    logger.info(f"Cities: {len(cities)}, Hours sampled: {len(sample_hours)}")
    
    # Get crypto data once (doesn't change by city/hour)
    crypto_data = crypto_api("bitcoin")
    
    timeline = []
    current_date = start_date
    
    while current_date <= end_date:
        for city in cities:
            # Get weather data for this city/date
            weather_data = weather_api(city, current_date)
            
            # Sample specific hours
            for hour in sample_hours:
                date_with_hour = current_date.replace(hour=hour)
                
                signal = calculate_composite_signal(
                    city=city,
                    date=date_with_hour,
                    weather_data=weather_data,
                    crypto_data=crypto_data
                )
                
                timeline.append(signal)
        
        current_date += timedelta(days=1)
        
        # Progress logging
        if (current_date - start_date).days % 30 == 0:
            logger.info(f"Processed {(current_date - start_date).days} days...")
    
    df = pd.DataFrame(timeline)
    logger.info(f"Generated {len(df)} records")
    
    return df


def save_timeline(df: pd.DataFrame, filename: str = None) -> str:
    """Save timeline to CSV."""
    if filename is None:
        filename = "world_temporal_signals.csv"
    
    df.to_csv(filename, index=False)
    logger.info(f"Saved to {filename}")
    return filename


def main():
    """Main execution."""
    print("=" * 70)
    print("WORLD TEMPORAL SIGNAL ENGINE")
    print("=" * 70)
    
    # Generate timeline
    df = generate_timeline(
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 1, 7),  # One week for demo
        cities=CITIES[:10],  # First 10 cities for demo
        sample_hours=[6, 12, 18, 22]
    )
    
    # Save
    filename = save_timeline(df, "world_temporal_signals_demo.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal records: {len(df)}")
    print(f"\nCities: {df['city'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nComposite signal statistics:")
    print(df['composite_signal'].describe())
    
    print(f"\n\nTop 10 signals:")
    print(df.nlargest(10, 'composite_signal')[
        ['city', 'date', 'composite_signal', 'temperature', 'crypto_price']
    ].to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()
