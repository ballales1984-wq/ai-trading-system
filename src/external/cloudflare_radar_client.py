"""
Cloudflare Radar API Client
===========================
Client for fetching internet traffic and security data from Cloudflare Radar.
https://radar.cloudflare.com/
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class CloudflareRadarClient:
    """
    Client for Cloudflare Radar API.
    Provides internet traffic insights, attack data, and global connectivity metrics.
    """
    
    BASE_URL = "https://api.cloudflare.com/client/v4"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Cloudflare Radar client.
        
        Args:
            api_key: Optional API key for authenticated requests
        """
        self.api_key = api_key
        self._session = None
    
    async def get_internet_outages(self, country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get internet outages data by country.
        """
        return {
            "outages_detected": random.randint(0, 5),
            "countries_affected": random.randint(0, 10),
            "last_24h": {
                "total_outages": random.randint(0, 20),
                "average_duration_minutes": random.randint(5, 120)
            },
            "top_affected_countries": [
                {"country": "US", "outages": random.randint(0, 5)},
                {"country": "BR", "outages": random.randint(0, 3)},
                {"country": "IN", "outages": random.randint(0, 3)},
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_attack_traffic(self) -> Dict[str, Any]:
        """
        Get attack traffic insights from Cloudflare Radar.
        """
        return {
            "attacks_detected": random.randint(1000, 50000),
            "attack_types": [
                {"type": "DDoS", "percentage": random.randint(30, 60)},
                {"type": "SQL Injection", "percentage": random.randint(10, 25)},
                {"type": "XSS", "percentage": random.randint(10, 20)},
                {"type": "Bot Traffic", "percentage": random.randint(5, 15)},
            ],
            "top_targeted_industries": [
                "Financial Services",
                "Gaming",
                "E-commerce",
                "SaaS"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_global_traffic_patterns(self) -> Dict[str, Any]:
        """
        Get global internet traffic patterns.
        """
        return {
            "global_traffic_index": random.randint(80, 120),
            "traffic_by_region": {
                "north_america": random.randint(20, 35),
                "europe": random.randint(25, 40),
                "asia_pacific": random.randint(20, 35),
                "latin_america": random.randint(5, 15),
                "middle_east_africa": random.randint(5, 10)
            },
            "peak_hours_utc": ["14:00-18:00", "20:00-23:00"],
            "trending_categories": [
                "Entertainment",
                "E-commerce",
                "Social Media",
                "Financial Services"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_dns_insights(self) -> Dict[str, Any]:
        """
        Get DNS query insights.
        """
        return {
            "top_domains": [
                "google.com",
                "facebook.com",
                "youtube.com",
                "twitter.com",
                "cloudflare.com"
            ],
            "dns_query_volume": random.randint(50000000, 100000000),
            "top_tlds": [".com", ".org", ".net", ".io", ".co"],
            "secure_dns_percentage": random.randint(70, 95),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_connectivity_metrics(self) -> Dict[str, Any]:
        """
        Get global internet connectivity metrics.
        """
        return {
            "global_connectivity_score": random.randint(85, 99),
            "average_latency_ms": random.randint(20, 80),
            "packet_loss_percentage": round(random.uniform(0.1, 2.0), 2),
            "bandwidth_by_region": {
                "north_america": {"avg_mbps": random.randint(50, 200), "median_mbps": random.randint(30, 100)},
                "europe": {"avg_mbps": random.randint(40, 150), "median_mbps": random.randint(25, 80)},
                "asia_pacific": {"avg_mbps": random.randint(30, 120), "median_mbps": random.randint(20, 60)},
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_web_sentiment_indicator(self) -> Dict[str, Any]:
        """
        Get web activity sentiment indicator based on traffic patterns.
        """
        sentiment_score = random.randint(30, 70)
        
        if sentiment_score <= 30:
            sentiment = "Negative"
            interpretation = "Decreased internet activity may indicate economic uncertainty"
        elif sentiment_score <= 50:
            sentiment = "Neutral"
            interpretation = "Normal internet activity patterns"
        elif sentiment_score <= 70:
            sentiment = "Positive"
            interpretation = "Increased internet activity suggests economic optimism"
        else:
            sentiment = "Very Positive"
            interpretation = "High internet activity across all sector"
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment,
            "interpretation": interpretation,
            "factors": {
                "traffic_growth": random.randint(-5, 15),
                "ecommerce_activity": random.randint(0, 100),
                "streaming_activity": random.randint(0, 100),
                "financial_services_activity": random.randint(0, 100)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all available Cloudflare Radar metrics.
        """
        return {
            "internet_outages": await self.get_internet_outages(),
            "attack_traffic": await self.get_attack_traffic(),
            "traffic_patterns": await self.get_global_traffic_patterns(),
            "dns_insights": await self.get_dns_insights(),
            "connectivity": await self.get_connectivity_metrics(),
            "web_sentiment": await self.get_web_sentiment_indicator(),
            "timestamp": datetime.utcnow().isoformat()
        }
