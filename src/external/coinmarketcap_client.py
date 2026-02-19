"""
src/external/coinmarketcap_client.py
CoinMarketCap API Client
=========================
Provides crypto market data, rankings, global metrics, and on-chain data
from CoinMarketCap Pro API.

Endpoints used:
  - /v1/cryptocurrency/listings/latest  → top coins by market cap
  - /v1/cryptocurrency/quotes/latest    → price/volume for specific coins
  - /v1/cryptocurrency/map              → symbol → CMC ID mapping
  - /v1/global-metrics/quotes/latest    → total market cap, BTC dominance
  - /v2/cryptocurrency/ohlcv/latest     → OHLCV data
  - /v1/cryptocurrency/trending/latest  → trending coins

API Docs: https://coinmarketcap.com/api/documentation/v1/
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://pro-api.coinmarketcap.com"


class CoinMarketCapClient:
    """
    CoinMarketCap Pro API client.

    Usage:
        client = CoinMarketCapClient(api_key="your_key")
        # or from env: client = CoinMarketCapClient()

        listings = client.get_listings(limit=100)
        btc_quote = client.get_quote("BTC")
        global_data = client.get_global_metrics()
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("COINMARKETCAP_API_KEY", "")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json",
        })

        if not self.api_key:
            logger.warning("COINMARKETCAP_API_KEY not set — API calls will fail")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a GET request to CMC API."""
        url = f"{BASE_URL}{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status", {}).get("error_code", 0) != 0:
                error_msg = data["status"].get("error_message", "Unknown error")
                logger.error(f"CMC API error: {error_msg}")
                return {"error": error_msg}

            return data.get("data", {})

        except requests.exceptions.HTTPError as e:
            logger.error(f"CMC HTTP error: {e}")
            return {"error": str(e)}
        except requests.exceptions.RequestException as e:
            logger.error(f"CMC request error: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Listings & Rankings
    # ------------------------------------------------------------------

    def get_listings(
        self,
        limit: int = 100,
        start: int = 1,
        sort: str = "market_cap",
        convert: str = "USD",
    ) -> List[Dict[str, Any]]:
        """
        Get top cryptocurrencies by market cap.

        Returns list of coins with price, volume, market cap, % changes.
        """
        data = self._get("/v1/cryptocurrency/listings/latest", {
            "limit": limit,
            "start": start,
            "sort": sort,
            "convert": convert,
        })

        if isinstance(data, dict) and "error" in data:
            return []

        if not isinstance(data, list):
            return []

        results = []
        for coin in data:
            quote = coin.get("quote", {}).get(convert, {})
            results.append({
                "id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "slug": coin.get("slug"),
                "rank": coin.get("cmc_rank"),
                "price": quote.get("price"),
                "volume_24h": quote.get("volume_24h"),
                "market_cap": quote.get("market_cap"),
                "percent_change_1h": quote.get("percent_change_1h"),
                "percent_change_24h": quote.get("percent_change_24h"),
                "percent_change_7d": quote.get("percent_change_7d"),
                "percent_change_30d": quote.get("percent_change_30d"),
                "circulating_supply": coin.get("circulating_supply"),
                "total_supply": coin.get("total_supply"),
                "max_supply": coin.get("max_supply"),
                "last_updated": quote.get("last_updated"),
            })

        logger.info(f"CMC: fetched {len(results)} listings")
        return results

    # ------------------------------------------------------------------
    # Quotes (specific coins)
    # ------------------------------------------------------------------

    def get_quote(
        self,
        symbol: str,
        convert: str = "USD",
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for a specific cryptocurrency.

        Args:
            symbol: Coin symbol (e.g., "BTC", "ETH", "SOL")
        """
        data = self._get("/v1/cryptocurrency/quotes/latest", {
            "symbol": symbol.upper(),
            "convert": convert,
        })

        if isinstance(data, dict) and "error" in data:
            return None

        coin_data = data.get(symbol.upper())
        if not coin_data:
            return None

        quote = coin_data.get("quote", {}).get(convert, {})
        return {
            "symbol": coin_data.get("symbol"),
            "name": coin_data.get("name"),
            "rank": coin_data.get("cmc_rank"),
            "price": quote.get("price"),
            "volume_24h": quote.get("volume_24h"),
            "market_cap": quote.get("market_cap"),
            "percent_change_1h": quote.get("percent_change_1h"),
            "percent_change_24h": quote.get("percent_change_24h"),
            "percent_change_7d": quote.get("percent_change_7d"),
            "percent_change_30d": quote.get("percent_change_30d"),
            "percent_change_60d": quote.get("percent_change_60d"),
            "percent_change_90d": quote.get("percent_change_90d"),
            "fully_diluted_market_cap": quote.get("fully_diluted_market_cap"),
            "market_cap_dominance": quote.get("market_cap_dominance"),
            "last_updated": quote.get("last_updated"),
        }

    def get_quotes_multi(
        self,
        symbols: List[str],
        convert: str = "USD",
    ) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple coins at once."""
        symbol_str = ",".join(s.upper() for s in symbols)
        data = self._get("/v1/cryptocurrency/quotes/latest", {
            "symbol": symbol_str,
            "convert": convert,
        })

        if isinstance(data, dict) and "error" in data:
            return {}

        results = {}
        for sym in symbols:
            coin_data = data.get(sym.upper())
            if coin_data:
                quote = coin_data.get("quote", {}).get(convert, {})
                results[sym.upper()] = {
                    "price": quote.get("price"),
                    "volume_24h": quote.get("volume_24h"),
                    "market_cap": quote.get("market_cap"),
                    "percent_change_24h": quote.get("percent_change_24h"),
                }

        return results

    # ------------------------------------------------------------------
    # Global Metrics
    # ------------------------------------------------------------------

    def get_global_metrics(self, convert: str = "USD") -> Dict[str, Any]:
        """
        Get global cryptocurrency market metrics.

        Returns: total market cap, BTC dominance, ETH dominance,
                 active cryptocurrencies, total exchanges, etc.
        """
        data = self._get("/v1/global-metrics/quotes/latest", {
            "convert": convert,
        })

        if isinstance(data, dict) and "error" in data:
            return {}

        quote = data.get("quote", {}).get(convert, {})
        return {
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
            "total_cryptocurrencies": data.get("total_cryptocurrencies"),
            "active_exchanges": data.get("active_exchanges"),
            "total_exchanges": data.get("total_exchanges"),
            "btc_dominance": data.get("btc_dominance"),
            "eth_dominance": data.get("eth_dominance"),
            "defi_volume_24h": data.get("defi_volume_24h"),
            "defi_market_cap": data.get("defi_market_cap"),
            "stablecoin_volume_24h": data.get("stablecoin_volume_24h"),
            "stablecoin_market_cap": data.get("stablecoin_market_cap"),
            "total_market_cap": quote.get("total_market_cap"),
            "total_volume_24h": quote.get("total_volume_24h"),
            "last_updated": data.get("last_updated"),
        }

    # ------------------------------------------------------------------
    # Trending & Categories
    # ------------------------------------------------------------------

    def get_trending(self, limit: int = 20, convert: str = "USD") -> List[Dict[str, Any]]:
        """Get trending cryptocurrencies."""
        data = self._get("/v1/cryptocurrency/trending/latest", {
            "limit": limit,
            "convert": convert,
        })

        if isinstance(data, dict) and "error" in data:
            return []

        if not isinstance(data, list):
            return []

        return [
            {
                "symbol": coin.get("symbol"),
                "name": coin.get("name"),
                "rank": coin.get("cmc_rank"),
                "price": coin.get("quote", {}).get(convert, {}).get("price"),
                "percent_change_24h": coin.get("quote", {}).get(convert, {}).get("percent_change_24h"),
            }
            for coin in data
        ]

    def get_categories(self) -> List[Dict[str, Any]]:
        """Get cryptocurrency categories (DeFi, Layer 1, Meme, etc.)."""
        data = self._get("/v1/cryptocurrency/categories")

        if isinstance(data, dict) and "error" in data:
            return []

        if not isinstance(data, list):
            return []

        return [
            {
                "id": cat.get("id"),
                "name": cat.get("name"),
                "title": cat.get("title"),
                "num_tokens": cat.get("num_tokens"),
                "market_cap": cat.get("market_cap"),
                "volume_24h": cat.get("volume"),
                "avg_price_change": cat.get("avg_price_change"),
            }
            for cat in data
        ]

    # ------------------------------------------------------------------
    # Symbol Mapping
    # ------------------------------------------------------------------

    def get_id_map(self, limit: int = 5000) -> Dict[str, int]:
        """
        Get symbol → CMC ID mapping.
        Useful for converting symbols to CMC IDs for other endpoints.
        """
        data = self._get("/v1/cryptocurrency/map", {"limit": limit})

        if isinstance(data, dict) and "error" in data:
            return {}

        if not isinstance(data, list):
            return {}

        return {coin["symbol"]: coin["id"] for coin in data}

    # ------------------------------------------------------------------
    # Fear & Greed (via global metrics proxy)
    # ------------------------------------------------------------------

    def get_market_sentiment_proxy(self) -> Dict[str, Any]:
        """
        Estimate market sentiment from global metrics.
        Uses BTC dominance, volume changes, and market cap changes
        as a proxy for fear/greed.
        """
        metrics = self.get_global_metrics()
        if not metrics:
            return {"sentiment": "neutral", "score": 50}

        btc_dom = metrics.get("btc_dominance", 50)
        total_mc = metrics.get("total_market_cap", 0)
        total_vol = metrics.get("total_volume_24h", 0)

        # Simple heuristic:
        # High BTC dominance (>60%) = fear (flight to safety)
        # Low BTC dominance (<40%) = greed (alt season)
        # High volume relative to market cap = activity/greed
        score = 50  # neutral

        if btc_dom and btc_dom > 60:
            score -= 15  # fear
        elif btc_dom and btc_dom < 40:
            score += 15  # greed

        if total_mc and total_vol:
            vol_ratio = total_vol / total_mc if total_mc > 0 else 0
            if vol_ratio > 0.1:
                score += 10  # high activity
            elif vol_ratio < 0.03:
                score -= 10  # low activity

        score = max(0, min(100, score))

        if score >= 75:
            sentiment = "extreme_greed"
        elif score >= 55:
            sentiment = "greed"
        elif score >= 45:
            sentiment = "neutral"
        elif score >= 25:
            sentiment = "fear"
        else:
            sentiment = "extreme_fear"

        return {
            "sentiment": sentiment,
            "score": score,
            "btc_dominance": btc_dom,
            "total_market_cap": total_mc,
            "total_volume_24h": total_vol,
        }

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test if the API key is valid."""
        data = self._get("/v1/cryptocurrency/map", {"limit": 1})
        if isinstance(data, dict) and "error" in data:
            return False
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_coinmarketcap_client(
    api_key: Optional[str] = None,
) -> CoinMarketCapClient:
    """Create a CoinMarketCap client from env or explicit key."""
    return CoinMarketCapClient(api_key=api_key)
