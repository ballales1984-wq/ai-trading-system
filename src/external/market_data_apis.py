"""
Market Data API Clients
========================
Clients for: Binance, CoinGecko, Alpha Vantage, Quandl/Nasdaq Data Link.

These provide OHLCV price data, historical series, and real-time quotes
that feed into the Technical Indicators engine and Monte Carlo Level 1.
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
# Binance API Client
# ---------------------------------------------------------------------------

class BinanceMarketClient(BaseAPIClient):
    """
    Binance REST API for crypto OHLCV data.
    Docs: https://binance-docs.github.io/apidocs/spot/en/
    """

    def __init__(self, api_key: str = "", testnet: bool = False):
        base = (
            "https://testnet.binance.vision/api/v3"
            if testnet
            else "https://api.binance.com/api/v3"
        )
        super().__init__(
            name="binance",
            category=APICategory.MARKET_DATA,
            api_key=api_key,
            base_url=base,
            rate_limit=RateLimitConfig(
                max_requests_per_minute=1200,
                max_requests_per_second=20,
            ),
        )
        self.testnet = testnet

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        symbol = kwargs.get("symbol", "BTCUSDT")
        interval = kwargs.get("interval", "1h")
        limit = kwargs.get("limit", 100)

        url = f"{self.base_url}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        records: List[NormalizedRecord] = []
        if aiohttp is None:
            logger.warning("aiohttp not installed — skipping Binance fetch")
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Binance HTTP {resp.status}")
                data = await resp.json()

        for candle in data:
            ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.MARKET_DATA,
                    source_api="binance",
                    data_type="ohlcv",
                    payload={
                        "asset": symbol,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    },
                    quality=DataQuality.HIGH,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/ping") as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CoinGecko API Client
# ---------------------------------------------------------------------------

class CoinGeckoClient(BaseAPIClient):
    """
    CoinGecko API for crypto market data (free tier).
    Docs: https://www.coingecko.com/en/api/documentation
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="coingecko",
            category=APICategory.MARKET_DATA,
            api_key=api_key,
            base_url="https://api.coingecko.com/api/v3",
            rate_limit=RateLimitConfig(
                max_requests_per_minute=30,
                max_requests_per_second=1,
            ),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        symbol = kwargs.get("symbol", "bitcoin")
        # CoinGecko uses coin IDs, not ticker symbols
        coin_id = self._symbol_to_id(symbol)
        days = kwargs.get("limit", 100)

        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}

        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        records: List[NormalizedRecord] = []
        if aiohttp is None:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"CoinGecko HTTP {resp.status}")
                data = await resp.json()

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        for i, (ts_ms, price) in enumerate(prices):
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            vol = volumes[i][1] if i < len(volumes) else 0.0
            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.MARKET_DATA,
                    source_api="coingecko",
                    data_type="price",
                    payload={
                        "asset": coin_id,
                        "close": price,
                        "volume": vol,
                    },
                    quality=DataQuality.MEDIUM,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/ping") as resp:
                    return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _symbol_to_id(symbol: str) -> str:
        mapping = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "SOLUSDT": "solana",
        }
        return mapping.get(symbol.upper(), symbol.lower())


# ---------------------------------------------------------------------------
# Alpha Vantage API Client
# ---------------------------------------------------------------------------

class AlphaVantageClient(BaseAPIClient):
    """
    Alpha Vantage for stocks, forex, and commodities.
    Docs: https://www.alphavantage.co/documentation/
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="alpha_vantage",
            category=APICategory.MARKET_DATA,
            api_key=api_key,
            base_url="https://www.alphavantage.co/query",
            rate_limit=RateLimitConfig(
                max_requests_per_minute=5,
                max_requests_per_second=1,
            ),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        symbol = kwargs.get("symbol", "AAPL")
        interval = kwargs.get("interval", "daily")

        function_map = {
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",
            "1min": "TIME_SERIES_INTRADAY",
            "5min": "TIME_SERIES_INTRADAY",
            "15min": "TIME_SERIES_INTRADAY",
            "60min": "TIME_SERIES_INTRADAY",
        }
        func = function_map.get(interval, "TIME_SERIES_DAILY")
        params: Dict[str, Any] = {
            "function": func,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "compact",
        }
        if "INTRADAY" in func:
            params["interval"] = interval

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"AlphaVantage HTTP {resp.status}")
                data = await resp.json()

        # find the time-series key
        ts_key = None
        for k in data:
            if "Time Series" in k:
                ts_key = k
                break
        if not ts_key:
            logger.warning(f"AlphaVantage: no time series in response for {symbol}")
            return records

        for date_str, values in data[ts_key].items():
            try:
                ts = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                ts = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.MARKET_DATA,
                    source_api="alpha_vantage",
                    data_type="ohlcv",
                    payload={
                        "asset": symbol,
                        "open": float(values.get("1. open", 0)),
                        "high": float(values.get("2. high", 0)),
                        "low": float(values.get("3. low", 0)),
                        "close": float(values.get("4. close", 0)),
                        "volume": float(values.get("5. volume", 0)),
                    },
                    quality=DataQuality.HIGH,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "IBM",
                "interval": "5min",
                "apikey": self.api_key,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Quandl / Nasdaq Data Link Client
# ---------------------------------------------------------------------------

class QuandlClient(BaseAPIClient):
    """
    Quandl (Nasdaq Data Link) for quantitative historical series.
    Docs: https://docs.data.nasdaq.com/
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="quandl",
            category=APICategory.MARKET_DATA,
            api_key=api_key,
            base_url="https://data.nasdaq.com/api/v3",
            rate_limit=RateLimitConfig(
                max_requests_per_minute=300,
                max_requests_per_second=5,
            ),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        dataset = kwargs.get("symbol", "WIKI/AAPL")
        limit = kwargs.get("limit", 100)

        url = f"{self.base_url}/datasets/{dataset}/data.json"
        params: Dict[str, Any] = {"api_key": self.api_key, "limit": limit}

        records: List[NormalizedRecord] = []
        if aiohttp is None or not self.api_key:
            return records

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Quandl HTTP {resp.status}")
                data = await resp.json()

        dataset_data = data.get("dataset_data", {})
        columns = dataset_data.get("column_names", [])
        rows = dataset_data.get("data", [])

        for row in rows:
            row_dict = dict(zip(columns, row))
            date_str = row_dict.get("Date", "")
            try:
                ts = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            records.append(
                NormalizedRecord(
                    timestamp=ts,
                    category=APICategory.MARKET_DATA,
                    source_api="quandl",
                    data_type="ohlcv",
                    payload={
                        "asset": dataset,
                        "open": float(row_dict.get("Open", 0)),
                        "high": float(row_dict.get("High", 0)),
                        "low": float(row_dict.get("Low", 0)),
                        "close": float(row_dict.get("Close", 0)),
                        "volume": float(row_dict.get("Volume", 0)),
                    },
                    quality=DataQuality.HIGH,
                )
            )
        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            url = f"{self.base_url}/datasets/WIKI/AAPL/metadata.json"
            params = {"api_key": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CoinMarketCap API Client
# ---------------------------------------------------------------------------

class CoinMarketCapClient(BaseAPIClient):
    """
    CoinMarketCap Pro API for crypto market data, rankings, and global metrics.
    Docs: https://coinmarketcap.com/api/documentation/v1/
    """

    def __init__(self, api_key: str = ""):
        super().__init__(
            name="coinmarketcap",
            category=APICategory.MARKET_DATA,
            api_key=api_key,
            base_url="https://pro-api.coinmarketcap.com",
            rate_limit=RateLimitConfig(
                max_requests_per_minute=30,
                max_requests_per_second=1,
            ),
        )

    async def fetch(self, **kwargs) -> List[NormalizedRecord]:
        symbol = kwargs.get("symbol", "BTC")
        limit = kwargs.get("limit", 100)

        records: List[NormalizedRecord] = []
        if not self.api_key:
            logger.warning("CoinMarketCap API key not set")
            return records

        # Get listings
        listings_url = f"{self.base_url}/v1/cryptocurrency/listings/latest"
        params = {"limit": limit, "convert": "USD"}

        if aiohttp is None:
            logger.warning("aiohttp not installed — skipping CoinMarketCap fetch")
            return records

        headers = {"X-CMC_PRO_API_KEY": self.api_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(listings_url, params=params, headers=headers) as resp:
                    if resp.status != 200:
                        logger.error(f"CoinMarketCap HTTP {resp.status}")
                        return records
                    data = await resp.json()

            if "data" not in data:
                return records

            for coin in data.get("data", []):
                quote = coin.get("quote", {}).get("USD", {})
                ts = datetime.now(timezone.utc)

                records.append(
                    NormalizedRecord(
                        timestamp=ts,
                        category=APICategory.MARKET_DATA,
                        source_api="coinmarketcap",
                        data_type="market_data",
                        payload={
                            "asset": coin.get("symbol"),
                            "name": coin.get("name"),
                            "rank": coin.get("cmc_rank"),
                            "price": quote.get("price"),
                            "volume_24h": quote.get("volume_24h"),
                            "market_cap": quote.get("market_cap"),
                            "percent_change_24h": quote.get("percent_change_24h"),
                            "percent_change_7d": quote.get("percent_change_7d"),
                        },
                        quality=DataQuality.HIGH,
                    )
                )
        except Exception as e:
            logger.error(f"CoinMarketCap fetch error: {e}")

        return records

    async def health_check(self) -> bool:
        if aiohttp is None or not self.api_key:
            return False
        try:
            headers = {"X-CMC_PRO_API_KEY": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/v1/cryptocurrency/map",
                    params={"limit": 1},
                    headers=headers
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_market_data_clients(
    binance_key: str = "",
    coingecko_key: str = "",
    coinmarketcap_key: str = "",
    alpha_vantage_key: str = "",
    quandl_key: str = "",
    binance_testnet: bool = False,
) -> List[BaseAPIClient]:
    """Create all market-data clients from provided keys."""
    clients: List[BaseAPIClient] = []
    clients.append(BinanceMarketClient(api_key=binance_key, testnet=binance_testnet))
    clients.append(CoinGeckoClient(api_key=coingecko_key))
    if coinmarketcap_key:
        clients.append(CoinMarketCapClient(api_key=coinmarketcap_key))
    if alpha_vantage_key:
        clients.append(AlphaVantageClient(api_key=alpha_vantage_key))
    if quandl_key:
        clients.append(QuandlClient(api_key=quandl_key))
    return clients


def create_market_data_clients_from_env() -> List[BaseAPIClient]:
    """Create market-data clients using environment variables."""
    return create_market_data_clients(
        binance_key=os.getenv("BINANCE_API_KEY", ""),
        coingecko_key=os.getenv("COINGECKO_API_KEY", ""),
        coinmarketcap_key=os.getenv("COINMARKETCAP_API_KEY", ""),
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
        quandl_key=os.getenv("QUANDL_API_KEY", ""),
        binance_testnet=os.getenv("BINANCE_TESTNET", "false").lower() == "true",
    )
