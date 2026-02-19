"""
External API Clients
=====================
Unified API clients for all external data sources feeding the trading engine.

Architecture:
    APIRegistry (central dispatcher)
    ├── Market Data APIs: Binance, CoinGecko, Alpha Vantage, Quandl
    ├── Sentiment APIs: NewsAPI, Benzinga, Twitter/X, GDELT
    ├── Macro Event APIs: Trading Economics, EconPulse, Investing.com
    ├── Natural Event APIs: Open-Meteo, Climate TRACE, USGS Water
    ├── Innovation APIs: EIA, Google Patents, Lens.org
    └── Exchange Clients: Bybit, OKX (existing)

Usage:
    from src.external import create_full_registry

    registry = create_full_registry()
    prices = await registry.fetch_market_data("BTCUSDT", "1h")
    sentiment = await registry.fetch_sentiment("bitcoin")
"""

# -- Existing exchange clients --
from .bybit_client import BybitClient, create_bybit_client
from .okx_client import OKXClient, create_okx_client

# -- API Registry --
from .api_registry import (
    APICategory,
    APIRegistry,
    BaseAPIClient,
    DataQuality,
    NormalizedRecord,
    RateLimitConfig,
)

# -- Market Data --
from .market_data_apis import (
    AlphaVantageClient,
    BinanceMarketClient,
    CoinGeckoClient,
    QuandlClient,
    create_market_data_clients,
    create_market_data_clients_from_env,
)

# -- Sentiment --
from .sentiment_apis import (
    BenzingaClient,
    GDELTClient,
    NewsAPIClient,
    TwitterSentimentClient,
    create_sentiment_clients,
    create_sentiment_clients_from_env,
)

# -- Macro Events --
from .macro_event_apis import (
    EconPulseClient,
    InvestingComClient,
    TradingEconomicsClient,
    create_macro_event_clients,
    create_macro_event_clients_from_env,
)

# -- Natural Events --
from .natural_event_apis import (
    ClimateTRACEClient,
    OpenMeteoClient,
    USGSWaterClient,
    create_natural_event_clients,
    create_natural_event_clients_from_env,
)

# -- Innovation / Energy --
from .innovation_apis import (
    EIAClient,
    GooglePatentsClient,
    LensOrgClient,
    create_innovation_clients,
    create_innovation_clients_from_env,
)


# ---------------------------------------------------------------------------
# Convenience: create a fully-populated registry
# ---------------------------------------------------------------------------

def create_exchange_client(
    exchange: str,
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
    **kwargs
):
    """Factory function to create exchange client."""
    exchange = exchange.lower()

    if exchange == "bybit":
        return BybitClient(api_key, api_secret, testnet, **kwargs)
    elif exchange == "okx":
        passphrase = kwargs.get("passphrase", "")
        return OKXClient(api_key, api_secret, passphrase, testnet, **kwargs)
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")


def create_full_registry() -> APIRegistry:
    """
    Create an APIRegistry pre-loaded with all API clients,
    configured from environment variables.

    Returns a ready-to-use registry.
    """
    registry = APIRegistry()

    # Market data
    for client in create_market_data_clients_from_env():
        registry.register(client)

    # Sentiment
    for client in create_sentiment_clients_from_env():
        registry.register(client)

    # Macro events
    for client in create_macro_event_clients_from_env():
        registry.register(client)

    # Natural events (all free)
    for client in create_natural_event_clients_from_env():
        registry.register(client)

    # Innovation / energy
    for client in create_innovation_clients_from_env():
        registry.register(client)

    return registry


__all__ = [
    # Exchange clients
    "BybitClient",
    "create_bybit_client",
    "OKXClient",
    "create_okx_client",
    "create_exchange_client",
    # Registry
    "APICategory",
    "APIRegistry",
    "BaseAPIClient",
    "DataQuality",
    "NormalizedRecord",
    "RateLimitConfig",
    # Market data
    "BinanceMarketClient",
    "CoinGeckoClient",
    "AlphaVantageClient",
    "QuandlClient",
    "create_market_data_clients",
    "create_market_data_clients_from_env",
    # Sentiment
    "NewsAPIClient",
    "BenzingaClient",
    "TwitterSentimentClient",
    "GDELTClient",
    "create_sentiment_clients",
    "create_sentiment_clients_from_env",
    # Macro events
    "TradingEconomicsClient",
    "EconPulseClient",
    "InvestingComClient",
    "create_macro_event_clients",
    "create_macro_event_clients_from_env",
    # Natural events
    "OpenMeteoClient",
    "ClimateTRACEClient",
    "USGSWaterClient",
    "create_natural_event_clients",
    "create_natural_event_clients_from_env",
    # Innovation
    "EIAClient",
    "GooglePatentsClient",
    "LensOrgClient",
    "create_innovation_clients",
    "create_innovation_clients_from_env",
    # Full registry
    "create_full_registry",
]
