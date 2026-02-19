# AI Trading System - API Reference Guide

## Panoramica delle API Integrate

Questo documento descrive tutte le API esterne integrate nel sistema di trading.

---

## 📊 Market Data APIs

### 1. Binance Market Client
**File**: [`src/external/market_data_apis.py`](src/external/market_data_apis.py)
**Classe**: `BinanceMarketClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Prezzi real-time | Ottiene prezzi spot e futures |
| Order book | Depth of market per liquidità |
| Klines/Candlestick | Dati OHLCV per analisi tecnica |
| Recent trades | Storico trade recenti |

**Environment Variables**:
- `BINANCE_API_KEY` - API key
- `BINANCE_SECRET_KEY` - Secret key
- `BINANCE_TESTNET` - Usa testnet (true/false)

**Rate Limits**: 1200 richieste/minuto

---

### 2. CoinGecko Client
**File**: [`src/external/market_data_apis.py`](src/external/market_data_apis.py)
**Classe**: `CoinGeckoClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Prezzi crypto | Prezzi in tempo reale |
| Market data | Cap, volume, supply |
| Historical data | Dati storici |
| Trending | Crypto più popolari |

**Environment Variables**:
- `COINGECKO_API_KEY` - API key (opzionale per plan free)

**Rate Limits**: 10-50 richieste/minuto (free), 500+ (paid)

---

### 3. CoinMarketCap Client
**File**: [`src/external/market_data_apis.py`](src/external/market_data_apis.py), [`src/external/coinmarketcap_client.py`](src/external/coinmarketcap_client.py)
**Classe**: `CoinMarketCapClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Listings | Top crypto per cap |
| Quotes | Prezzi multi-valuta |
| Metadata | Info su progetti |
| Global metrics | Statistiche globali mercato |

**Environment Variables**:
- `COINMARKETCAP_API_KEY` - API key richiesta

**Rate Limits**: 333 richieste/giorno (basic), 10000 (professional)

---

### 4. Alpha Vantage Client
**File**: [`src/external/market_data_apis.py`](src/external/market_data_apis.py)
**Classe**: `AlphaVantageClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Stock prices | Azioni, ETF, indici |
| Forex | Tassi di cambio |
| Technical indicators | SMA, EMA, RSI, etc. |
| Crypto | Dati cryptocurrency |

**Environment Variables**:
- `ALPHA_VANTAGE_API_KEY` - API key richiesta

**Rate Limits**: 5 richieste/minuto (free), 600/minuto (premium)

---

### 5. Quandl Client
**File**: [`src/external/market_data_apis.py`](src/external/market_data_apis.py)
**Classe**: `QuandlClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Economic data | Dati macroeconomici |
| Financial data | Bilanci aziendali |
| Futures | Dati futures |
| Alternative data | Dati alternativi |

**Environment Variables**:
- `QUANDL_API_KEY` - API key

**Rate Limits**: Varia per dataset

---

## 📰 Sentiment & News APIs

### 6. NewsAPI Client
**File**: [`src/external/sentiment_apis.py`](src/external/sentiment_apis.py)
**Classe**: `NewsAPIClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Top headlines | Notizie principali |
| Everything | Ricerca globale |
| Sources | Fonti di notizie |
| Sentiment analysis | Analisi sentiment |

**Environment Variables**:
- `NEWSAPI_KEY` - API key richiesta

**Rate Limits**: 100 richieste/giorno (free), 10000 (developer)

---

### 7. Benzinga Client
**File**: [`src/external/sentiment_apis.py`](src/external/sentiment_apis.py)
**Classe**: `BenzingaClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| News feed | Notizie finanziarie |
| Earnings | Calcolo earnings |
| Ratings | Rating analisti |
| IPOs | Calendario IPO |

**Environment Variables**:
- `BENZINGA_API_KEY` - API key

**Rate Limits**: Varia per piano

---

### 8. Twitter/X Sentiment Client
**File**: [`src/external/sentiment_apis.py`](src/external/sentiment_apis.py)
**Classe**: `TwitterSentimentClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Tweet search | Ricerca tweet per keyword |
| User timeline | Tweet di utenti specifici |
| Sentiment scoring | Score sentiment tweet |
| Trending topics | Argomenti trending |

**Environment Variables**:
- `TWITTER_BEARER_TOKEN` - Bearer token OAuth 2.0

**Rate Limits**: 15 richieste/15min (search), 900/15min (timeline)

---

### 9. GDELT Client
**File**: [`src/external/sentiment_apis.py`](src/external/sentiment_apis.py)
**Classe**: `GDELTClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Global news | Notizie globali |
| Event database | Eventi geopolitici |
| Tone analysis | Tono articoli |
| GKG (Global Knowledge Graph) | Knowledge graph globale |

**Environment Variables**: Nessuna richiesta (API gratuita)

**Rate Limits**: Rate limiting dinamico

---

## 📈 Macro Event APIs

### 10. Trading Economics Client
**File**: [`src/external/macro_event_apis.py`](src/external/macro_event_apis.py)
**Classe**: `TradingEconomicsClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Economic calendar | Calendario eventi |
| Indicators | Indicatori macro |
| Forecasts | Previsioni |
| Historical data | Dati storici |

**Environment Variables**:
- `TRADING_ECONOMICS_API_KEY` - API key

**Rate Limits**: Varia per piano

---

### 11. EconPulse Client
**File**: [`src/external/macro_event_apis.py`](src/external/macro_event_apis.py)
**Classe**: `EconPulseClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Economic indicators | Indicatori economici |
| Market pulse | Battito mercato |
| Country data | Dati per paese |

**Environment Variables**:
- `ECONPULSE_API_KEY` - API key

---

### 12. Investing.com Client
**File**: [`src/external/macro_event_apis.py`](src/external/macro_event_apis.py)
**Classe**: `InvestingComClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Economic calendar | Calendario economico |
| Market quotes | Quotazioni |
| Analysis | Analisi di mercato |

**Environment Variables**:
- `INVESTING_COM_API_KEY` - API key

---

## 🔬 Innovation & Alternative Data APIs

### 13. EIA (Energy Information Administration) Client
**File**: [`src/external/innovation_apis.py`](src/external/innovation_apis.py)
**Classe**: `EIAClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Energy data | Dati energetici |
| Petroleum | Dati petrolio |
| Natural gas | Gas naturale |
| Electricity | Elettricità |

**Environment Variables**:
- `EIA_API_KEY` - API key (gratuita)

**Rate Limits**: Varia

---

### 14. Google Patents Client
**File**: [`src/external/innovation_apis.py`](src/external/innovation_apis.py)
**Classe**: `GooglePatentsClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Patent search | Ricerca brevetti |
| Patent metadata | Info brevetti |
| Innovation trends | Trend innovazione |

**Environment Variables**:
- `SERPAPI_KEY` - SerpAPI key

---

### 15. Lens.org Client
**File**: [`src/external/innovation_apis.py`](src/external/innovation_apis.py)
**Classe**: `LensOrgClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Patent data | Dati brevettuali |
| Scholar data | Dati accademici |
| Citation analysis | Analisi citazioni |

**Environment Variables**:
- `LENS_ORG_API_KEY` - API key

---

## 🌍 Natural Event APIs

### 16. Open-Meteo Client
**File**: [`src/external/natural_event_apis.py`](src/external/natural_event_apis.py)
**Classe**: `OpenMeteoClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Weather forecast | Previsioni meteo |
| Historical weather | Meteo storico |
| Air quality | Qualità aria |

**Environment Variables**: Nessuna richiesta (API gratuita)

**Rate Limits**: Nessun rate limit severo

---

### 17. Climate TRACE Client
**File**: [`src/external/natural_event_apis.py`](src/external/natural_event_apis.py)
**Classe**: `ClimateTRACEClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Emissions data | Dati emissioni |
| Facility data | Dati impianti |
| Climate tracking | Monitoraggio clima |

**Environment Variables**: Nessuna richiesta

---

### 18. USGS Water Client
**File**: [`src/external/natural_event_apis.py`](src/external/natural_event_apis.py)
**Classe**: `USGSWaterClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Water levels | Livelli acqua |
| Stream flow | Portata fiumi |
| Groundwater | Acque sotterranee |

**Environment Variables**: Nessuna richiesta (API governativa gratuita)

---

## 💱 Exchange APIs

### 19. Bybit Client
**File**: [`src/external/bybit_client.py`](src/external/bybit_client.py)
**Classe**: `BybitClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Spot trading | Trading spot |
| Futures trading | Trading futures |
| Account info | Info account |
| Order management | Gestione ordini |

**Environment Variables**:
- `BYBIT_API_KEY` - API key
- `BYBIT_SECRET_KEY` - Secret key

**Rate Limits**: Varia per endpoint

---

### 20. OKX Client
**File**: [`src/external/okx_client.py`](src/external/okx_client.py)
**Classe**: `OKXClient`

| Funzionalità | Descrizione |
|-------------|-------------|
| Spot trading | Trading spot |
| Derivatives | Derivati |
| Account | Gestione account |
| Market data | Dati mercato |

**Environment Variables**:
- `OKX_API_KEY` - API key
- `OKX_SECRET_KEY` - Secret key
- `OKX_PASSPHRASE` - Passphrase

---

## 🤖 Telegram Bot API

### 21. Telegram Notifier
**File**: [`src/live/telegram_notifier.py`](src/live/telegram_notifier.py)
**Classe**: `TelegramNotifier`

| Funzionalità | Descrizione |
|-------------|-------------|
| Send messages | Invio messaggi |
| Trade alerts | Alert trading |
| Portfolio updates | Aggiornamenti portfolio |
| Inline keyboards | Pulsanti interattivi |

**Environment Variables**:
- `TELEGRAM_BOT_TOKEN` - Bot token
- `TELEGRAM_CHAT_ID` - Chat ID

**Rate Limits**: 30 messaggi/secondo

---

## 🔐 Security APIs

### 22. JWT Authentication
**File**: [`app/core/security.py`](app/core/security.py)
**Classe**: `User`, `TokenData`

| Funzionalità | Descrizione |
|-------------|-------------|
| Token creation | Creazione JWT |
| Token validation | Validazione token |
| Password hashing | Hash bcrypt |
| User authentication | Autenticazione utenti |

**Environment Variables**:
- `SECRET_KEY` - Chiave segreta JWT

---

## 📋 Configurazione Environment

Crea un file `.env` con le tue API key:

```bash
# Market Data
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
COINGECKO_API_KEY=your_key
COINMARKETCAP_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
QUANDL_API_KEY=your_key

# Sentiment & News
NEWSAPI_KEY=your_key
BENZINGA_API_KEY=your_key
TWITTER_BEARER_TOKEN=your_token

# Macro Events
TRADING_ECONOMICS_API_KEY=your_key
ECONPULSE_API_KEY=your_key
INVESTING_COM_API_KEY=your_key

# Innovation
EIA_API_KEY=your_key
SERPAPI_KEY=your_key
LENS_ORG_API_KEY=your_key

# Exchanges
BYBIT_API_KEY=your_key
BYBIT_SECRET_KEY=your_secret
OKX_API_KEY=your_key
OKX_SECRET_KEY=your_secret
OKX_PASSPHRASE=your_passphrase

# Notifications
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Security
SECRET_KEY=your_secret_key
```

---

## 📁 Struttura File

```
src/external/
├── __init__.py
├── api_registry.py          # Base class per tutti i client
├── market_data_apis.py      # Binance, CoinGecko, AlphaVantage, etc.
├── sentiment_apis.py        # NewsAPI, Twitter, Benzinga, GDELT
├── macro_event_apis.py      # TradingEconomics, EconPulse
├── innovation_apis.py       # EIA, Patents, Lens.org
├── natural_event_apis.py    # OpenMeteo, ClimateTRACE, USGS
├── bybit_client.py          # Bybit exchange
├── okx_client.py            # OKX exchange
└── coinmarketcap_client.py  # CoinMarketCap dedicato
```

---

*Ultimo aggiornamento: 2026-02-19*
