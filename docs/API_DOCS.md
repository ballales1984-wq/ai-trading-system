# AI Trading System — API Reference

> Base URL: `https://<host>/api/v1`  
> Auth: `Bearer <JWT>` — ottenuto via `/auth/login`  
> Tutti gli endpoint JSON restituiscono `Content-Type: application/json`

---

## 🔐 Auth

### POST /auth/login

Login utente e ottenimento del JWT.

```bash
curl -X POST https://HOST/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

**Response 200**
```json
{ "access_token": "eyJ...", "token_type": "bearer" }
```

---

## 📊 Portfolio

### GET /portfolio/summary/dual

Riassunto portafoglio (real + simulato).

```bash
curl -H "Authorization: Bearer $TOKEN" \
     https://HOST/api/v1/portfolio/summary/dual
```

**Response 200**
```json
{
  "real": { "total_value": 100000.0, "daily_pnl": -120.5, "unrealized_pnl": 0, "cash_balance": 100000.0 },
  "simulated": { "total_value": 98432.5, "daily_pnl": 432.1, "unrealized_pnl": 812.3, "num_positions": 3 }
}
```

### GET /portfolio/positions

Lista posizioni aperte.

| Param | Tipo | Default | Note |
|-------|------|---------|------|
| `symbol` | string | — | filtra per simbolo |

```bash
curl -H "Authorization: Bearer $TOKEN" \
     "https://HOST/api/v1/portfolio/positions?symbol=BTCUSDT"
```

**Response 200** — Array di posizioni
```json
[
  {
    "position_id": "pos_001",
    "symbol": "BTCUSDT",
    "side": "LONG",
    "quantity": 0.05,
    "entry_price": 65000.0,
    "current_price": 67200.0,
    "market_value": 3360.0,
    "unrealized_pnl": 110.0
  }
]
```

### GET /portfolio/performance

Metriche di performance del portafoglio.

```bash
curl -H "Authorization: Bearer $TOKEN" \
     https://HOST/api/v1/portfolio/performance
```

**Response 200**
```json
{
  "total_return_pct": 5.2,
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.1,
  "max_drawdown_pct": -3.4,
  "win_rate": 0.66,
  "profit_factor": 2.1,
  "num_winning_trades": 42,
  "num_losing_trades": 22
}
```

### GET /portfolio/history

Curva equity storica.

| Param | Tipo | Default |
|-------|------|---------|
| `days` | int | 30 |

```bash
curl -H "Authorization: Bearer $TOKEN" \
     "https://HOST/api/v1/portfolio/history?days=30"
```

### GET /portfolio/allocation

Allocazione del capitale per simbolo.

---

## 📈 Market

### GET /market/prices

Prezzi di tutti gli asset supportati.

```bash
curl https://HOST/api/v1/market/prices
```

**Response 200**
```json
{
  "markets": [
    { "symbol": "BTCUSDT", "price": 67200.0, "change_pct_24h": 2.15, "volume_24h": 1234567890 }
  ]
}
```

### GET /market/price/{symbol}

Prezzo singolo asset.

```bash
curl https://HOST/api/v1/market/price/BTCUSDT
```

### GET /market/candles/{symbol}

Dati OHLCV.

| Param | Tipo | Default |
|-------|------|---------|
| `interval` | string | `1h` (`1m`, `5m`, `1h`, `4h`, `1d`) |
| `limit` | int | 100 |

```bash
curl "https://HOST/api/v1/market/candles/BTCUSDT?interval=1h&limit=50"
```

### GET /market/sentiment

Sentiment aggregato da news e social.

### GET /market/orderbook/{symbol}

Order book (bid/ask).

---

## 📋 Orders

### GET /orders

Lista ordini.

| Param | Tipo | Note |
|-------|------|------|
| `symbol` | string | filtra per simbolo |
| `status` | string | `PENDING`, `FILLED`, `CANCELLED` |

### POST /orders

Crea nuovo ordine.

```bash
curl -X POST https://HOST/api/v1/orders \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "order_type": "MARKET",
    "quantity": 0.01,
    "broker": "paper"
  }'
```

**Response 201**
```json
{
  "order_id": "ord_xyz123",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "status": "PENDING",
  "quantity": 0.01,
  "created_at": "2026-03-14T00:30:00Z"
}
```

### GET /orders/{order_id}

Dettaglio singolo ordine.

### DELETE /orders/{order_id}

Cancella ordine in pending.

### POST /orders/{order_id}/execute

Forza esecuzione ordine (demo mode).

### GET /orders/emergency/status

Stato del kill switch di emergenza.

### POST /orders/emergency/activate

Attiva emergency stop (richiede header `X-Admin-Key`).

```bash
curl -X POST https://HOST/api/v1/orders/emergency/activate \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Admin-Key: YOUR_ADMIN_KEY" \
  -H "X-Admin-User: admin" \
  -d '{"confirm": true, "reason": "Market crash detected"}'
```

---

## 🛡️ Risk

### GET /risk/metrics

Metriche di rischio correnti.

```bash
curl -H "Authorization: Bearer $TOKEN" https://HOST/api/v1/risk/metrics
```

**Response 200**
```json
{
  "var_1d": 0.012,
  "cvar_1d": 0.018,
  "volatility": 0.025,
  "beta": 1.1,
  "leverage": 1.2,
  "sharpe_ratio": 1.85,
  "margin_utilization": 0.08
}
```

### GET /risk/correlation

Matrice di correlazione tra asset.

---

## 📰 News

### GET /news

Lista ultime notizie.

| Param | Tipo | Default |
|-------|------|---------|
| `limit` | int | 20 |
| `refresh` | bool | false |

### GET /news/{symbol}

Notizie filtrate per simbolo.

---

## 🔧 System

### GET /health

Health check con stato di DB, Redis e API esterne.

**Response 200**
```json
{
  "status": "healthy",
  "database": "ok",
  "redis": "ok",
  "version": "2.0.0"
}
```

### GET /metrics

Endpoint Prometheus (metriche esposte per scraping).

Metriche principali:

| Metrica | Tipo | Descrizione |
|---------|------|-------------|
| `trading_orders_total` | Counter | Ordini totali creati |
| `trading_pnl_current` | Gauge | PnL corrente |
| `trading_positions_open` | Gauge | Posizioni aperte |
| `risk_var_current` | Gauge | VaR a 1 giorno |
| `risk_drawdown_current` | Gauge | Drawdown corrente |
| `risk_circuit_breaker_state` | Gauge | 0=closed, 1=open |
| `http_requests_total` | Counter | Request totali per endpoint |
| `http_request_duration_seconds` | Histogram | Latenza API |
| `cache_hit_ratio` | Gauge | Hit rate cache |

### GET /api/v1/rate-limit/stats

Statistiche rate limiting.

---

## 🌐 WebSocket

### WS /ws/prices

Stream real-time prezzi e portfolio updates.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/prices');
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  // msg.type: 'price_update' | 'portfolio_update' | 'ping'
  // msg.data: { symbol, price, change_pct_24h } | { total_value, daily_pnl, ... }
};
```

---

## ⚡ Quick Start

```bash
# 1. Avvia il backend
uvicorn app.main:app --reload

# 2. Ottieni token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r .access_token)

# 3. Chiamata autenticata
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/portfolio/summary/dual | jq
```

> 📖 **Swagger UI**: http://localhost:8000/docs  
> 📖 **ReDoc**: http://localhost:8000/redoc
