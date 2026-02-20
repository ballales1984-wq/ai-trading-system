# AI Trading System - API Reference

## REST API Endpoints

Base URL: `http://localhost:8000`

### Health & Status

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-20T00:00:00Z",
  "version": "1.0.0"
}
```

#### GET /ready
Readiness check for Kubernetes.

**Response:**
```json
{
  "ready": true,
  "agents": {
    "market_data": "running",
    "montecarlo": "running",
    "risk": "running"
  }
}
```

#### GET /status
Get full system status.

**Response:**
```json
{
  "mode": "paper_trading",
  "agents_running": 4,
  "portfolio_value": 100000.0,
  "daily_pnl": 1250.50,
  "positions": 3,
  "last_update": "2026-02-20T00:00:00Z"
}
```

---

### Market Data

#### GET /market/price/{symbol}
Get current price for a symbol.

**Parameters:**
- `symbol` (path): Trading pair (e.g., BTCUSDT)

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "price": 42000.50,
  "volume_24h": 1500000000,
  "change_24h": 2.5,
  "timestamp": "2026-02-20T00:00:00Z"
}
```

#### GET /market/prices
Get all tracked prices.

**Response:**
```json
{
  "prices": {
    "BTCUSDT": 42000.50,
    "ETHUSDT": 2800.00,
    "BNBUSDT": 350.00
  },
  "timestamp": "2026-02-20T00:00:00Z"
}
```

#### GET /market/history/{symbol}
Get price history.

**Parameters:**
- `symbol` (path): Trading pair
- `interval` (query): Time interval (1m, 5m, 1h, 1d)
- `limit` (query): Number of candles (default: 100)

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "1h",
  "candles": [
    {
      "timestamp": "2026-02-20T00:00:00Z",
      "open": 41800.00,
      "high": 42100.00,
      "low": 41750.00,
      "close": 42000.50,
      "volume": 15000
    }
  ]
}
```

---

### Orders

#### POST /orders
Place a new order.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "type": "LIMIT",
  "quantity": 0.01,
  "price": 42000.00,
  "stop_loss": 40000.00,
  "take_profit": 45000.00
}
```

**Response:**
```json
{
  "order_id": "12345",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "type": "LIMIT",
  "quantity": 0.01,
  "price": 42000.00,
  "status": "PENDING",
  "timestamp": "2026-02-20T00:00:00Z"
}
```

#### GET /orders
Get all orders.

**Parameters:**
- `symbol` (query): Filter by symbol
- `status` (query): Filter by status (PENDING, FILLED, CANCELLED)
- `limit` (query): Number of orders (default: 50)

**Response:**
```json
{
  "orders": [
    {
      "order_id": "12345",
      "symbol": "BTCUSDT",
      "side": "BUY",
      "status": "FILLED",
      "filled_quantity": 0.01,
      "filled_price": 42001.50,
      "timestamp": "2026-02-20T00:00:00Z"
    }
  ]
}
```

#### DELETE /orders/{order_id}
Cancel an order.

**Response:**
```json
{
  "order_id": "12345",
  "status": "CANCELLED",
  "timestamp": "2026-02-20T00:00:00Z"
}
```

---

### Portfolio

#### GET /portfolio
Get portfolio summary.

**Response:**
```json
{
  "total_value": 105000.00,
  "cash": 50000.00,
  "positions_value": 55000.00,
  "daily_pnl": 1250.50,
  "total_pnl": 5000.00,
  "positions": [
    {
      "symbol": "BTCUSDT",
      "quantity": 0.5,
      "avg_price": 40000.00,
      "current_price": 42000.00,
      "pnl": 1000.00,
      "pnl_percent": 5.0
    }
  ]
}
```

#### GET /portfolio/positions
Get all positions.

**Response:**
```json
{
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "LONG",
      "quantity": 0.5,
      "entry_price": 40000.00,
      "current_price": 42000.00,
      "unrealized_pnl": 1000.00,
      "stop_loss": 38000.00,
      "take_profit": 45000.00
    }
  ]
}
```

#### GET /portfolio/performance
Get performance metrics.

**Response:**
```json
{
  "total_return": 5.0,
  "daily_return": 1.25,
  "sharpe_ratio": 2.5,
  "sortino_ratio": 3.0,
  "max_drawdown": -5.0,
  "win_rate": 65.0,
  "profit_factor": 1.8,
  "total_trades": 100,
  "winning_trades": 65,
  "losing_trades": 35
}
```

---

### Risk

#### GET /risk/metrics
Get risk metrics.

**Response:**
```json
{
  "portfolio_var_95": -0.03,
  "portfolio_cvar_95": -0.045,
  "max_drawdown": -0.05,
  "volatility": 0.35,
  "sharpe_ratio": 2.5,
  "risk_level": "medium"
}
```

#### GET /risk/metrics/{symbol}
Get risk metrics for a specific symbol.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "var_95": -0.05,
  "var_99": -0.08,
  "cvar_95": -0.07,
  "cvar_99": -0.10,
  "max_drawdown": -0.10,
  "volatility": 0.45,
  "sharpe_ratio": 2.0,
  "sortino_ratio": 2.5,
  "beta": 1.2,
  "risk_level": "high"
}
```

#### GET /risk/alerts
Get recent risk alerts.

**Response:**
```json
{
  "alerts": [
    {
      "type": "var_breach",
      "symbol": "BTCUSDT",
      "value": -0.06,
      "threshold": -0.05,
      "severity": "high",
      "timestamp": "2026-02-20T00:00:00Z"
    }
  ]
}
```

#### POST /risk/position-size
Get recommended position size.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "portfolio_value": 100000.00,
  "risk_per_trade": 0.02
}
```

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "recommended_size": 0.05,
  "recommended_value": 2100.00,
  "risk_amount": 2000.00,
  "stop_loss_price": 40000.00
}
```

---

### Strategy

#### GET /strategy/signals
Get recent trading signals.

**Response:**
```json
{
  "signals": [
    {
      "symbol": "BTCUSDT",
      "action": "BUY",
      "confidence": 0.75,
      "strength": "moderate",
      "strategy": "momentum",
      "reason": "Positive momentum with volume confirmation",
      "timestamp": "2026-02-20T00:00:00Z"
    }
  ]
}
```

#### GET /strategy/performance
Get strategy performance.

**Response:**
```json
{
  "strategies": {
    "momentum": {
      "total_signals": 50,
      "winning_signals": 32,
      "losing_signals": 18,
      "win_rate": 0.64,
      "total_pnl": 2500.00,
      "profit_factor": 1.8
    }
  }
}
```

#### POST /strategy/optimize
Run strategy optimization.

**Request Body:**
```json
{
  "strategy": "momentum",
  "generations": 10,
  "population_size": 20
}
```

**Response:**
```json
{
  "optimization_id": "opt_123",
  "status": "running",
  "estimated_time": 300
}
```

---

### Monte Carlo

#### GET /montecarlo/simulation/{symbol}
Get latest simulation results.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "level": "conditional",
  "current_price": 42000.00,
  "mean_price": 43000.00,
  "std_price": 2000.00,
  "percentiles": {
    "p5": 39000.00,
    "p25": 41500.00,
    "p50": 43000.00,
    "p75": 44500.00,
    "p95": 46000.00
  },
  "var_95": -0.07,
  "cvar_95": -0.09,
  "probability_up": 0.65,
  "probability_down": 0.35,
  "timestamp": "2026-02-20T00:00:00Z"
}
```

---

## WebSocket API

### Connection
```
ws://localhost:8000/ws
```

### Subscribe to Market Data
```json
{
  "action": "subscribe",
  "channel": "market_data",
  "symbols": ["BTCUSDT", "ETHUSDT"]
}
```

### Subscribe to Signals
```json
{
  "action": "subscribe",
  "channel": "signals"
}
```

### Subscribe to Risk Alerts
```json
{
  "action": "subscribe",
  "channel": "risk_alerts"
}
```

### Message Format
```json
{
  "channel": "market_data",
  "data": {
    "symbol": "BTCUSDT",
    "price": 42000.50,
    "volume": 15000,
    "timestamp": "2026-02-20T00:00:00Z"
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Symbol 'INVALID' not found",
    "details": {}
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_SYMBOL` | Symbol not found |
| `INVALID_ORDER` | Invalid order parameters |
| `INSUFFICIENT_BALANCE` | Not enough balance |
| `RISK_LIMIT_EXCEEDED` | Position would exceed risk limits |
| `MARKET_CLOSED` | Market is closed |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/market/*` | 100 req/min |
| `/orders` | 50 req/min |
| `/portfolio/*` | 60 req/min |
| `/risk/*` | 60 req/min |
| `/strategy/*` | 30 req/min |

---

## Authentication

API uses Bearer token authentication:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/portfolio
```

---

## SDK Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Get prices
response = requests.get(f"{BASE_URL}/market/prices")
prices = response.json()["prices"]

# Place order
order = {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "quantity": 0.01,
    "price": 42000.00
}
response = requests.post(f"{BASE_URL}/orders", json=order)
```

### JavaScript

```javascript
const BASE_URL = "http://localhost:8000";

// Get prices
const response = await fetch(`${BASE_URL}/market/prices`);
const data = await response.json();

// WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws`);
ws.onopen = () => {
  ws.send(JSON.stringify({
    action: "subscribe",
    channel: "market_data",
    symbols: ["BTCUSDT"]
  }));
};
ws.onmessage = (event) => {
  console.log(JSON.parse(event.data));
};
```
