# AI Trading System - API Reference

## 🌐 REST API Endpoints

Base URL: `http://localhost:8000`

### Health & Status

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-20T00:00:00Z",
  "version": "2.0.0"
}
```

#### GET /ready
Readiness check for Kubernetes.

**Response:**
```json
{
  "ready": true,
  "services": {
    "database": "connected",
    "redis": "connected",
    "exchange": "connected"
  }
}
```

---

## 📊 Market Data

#### GET /api/market/price/{symbol}
Get current price for a symbol.

**Parameters:**
- `symbol` (path): Trading pair (e.g., BTCUSDT)

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "price": 42000.50,
  "timestamp": "2026-02-20T00:00:00Z",
  "source": "binance"
}
```

#### GET /api/market/prices
Get all cached prices.

**Response:**
```json
{
  "prices": {
    "BTCUSDT": 42000.50,
    "ETHUSDT": 3000.25,
    "BNBUSDT": 350.10
  },
  "timestamp": "2026-02-20T00:00:00Z"
}
```

#### GET /api/market/history/{symbol}
Get price history.

**Parameters:**
- `symbol` (path): Trading pair
- `limit` (query): Number of entries (default: 100)

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "history": [
    {"price": 42000.50, "timestamp": "2026-02-20T00:00:00Z"},
    {"price": 41950.00, "timestamp": "2026-02-19T23:59:00Z"}
  ]
}
```

---

## 📈 Signals

#### GET /api/signals
Get recent trading signals.

**Parameters:**
- `symbol` (query, optional): Filter by symbol
- `limit` (query): Number of signals (default: 50)

**Response:**
```json
{
  "signals": [
    {
      "id": "sig_001",
      "symbol": "BTCUSDT",
      "signal_type": "BUY",
      "strength": "strong",
      "confidence": 0.85,
      "price": 42000.50,
      "strategy": "momentum",
      "timestamp": "2026-02-20T00:00:00Z"
    }
  ]
}
```

#### POST /api/signals/generate
Manually trigger signal generation.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "strategies": ["momentum", "mean_reversion"]
}
```

**Response:**
```json
{
  "signal": {
    "symbol": "BTCUSDT",
    "signal_type": "BUY",
    "confidence": 0.82,
    "metadata": {
      "momentum": 0.03,
      "z_score": -1.5
    }
  }
}
```

---

## 📦 Orders

#### GET /api/orders
Get order history.

**Parameters:**
- `symbol` (query, optional): Filter by symbol
- `status` (query, optional): Filter by status
- `limit` (query): Number of orders (default: 50)

**Response:**
```json
{
  "orders": [
    {
      "id": "ord_001",
      "symbol": "BTCUSDT",
      "side": "BUY",
      "type": "LIMIT",
      "quantity": 0.1,
      "price": 42000.00,
      "status": "FILLED",
      "filled_quantity": 0.1,
      "timestamp": "2026-02-20T00:00:00Z"
    }
  ]
}
```

#### POST /api/orders
Create a new order.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "type": "LIMIT",
  "quantity": 0.1,
  "price": 42000.00
}
```

**Response:**
```json
{
  "order_id": "ord_002",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "status": "PENDING",
  "timestamp": "2026-02-20T00:00:00Z"
}
```

#### DELETE /api/orders/{order_id}
Cancel an order.

**Response:**
```json
{
  "order_id": "ord_002",
  "status": "CANCELLED",
  "timestamp": "2026-02-20T00:00:00Z"
}
```

---

## 💼 Portfolio

#### GET /api/portfolio
Get current portfolio.

**Response:**
```json
{
  "total_value": 100000.00,
  "cash": 50000.00,
  "positions_value": 50000.00,
  "daily_pnl": 1500.00,
  "total_pnl": 5000.00,
  "positions": [
    {
      "symbol": "BTCUSDT",
      "quantity": 1.0,
      "entry_price": 40000.00,
      "current_price": 42000.00,
      "unrealized_pnl": 2000.00,
      "unrealized_pnl_pct": 5.0
    }
  ]
}
```

#### GET /api/portfolio/performance
Get portfolio performance metrics.

**Response:**
```json
{
  "total_return": 0.15,
  "annualized_return": 0.25,
  "sharpe_ratio": 1.8,
  "sortino_ratio": 2.1,
  "max_drawdown": -0.08,
  "win_rate": 0.65,
  "profit_factor": 1.8
}
```

---

## 🛡️ Risk

#### GET /api/risk/{symbol}
Get risk metrics for a symbol.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "var_95": -0.05,
  "var_99": -0.08,
  "cvar_95": -0.07,
  "max_drawdown": -0.10,
  "volatility": 0.65,
  "sharpe_ratio": 1.5,
  "risk_level": "medium"
}
```

#### GET /api/risk/portfolio
Get portfolio-level risk metrics.

**Response:**
```json
{
  "portfolio_var": -0.04,
  "portfolio_cvar": -0.06,
  "diversification_ratio": 1.3,
  "correlation_matrix": {
    "BTCUSDT-ETHUSDT": 0.85
  },
  "risk_contribution": {
    "BTCUSDT": 0.6,
    "ETHUSDT": 0.4
  }
}
```

#### POST /api/risk/check
Check if a trade passes risk limits.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 1.0,
  "price": 42000.00
}
```

**Response:**
```json
{
  "approved": true,
  "warnings": [],
  "position_size_recommendation": 0.5
}
```

---

## 🎲 Monte Carlo

#### GET /api/montecarlo/{symbol}
Get Monte Carlo simulation results.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "level": "conditional",
  "mean_price": 43000.00,
  "std_price": 2000.00,
  "percentiles": {
    "p5": 40000.00,
    "p25": 41500.00,
    "p50": 43000.00,
    "p75": 44500.00,
    "p95": 46000.00
  },
  "var_95": -0.05,
  "probability_up": 0.65,
  "timestamp": "2026-02-20T00:00:00Z"
}
```

#### POST /api/montecarlo/run
Run Monte Carlo simulation.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "n_paths": 1000,
  "n_steps": 50,
  "time_horizon": 1.0
}
```

**Response:**
```json
{
  "job_id": "mc_001",
  "status": "running",
  "estimated_completion": "2026-02-20T00:01:00Z"
}
```

---

## 🧠 Strategy

#### GET /api/strategy/list
List available strategies.

**Response:**
```json
{
  "strategies": [
    {
      "name": "momentum",
      "enabled": true,
      "weight": 0.4,
      "performance": {
        "win_rate": 0.62,
        "total_trades": 150
      }
    },
    {
      "name": "mean_reversion",
      "enabled": true,
      "weight": 0.3,
      "performance": {
        "win_rate": 0.58,
        "total_trades": 120
      }
    }
  ]
}
```

#### PUT /api/strategy/{name}/config
Update strategy configuration.

**Request Body:**
```json
{
  "enabled": true,
  "weight": 0.5,
  "params": {
    "momentum_period": 15,
    "momentum_threshold": 0.025
  }
}
```

**Response:**
```json
{
  "name": "momentum",
  "updated": true,
  "config": {
    "enabled": true,
    "weight": 0.5,
    "params": {
      "momentum_period": 15,
      "momentum_threshold": 0.025
    }
  }
}
```

---

## 🧬 AutoML

#### POST /api/automl/optimize
Start parameter optimization.

**Request Body:**
```json
{
  "strategy": "momentum",
  "param_space": {
    "momentum_period": [5, 30],
    "momentum_threshold": [0.01, 0.05]
  },
  "generations": 10,
  "population_size": 20
}
```

**Response:**
```json
{
  "job_id": "opt_001",
  "status": "running",
  "estimated_time": 300
}
```

#### GET /api/automl/status/{job_id}
Get optimization status.

**Response:**
```json
{
  "job_id": "opt_001",
  "status": "completed",
  "generation": 10,
  "best_params": {
    "momentum_period": 12,
    "momentum_threshold": 0.023
  },
  "best_fitness": 0.85
}
```

---

## 📡 WebSocket

### Connect
```
ws://localhost:8000/ws
```

### Subscribe to Channels

```json
{
  "action": "subscribe",
  "channels": ["prices", "signals", "orders"]
}
```

### Message Types

#### Price Update
```json
{
  "type": "price_update",
  "data": {
    "symbol": "BTCUSDT",
    "price": 42000.50,
    "timestamp": "2026-02-20T00:00:00Z"
  }
}
```

#### Signal Generated
```json
{
  "type": "signal",
  "data": {
    "symbol": "BTCUSDT",
    "signal_type": "BUY",
    "confidence": 0.85
  }
}
```

#### Order Update
```json
{
  "type": "order_update",
  "data": {
    "order_id": "ord_001",
    "status": "FILLED",
    "filled_quantity": 0.1
  }
}
```

---

## 🔐 Authentication

All API endpoints require authentication via JWT token.

### Get Token

```bash
POST /api/auth/token
{
  "username": "user",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Use Token

```bash
curl -H "Authorization: Bearer eyJ..." http://localhost:8000/api/portfolio
```

---

## 📝 Error Responses

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
| `RISK_LIMIT` | Risk limit exceeded |
| `INSUFFICIENT_BALANCE` | Not enough balance |
| `MARKET_CLOSED` | Market is closed |
| `RATE_LIMIT` | Too many requests |

---

## 🚦 Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/market/*` | 100/min |
| `/api/orders/*` | 30/min |
| `/api/signals/*` | 60/min |
| `/api/portfolio/*` | 60/min |
| `/api/risk/*` | 60/min |

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1708400000
```
