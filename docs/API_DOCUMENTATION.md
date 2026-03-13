# AI Trading System - API Documentation

## Overview

The AI Trading System provides a comprehensive REST API for algorithmic trading with risk management, portfolio optimization, and real-time market data.

## Base URL

```
Production: https://your-domain.com
Development: http://localhost:8000
```

## Authentication

### JWT Authentication

The API uses JWT (JSON Web Tokens) for authentication.

#### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
    "username": "admin",
    "password": "admin123"
}
```

**Response:**
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer"
}
```

#### Using the Token

Include the token in the Authorization header:

```http
GET /api/v1/portfolio/positions
Authorization: Bearer <your_token>
```

## Security Headers

All responses include the following security headers:

| Header | Value | Description |
|--------|-------|-------------|
| Strict-Transport-Security | max-age=31536000 | HSTS preload |
| X-Frame-Options | DENY | Prevent clickjacking |
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| X-XSS-Protection | 1; mode=block | XSS filtering |
| Referrer-Policy | strict-origin-when-cross-origin | Referrer policy |
| Permissions-Policy | geolocation=(), microphone=(), etc. | Feature permissions |
| Content-Security-Policy | default-src 'self';... | Content security |

## Rate Limiting

Rate limiting is applied to all API endpoints:

| Limit | Requests |
|-------|----------|
| Per Minute | 60 |
| Per Hour | 1,000 |
| Per Day | 10,000 |

Rate limit headers are included in responses:

- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## API Endpoints

### Health & Monitoring

#### GET /health

Basic health check.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "environment": "production"
}
```

#### GET /api/monitoring/health

Detailed health monitoring.

**Response:**
```json
{
    "status": "healthy",
    "monitoring": "enabled",
    "metrics": {
        "requests": {
            "total": 1000,
            "by_endpoint": {...},
            "by_status": {...}
        },
        "performance": {
            "avg_response_time_ms": 45.2,
            "p50_ms": 30.1,
            "p95_ms": 120.5,
            "p99_ms": 250.0
        },
        "errors": {
            "total": 5,
            "rate": 0.5
        }
    },
    "timestamp": "2026-03-13T12:00:00"
}
```

#### GET /api/monitoring/metrics

Get application performance metrics.

#### GET /api/security/headers

Get current security configuration.

### Portfolio Management

#### GET /api/v1/portfolio

Get portfolio summary.

**Response:**
```json
{
    "total_value": 100000.00,
    "positions": [...],
    "cash": 50000.00,
    "pnl": 5000.00,
    "pnl_pct": 5.0
}
```

#### GET /api/v1/portfolio/positions

Get all open positions.

#### GET /api/v1/portfolio/performance

Get portfolio performance metrics.

### Trading

#### GET /api/v1/orders

Get all orders.

#### POST /api/v1/orders

Create a new order.

**Request:**
```json
{
    "symbol": "BTC/USDT",
    "side": "buy",
    "order_type": "limit",
    "quantity": 0.1,
    "price": 45000.00
}
```

#### DELETE /api/v1/orders/{order_id}

Cancel an order.

### Market Data

#### GET /api/v1/market/prices

Get current market prices.

#### GET /api/v1/market/{symbol}

Get price for specific symbol.

#### GET /api/v1/market/candles/{symbol}

Get candlestick data.

### Risk Management

#### GET /api/v1/risk/metrics

Get risk metrics.

**Response:**
```json
{
    "var_95": 2500.00,
    "cvar_95": 3500.00,
    "sharpe_ratio": 1.8,
    "sortino_ratio": 2.1,
    "max_drawdown": 0.15,
    "volatility": 0.12
}
```

#### GET /api/v1/risk/limits

Get current risk limits.

### News & Sentiment

#### GET /api/v1/news

Get latest news.

### Cache

#### GET /api/v1/cache/stats

Get cache statistics.

#### POST /api/v1/cache/clear

Clear cache.

## Audit & Security

### Audit Trail

#### GET /api/audit/events

Query audit events with filters.

**Query:**
- `event_type`: LOGIN|ORDER_CREATED|EMERGENCY_STOP
- `user_id`: Filter user
- `limit`: Max results (default 100)

**Example:**
```
GET /api/audit/events?event_type=ORDER_CREATED&limit=10
```

**Response:**
```json
{
  "events": [
    {
      "id": "uuid-123",
      "timestamp": "2026-03-13T14:30:00Z",
      "event_type": "ORDER_CREATED",
      "user_id": "user456",
      "ip_address": "192.168.1.100",
      "resource_type": "order",
      "resource_id": "order-789",
      "action": "Create order BTCUSDT BUY",
      "details": {
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 0.01,
        "strategy_id": "strat_001"
      },
      "risk_level": "low"
    }
  ],
  "total": 50
}
```

#### GET /api/audit/stats

Audit statistics.

**Response:**
```json
{
  "total_events": 1250,
  "by_type": {"ORDER_CREATED": 500, "LOGIN": 300, "EMERGENCY_STOP": 1},
  "by_user": {"user456": 800},
  "violations": 5,
  "by_risk_level": {"low": 1100, "medium": 120, "high": 25, "critical": 5}
}
```

### Rate Limiting Stats

#### GET /api/v1/rate-limit/stats

Current rate limit status.

**Response:**
```json
{
  "client_id": "192.168.1.100",
  "minute_limit": 60,
  "minute_remaining": 45,
  "hour_remaining": 950,
  "day_remaining": 9800,
  "reset_minute": 1741975200
}
```

### OpenAPI/Swagger

Access interactive docs:
- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI JSON:** `/openapi.json`


### Performance

#### GET /api/performance/metrics

Get function performance metrics.

#### GET /api/performance/slowest

Get slowest functions.

#### GET /api/performance/cache/stats

Get cache performance statistics.

## Error Responses

### 400 Bad Request

```json
{
    "error": "Invalid request",
    "details": {...}
}
```

### 401 Unauthorized

```json
{
    "error": "Authentication required",
    "detail": "Not authenticated"
}
```

### 403 Forbidden

```json
{
    "error": "Access denied",
    "detail": "Insufficient permissions"
}
```

### 404 Not Found

```json
{
    "error": "Resource not found",
    "detail": "The requested resource does not exist"
}
```

### 429 Too Many Requests

```json
{
    "error": "Rate limit exceeded",
    "message": "Too many requests. Please slow down.",
    "retry_after": 60
}
```

### 500 Internal Server Error

```json
{
    "error": "Internal server error",
    "detail": "An unexpected error occurred"
}
```

## WebSocket API

Real-time data is available via WebSocket:

```javascript
const ws = new WebSocket('wss://your-domain.com/ws');

// Subscribe to market data
ws.send(JSON.stringify({
    action: 'subscribe',
    channel: 'market/BTC-USDT'
}));

// Receive updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

## SDK Usage

### Python

```python
from ai_trading import TradingClient

client = TradingClient(
    api_key="your_api_key",
    secret_key="your_secret_key"
)

# Get portfolio
portfolio = client.portfolio.get()

# Place order
order = client.orders.create(
    symbol="BTC/USDT",
    side="buy",
    quantity=0.1
)
```

### JavaScript

```javascript
import { TradingClient } from 'ai-trading';

const client = new TradingClient({
    apiKey: 'your_api_key',
    secretKey: 'your_secret_key'
});

// Get portfolio
const portfolio = await client.portfolio.get();

// Place order
const order = await client.orders.create({
    symbol: 'BTC/USDT',
    side: 'buy',
    quantity: 0.1
});
```

## Rate Limit Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Server Error |

## Changelog

### v1.0.0 (2026-03-13)

- Initial release
- JWT authentication
- Portfolio management
- Order execution
- Risk management
- Real-time monitoring
- Comprehensive security headers
