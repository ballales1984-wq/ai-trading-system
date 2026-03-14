"""
AI Trading System — Locust Load Test Suite
==========================================
Scenari realistici per stress-testing dell'API FastAPI.

Usage:
    # Interactive UI
    locust -f locustfile.py --host=http://localhost:8000

    # Headless (CI/CD)
    locust -f locustfile.py --host=http://localhost:8000 \
           --users=50 --spawn-rate=5 --run-time=2m --headless

Scenari:
  - TradingUser:     Simula un trader attivo (portfolio + ordini)
  - MarketWatcher:   Simula un osservatore dei prezzi (read-only, alta frequenza)
  - RiskMonitor:     Simula un risk manager (metriche VaR, posizioni)
  - AdminUser:       Simula un amministratore (health check, stats)
"""

# pyre-ignore-all-errors[missing-import]
from locust import HttpUser, task, between, events  # type: ignore[attr-defined]
import random
import json
import logging

logger = logging.getLogger(__name__)

# ─── Credenziali demo ────────────────────────────────────────────────────────

DEMO_CREDENTIALS = {"username": "admin", "password": "admin123"}
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]

# ─── Shared token store (populated on login) ─────────────────────────────────

_auth_token: str | None = None


def _get_headers(token: str | None = None) -> dict:
    """Return auth headers if token available."""
    t = token or _auth_token
    if t:
        return {"Authorization": f"Bearer {t}"}
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — Trading User
# Peso: 40% del traffico
# ═════════════════════════════════════════════════════════════════════════════

class TradingUser(HttpUser):
    """
    Simula un trader attivo che:
    1. Legge il portfolio ogni poco
    2. Consulta prezzi di mercato
    3. Crea ordini (con bassa frequenza per non saturare la demo)
    4. Cancella ordini in pending
    """
    weight = 40
    wait_time = between(2, 8)   # pausa realistica tra azioni
    token: str | None = None

    def on_start(self):
        """Login all'avvio del task."""
        resp = self.client.post(
            "/api/v1/auth/login",
            json=DEMO_CREDENTIALS,
            name="/auth/login",
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token")
        else:
            # Tentiamo un altro formato compatibile con FastAPI OAuth2
            resp2 = self.client.post(
                "/api/v1/auth/token",
                data={"username": DEMO_CREDENTIALS["username"], "password": DEMO_CREDENTIALS["password"]},
                name="/auth/token",
            )
            if resp2.status_code == 200:
                self.token = resp2.json().get("access_token")

    # ── Read actions (frequenti) ──────────────────────────────────────────

    @task(5)
    def get_portfolio_summary(self):
        with self.client.get(
            "/api/v1/portfolio/summary/dual",
            headers=_get_headers(self.token),
            name="/portfolio/summary/dual",
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 401):
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(4)
    def get_positions(self):
        self.client.get(
            "/api/v1/portfolio/positions",
            headers=_get_headers(self.token),
            name="/portfolio/positions",
        )

    @task(3)
    def get_market_prices(self):
        self.client.get(
            "/api/v1/market/prices",
            headers=_get_headers(self.token),
            name="/market/prices",
        )

    @task(2)
    def get_orders(self):
        self.client.get(
            "/api/v1/orders",
            headers=_get_headers(self.token),
            name="/orders [list]",
        )

    # ── Write actions (bassa frequenza) ──────────────────────────────────

    @task(1)
    def create_order(self):
        symbol = random.choice(SYMBOLS)
        side = random.choice(["BUY", "SELL"])
        payload = {
            "symbol": symbol,
            "side": side,
            "order_type": "MARKET",
            "quantity": float(f"{random.uniform(0.001, 0.01):.4f}"),
            "broker": "paper",
        }
        with self.client.post(
            "/api/v1/orders",
            json=payload,
            headers=_get_headers(self.token),
            name="/orders [create]",
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 201, 400, 422):
                resp.failure(f"Create order failed: {resp.status_code}")


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Market Watcher (read-only, alta frequenza)
# Peso: 35% del traffico
# ═════════════════════════════════════════════════════════════════════════════

class MarketWatcher(HttpUser):
    """
    Simula un osservatore read-only dei mercati:
    prezzi in real-time, candelabri, order book, sentiment.
    Tipico uso: dashboard pubblica / monitoring passivo.
    """
    weight = 35
    wait_time = between(1, 4)

    @task(6)
    def get_all_prices(self):
        self.client.get("/api/v1/market/prices", name="/market/prices [watcher]")

    @task(4)
    def get_single_price(self):
        symbol = random.choice(SYMBOLS)
        self.client.get(
            f"/api/v1/market/price/{symbol}",
            name="/market/price/[symbol]",
        )

    @task(3)
    def get_candles(self):
        symbol = random.choice(SYMBOLS)
        interval = random.choice(["1m", "5m", "1h"])
        self.client.get(
            f"/api/v1/market/candles/{symbol}",
            params={"interval": interval, "limit": 50},
            name="/market/candles/[symbol]",
        )

    @task(2)
    def get_sentiment(self):
        self.client.get("/api/v1/market/sentiment", name="/market/sentiment")

    @task(1)
    def get_news(self):
        self.client.get(
            "/api/v1/news",
            params={"limit": 10},
            name="/news",
        )


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — Risk Monitor
# Peso: 15% del traffico
# ═════════════════════════════════════════════════════════════════════════════

class RiskMonitor(HttpUser):
    """
    Simula un risk manager che controlla VaR, CVaR,
    correlazioni e performance del portafoglio.
    """
    weight = 15
    wait_time = between(5, 15)
    token: str | None = None

    def on_start(self):
        resp = self.client.post(
            "/api/v1/auth/login",
            json=DEMO_CREDENTIALS,
            name="/auth/login [risk]",
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token")

    @task(4)
    def get_risk_metrics(self):
        self.client.get(
            "/api/v1/risk/metrics",
            headers=_get_headers(self.token),
            name="/risk/metrics",
        )

    @task(3)
    def get_correlation_matrix(self):
        self.client.get(
            "/api/v1/risk/correlation",
            headers=_get_headers(self.token),
            name="/risk/correlation",
        )

    @task(2)
    def get_portfolio_performance(self):
        self.client.get(
            "/api/v1/portfolio/performance",
            headers=_get_headers(self.token),
            name="/portfolio/performance",
        )

    @task(2)
    def get_portfolio_history(self):
        self.client.get(
            "/api/v1/portfolio/history",
            params={"days": 30},
            headers=_get_headers(self.token),
            name="/portfolio/history",
        )

    @task(1)
    def get_allocation(self):
        self.client.get(
            "/api/v1/portfolio/allocation",
            headers=_get_headers(self.token),
            name="/portfolio/allocation",
        )


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — Admin User
# Peso: 10% del traffico
# ═════════════════════════════════════════════════════════════════════════════

class AdminUser(HttpUser):
    """
    Simula un amministratore che verifica health, metriche di sistema
    e stats del rate limiter.
    """
    weight = 10
    wait_time = between(10, 30)

    @task(4)
    def health_check(self):
        with self.client.get("/api/v1/health", name="/health", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") not in ("healthy", "degraded", "ok"):
                    resp.failure(f"Unexpected health status: {data.get('status')}")
            elif resp.status_code == 503:
                # Degraded is acceptable — don't mark as failure
                resp.success()

    @task(3)
    def metrics_endpoint(self):
        self.client.get("/metrics", name="/metrics [prometheus]")

    @task(2)
    def rate_limit_stats(self):
        self.client.get(
            "/api/v1/rate-limit/stats",
            name="/rate-limit/stats",
        )

    @task(1)
    def root_check(self):
        self.client.get("/", name="/ [root]")


# ─── Event hooks ─────────────────────────────────────────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("🚀 AI Trading System load test starting…")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    logger.info(
        "✅ Load test complete — "
        f"Requests: {stats.num_requests} | "
        f"Failures: {stats.num_failures} | "
        f"Avg (ms): {stats.avg_response_time:.0f} | "
        f"RPS: {stats.current_rps:.1f}"
    )
