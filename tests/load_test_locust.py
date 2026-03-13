"""
Locust load testing for AI Trading System API.

Usage:
    locust -f tests/load_test_locust.py --host=http://localhost:8000
    
With authentication:
    locust -f tests/load_test_locust.py --host=http://localhost:8000 -u testuser -p testpass
    
Headless mode:
    locust -f tests/load_test_locust.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 60s
"""

import random
from locust import HttpUser, task, between, tag, events


class TradingUser(HttpUser):
    """Simulated trading user behavior for load testing."""
    
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Login to get token (if needed)
        response = self.client.post("/api/v1/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        if response.status_code == 200:
            self.token = response.json().get("access_token")
        else:
            self.token = None
    
    @task(3)
    @tag('market')
    def get_market_data(self):
        """Get market data - most frequent operation."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        symbol = random.choice(symbols)
        self.client.get(f"/api/v1/market/candle/{symbol}?interval=1h&limit=100")
    
    @task(2)
    @tag('portfolio')
    def get_portfolio(self):
        """Get portfolio positions."""
        self.client.get("/api/v1/portfolio/positions")
    
    @task(2)
    @tag('portfolio')
    def get_performance(self):
        """Get portfolio performance metrics."""
        self.client.get("/api/v1/portfolio/performance")
    
    @task(1)
    @tag('orders')
    def get_orders(self):
        """Get user orders."""
        self.client.get("/api/v1/orders")
    
    @task(1)
    @tag('orders')
    def create_order(self):
        """Create a test order."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        symbol = random.choice(symbols)
        self.client.post("/api/v1/orders", json={
            "symbol": symbol,
            "side": random.choice(["buy", "sell"]),
            "type": "market",
            "quantity": round(random.uniform(0.001, 0.1), 4)
        })
    
    @task(1)
    @tag('health')
    def check_health(self):
        """Check API health."""
        self.client.get("/health")
    
    @task(1)
    @tag('monitoring')
    def get_metrics(self):
        """Get monitoring metrics."""
        self.client.get("/api/monitoring/metrics")
    
    @task(1)
    @tag('risk')
    def get_risk_metrics(self):
        """Get risk metrics."""
        self.client.get("/api/v1/risk/metrics")


class AdminUser(HttpUser):
    """Admin user for testing administrative endpoints."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Admin login."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "adminpass"
        })
        if response.status_code == 200:
            self.token = response.json().get("access_token")
    
    @task(2)
    @tag('admin')
    def get_all_users(self):
        """Get all users (admin only)."""
        self.client.get("/api/v1/admin/users")
    
    @task(1)
    @tag('admin')
    def get_system_stats(self):
        """Get system statistics."""
        self.client.get("/api/v1/admin/stats")


# Event handlers for logging
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start."""
    print("Load test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test results."""
    print("Load test completed!")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time}ms")
