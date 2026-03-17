"""
OpenClaw API Wrapper for ai-trading-system
==========================================
Python wrapper to facilitate communication between OpenClaw agents
and the ai-trading-system FastAPI backend.

Usage:
    from api_wrapper import QuantTradingAPI
    
    api = QuantTradingAPI(base_url="http://localhost:8000")
    api.login("admin", "admin123")
    
    risk = api.get_risk_metrics()
    print(f"VaR: {risk['var_1d']}")
"""

import json
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class QuantTradingAPI:
    """
    API Wrapper for ai-trading-system.
    
    Provides a clean Python interface to all quantitative trading endpoints.
    """
    
    base_url: str = "http://localhost:8000/api/v1"
    timeout: int = 30
    _token: Optional[str] = None
    
    def __post_init__(self):
        """Load config from environment if available."""
        self.base_url = os.getenv("AI_TRADING_API_URL", self.base_url)
        self._token = os.getenv("AI_TRADING_API_TOKEN", None)
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate and get JWT token.
        
        Args:
            username: API username
            password: API password
            
        Returns:
            Dict with access_token and token_type
        """
        import requests
        
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        self._token = data.get("access_token")
        return data
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
    
    # =========================================================================
    # Portfolio Endpoints
    # =========================================================================
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get dual portfolio summary (real + simulated)."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/portfolio/summary/dual",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions, optionally filtered by symbol."""
        import requests
        
        params = {"symbol": symbol} if symbol else {}
        response = requests.get(
            f"{self.base_url}/portfolio/positions",
            headers=self._get_headers(),
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_performance(self) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/portfolio/performance",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # Risk Endpoints
    # =========================================================================
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current portfolio risk metrics (VaR, CVaR, volatility, etc.)."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/risk/metrics",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_risk_limits(self) -> List[Dict[str, Any]]:
        """Get current risk limits and their status."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/risk/limits",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def check_order_risk(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """
        Pre-execution risk check for an order.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            side: "BUY" or "SELL"
            quantity: Order quantity
            price: Order price
            
        Returns:
            Risk check result with approved status and reasons
        """
        import requests
        
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "price": price
        }
        
        response = requests.post(
            f"{self.base_url}/risk/check_order",
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def run_monte_carlo(
        self,
        simulations: int = 10000,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for VaR calculation.
        
        Args:
            simulations: Number of Monte Carlo simulations (1000-100000)
            confidence: Confidence level (0.9-0.99)
            
        Returns:
            VaR, CVaR, and distribution percentiles
        """
        import requests
        
        params = {
            "simulations": simulations,
            "confidence": confidence
        }
        
        response = requests.get(
            f"{self.base_url}/risk/var/monte_carlo",
            headers=self._get_headers(),
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def run_stress_test(self) -> Dict[str, Any]:
        """Run stress test scenarios on portfolio."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/risk/stress_test",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Get asset correlation matrix."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/risk/correlation",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_monte_carlo_distribution(
        self,
        simulations: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get Monte Carlo simulation distribution percentiles."""
        import requests
        
        params = {"simulations": simulations}
        
        response = requests.get(
            f"{self.base_url}/risk/monte-carlo",
            headers=self._get_headers(),
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_drawdown(self) -> List[Dict[str, Any]]:
        """Get drawdown data over time."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/risk/drawdown",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # Market Data Endpoints
    # =========================================================================
    
    def get_market_prices(self) -> List[Dict[str, Any]]:
        """Get current market prices for all symbols."""
        import requests
        
        response = requests.get(
            f"{self.base_url}/market/prices",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get OHLCV candle data for a symbol."""
        import requests
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(
            f"{self.base_url}/market/candles/{symbol}",
            headers=self._get_headers(),
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book data for a symbol."""
        import requests
        
        params = {"limit": limit}
        
        response = requests.get(
            f"{self.base_url}/market/orderbook/{symbol}",
            headers=self._get_headers(),
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # Order Management
    # =========================================================================
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a new order."""
        import requests
        
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity
        }
        if price:
            payload["price"] = price
        
        response = requests.post(
            f"{self.base_url}/orders",
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_orders(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get orders, optionally filtered."""
        import requests
        
        params = {}
        if status:
            params["status"] = status
        if symbol:
            params["symbol"] = symbol
        
        response = requests.get(
            f"{self.base_url}/orders",
            headers=self._get_headers(),
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # High-level Analysis Functions
    # =========================================================================
    
    def analyze_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """
        Complete trade analysis combining market data and risk metrics.
        
        Returns structured analysis with recommendations.
        """
        # Get all data in parallel
        import concurrent.futures
        
        results = {}
        
        def fetch_data():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'positions': executor.submit(self.get_positions),
                    'risk_metrics': executor.submit(self.get_risk_metrics),
                    'order_risk': executor.submit(
                        self.check_order_risk, symbol, side, quantity, price
                    )
                }
                for key, future in futures.items():
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        results[key] = {"error": str(e)}
        
        fetch_data()
        
        return {
            "trade": {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": quantity * price
            },
            "risk_check": results.get('order_risk', {}),
            "current_positions": results.get('positions', []),
            "portfolio_risk": results.get('risk_metrics', {}),
            "recommendation": self._generate_recommendation(results)
        }
    
    def _generate_recommendation(self, results: Dict[str, Any]) -> str:
        """Generate trading recommendation based on analysis."""
        risk_check = results.get('order_risk', {})
        
        if not risk_check:
            return "Unable to complete risk analysis"
        
        if risk_check.get('approved'):
            return "✅ Trade approved - passes all risk gates"
        else:
            reasons = risk_check.get('rejection_reasons', [])
            return f"❌ Trade rejected: {'; '.join(reasons)}"
    
    def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """
        Get complete portfolio snapshot for display.
        Combines portfolio, positions, and risk data.
        """
        summary = self.get_portfolio_summary()
        positions = self.get_positions()
        risk_metrics = self.get_risk_metrics()
        risk_limits = self.get_risk_limits()
        
        return {
            "summary": summary,
            "positions": positions,
            "risk_metrics": risk_metrics,
            "risk_limits": risk_limits,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI for testing the API wrapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quant Trading API CLI")
    parser.add_argument("--url", default="http://localhost:8000/api/v1", help="API base URL")
    parser.add_argument("--username", default="admin", help="API username")
    parser.add_argument("--password", default="admin123", help="API password")
    parser.add_argument("command", choices=["risk", "portfolio", "positions", "montecarlo", "analyze"])
    parser.add_argument("--symbol", help="Symbol for analysis")
    parser.add_argument("--side", help="Side (BUY/SELL)")
    parser.add_argument("--quantity", type=float, help="Quantity")
    parser.add_argument("--price", type=float, help="Price")
    
    args = parser.parse_args()
    
    api = QuantTradingAPI(base_url=args.url)
    api.login(args.username, args.password)
    
    if args.command == "risk":
        print(json.dumps(api.get_risk_metrics(), indent=2))
    elif args.command == "portfolio":
        print(json.dumps(api.get_portfolio_summary(), indent=2))
    elif args.command == "positions":
        print(json.dumps(api.get_positions(args.symbol), indent=2))
    elif args.command == "montecarlo":
        print(json.dumps(api.run_monte_carlo(), indent=2))
    elif args.command == "analyze":
        if not all([args.symbol, args.side, args.quantity, args.price]):
            print("Error: --symbol, --side, --quantity, --price required for analyze")
            return
        print(json.dumps(api.analyze_trade(args.symbol, args.side, args.quantity, args.price), indent=2))


if __name__ == "__main__":
    main()
