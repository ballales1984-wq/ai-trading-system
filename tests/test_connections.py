"""
Connection Testing Script
========================
Test all system connections: Brokers, Database, Redis.

Usage:
    python test_connections.py
    python test_connections.py --skip-brokers
    python test_connections.py --only redis,database
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ConnectionTester:
    """Test all system connections."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def print_header(self, title: str):
        """Print section header."""
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
    
    def print_result(self, name: str, success: bool, details: str = None):
        """Print test result."""
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} | {name}")
        if details:
            print(f"         | {details}")
    
    async def test_redis(self) -> bool:
        """Test Redis connection."""
        self.print_header("REDIS CACHE")
        
        try:
            from app.core.cache import RedisCacheManager
            
            cache = RedisCacheManager()
            connected = await cache.connect()
            
            if connected:
                health = await cache.health_check()
                
                self.print_result(
                    "Redis Connection",
                    health["connected"],
                    f"Latency: {health.get('latency_ms', 'N/A')}ms"
                )
                
                if health["connected"]:
                    # Test basic operations
                    test_key = "test:connection"
                    test_value = {"timestamp": datetime.now(timezone.utc).isoformat()}
                    
                    # Set
                    set_ok = await cache.set(test_key, test_value, ttl=60)
                    self.print_result("Redis SET", set_ok)
                    
                    # Get
                    get_value = await cache.get(test_key)
                    get_ok = get_value is not None
                    self.print_result("Redis GET", get_ok)
                    
                    # Delete
                    del_ok = await cache.delete(test_key) > 0
                    self.print_result("Redis DELETE", del_ok)
                    
                    # Store results
                    self.results["redis"] = {
                        "connected": True,
                        "latency_ms": health.get("latency_ms"),
                        "memory_used": health.get("memory_used"),
                        "keys_count": health.get("keys_count"),
                    }
                    
                await cache.disconnect()
                return True
            else:
                self.print_result("Redis Connection", False, "Failed to connect")
                self.results["redis"] = {"connected": False, "error": "Connection failed"}
                return False
                
        except ImportError as e:
            self.print_result("Redis Module", False, f"Missing dependency: {e}")
            self.results["redis"] = {"connected": False, "error": str(e)}
            return False
        except Exception as e:
            self.print_result("Redis", False, str(e))
            self.results["redis"] = {"connected": False, "error": str(e)}
            return False
    
    async def test_database(self) -> bool:
        """Test PostgreSQL/TimescaleDB connection."""
        self.print_header("POSTGRESQL / TIMESCALEDB")
        
        try:
            from app.core.database import DatabaseManager, AsyncDatabaseManager
            from sqlalchemy import text
            
            # Test sync connection
            sync_db = DatabaseManager()
            sync_connected = sync_db.connect()
            
            if sync_connected:
                health = sync_db.health_check()
                
                self.print_result(
                    "PostgreSQL Connection (Sync)",
                    health["connected"],
                    f"Pool: {health.get('pool_size', 0)} connections"
                )
                
                if health.get("database_version"):
                    print(f"         | Version: {health['database_version'][:50]}...")
                
                self.print_result(
                    "TimescaleDB Extension",
                    health.get("timescale_enabled", False),
                    f"Version: {health.get('timescale_version', 'N/A')}"
                )
                
                # Test query
                try:
                    with sync_db.session() as session:
                        result = session.execute(text("SELECT current_database(), current_user"))
                        row = result.fetchone()
                        self.print_result("Database Query", True, f"DB: {row[0]}, User: {row[1]}")
                except Exception as e:
                    self.print_result("Database Query", False, str(e))
                
                self.results["database_sync"] = {
                    "connected": True,
                    "pool_size": health.get("pool_size"),
                    "timescale_enabled": health.get("timescale_enabled"),
                }
                
                sync_db.disconnect()
            else:
                self.print_result("PostgreSQL Connection (Sync)", False, "Failed to connect")
                self.results["database_sync"] = {"connected": False}
            
            # Test async connection
            async_db = AsyncDatabaseManager()
            async_connected = await async_db.connect()
            
            if async_connected:
                health = await async_db.health_check()
                
                self.print_result(
                    "PostgreSQL Connection (Async)",
                    health["connected"],
                    f"Pool: {health.get('pool_size', 0)} connections"
                )
                
                self.results["database_async"] = {
                    "connected": True,
                    "pool_size": health.get("pool_size"),
                }
                
                await async_db.disconnect()
            else:
                self.print_result("PostgreSQL Connection (Async)", False, "Failed to connect")
                self.results["database_async"] = {"connected": False}
            
            return sync_connected or async_connected
            
        except ImportError as e:
            self.print_result("Database Module", False, f"Missing dependency: {e}")
            self.results["database"] = {"connected": False, "error": str(e)}
            return False
        except Exception as e:
            self.print_result("Database", False, str(e))
            self.results["database"] = {"connected": False, "error": str(e)}
            return False
    
    async def test_brokers(self) -> bool:
        """Test broker connections."""
        self.print_header("BROKER CONNECTIONS")
        
        results = {}
        all_passed = True
        
        # Test Binance
        try:
            from app.execution.broker_connector import BinanceConnector, BrokerOrder
            
            binance = BinanceConnector(testnet=True)
            connected = await binance.connect()
            
            self.print_result(
                "Binance (Testnet)",
                connected,
                "Paper trading mode" if connected else "Connection failed"
            )
            
            if connected:
                # Test public endpoint
                try:
                    price = await binance.get_symbol_price("BTCUSDT")
                    self.print_result("Binance Market Data", True, f"BTC/USDT: ${price:,.2f}")
                except Exception as e:
                    self.print_result("Binance Market Data", False, str(e))
            
            results["binance"] = {"connected": connected}
            await binance.disconnect()
            
        except ImportError as e:
            self.print_result("Binance Module", False, f"Missing dependency: {e}")
            results["binance"] = {"connected": False, "error": str(e)}
            all_passed = False
        except Exception as e:
            self.print_result("Binance", False, str(e))
            results["binance"] = {"connected": False, "error": str(e)}
            all_passed = False
        
        # Test Bybit
        try:
            from app.execution.broker_connector import BybitConnector
            
            bybit = BybitConnector(testnet=True)
            connected = await bybit.connect()
            
            self.print_result(
                "Bybit (Testnet)",
                connected,
                "Paper trading mode" if connected else "Connection failed"
            )
            
            if connected:
                try:
                    price = await bybit.get_symbol_price("BTCUSDT")
                    self.print_result("Bybit Market Data", True, f"BTC/USDT: ${price:,.2f}")
                except Exception as e:
                    self.print_result("Bybit Market Data", False, str(e))
            
            results["bybit"] = {"connected": connected}
            await bybit.disconnect()
            
        except ImportError as e:
            self.print_result("Bybit Module", False, f"Missing dependency: {e}")
            results["bybit"] = {"connected": False, "error": str(e)}
            all_passed = False
        except Exception as e:
            self.print_result("Bybit", False, str(e))
            results["bybit"] = {"connected": False, "error": str(e)}
            all_passed = False
        
        # Test Paper Trading
        try:
            from app.execution.broker_connector import PaperTradingConnector
            
            paper = PaperTradingConnector(initial_balance=100000)
            connected = await paper.connect()
            
            self.print_result(
                "Paper Trading",
                connected,
                f"Balance: ${paper.balance['USDT']:,.2f}"
            )
            
            results["paper"] = {"connected": connected}
            await paper.disconnect()
            
        except Exception as e:
            self.print_result("Paper Trading", False, str(e))
            results["paper"] = {"connected": False, "error": str(e)}
            all_passed = False
        
        self.results["brokers"] = results
        return all_passed
    
    def test_config(self) -> bool:
        """Test configuration loading."""
        self.print_header("CONFIGURATION")
        
        try:
            from app.core.config import settings
            
            self.print_result("Config Loading", True)
            
            # Check critical settings
            checks = [
                ("Database URL", bool(settings.database_url), 
                 f"postgresql://...@{settings.database_url.split('@')[-1]}" if settings.database_url else "Not set"),
                ("Redis URL", bool(settings.redis_url), settings.redis_url),
                ("Binance API Key", bool(settings.binance_api_key), 
                 "Set" if settings.binance_api_key else "Not set (optional)"),
                ("Secret Key", settings.secret_key != "dev-secret-key", 
                 "Production" if settings.secret_key != "dev-secret-key" else "Development"),
            ]
            
            for name, passed, detail in checks:
                self.print_result(name, passed, detail)
            
            self.results["config"] = {"loaded": True}
            return True
            
        except Exception as e:
            self.print_result("Configuration", False, str(e))
            self.results["config"] = {"loaded": False, "error": str(e)}
            return False
    
    def print_summary(self):
        """Print test summary."""
        self.print_header("SUMMARY")
        
        total = 0
        passed = 0
        
        for category, data in self.results.items():
            if isinstance(data, dict):
                if "connected" in data:
                    total += 1
                    if data["connected"]:
                        passed += 1
                elif "loaded" in data:
                    total += 1
                    if data["loaded"]:
                        passed += 1
            elif isinstance(data, dict):
                for sub_key, sub_data in data.items():
                    if isinstance(sub_data, dict) and "connected" in sub_data:
                        total += 1
                        if sub_data["connected"]:
                            passed += 1
        
        status = "[ALL PASSED]" if passed == total else f"[{passed}/{total} PASSED]"
        print(f"\n  {status}")
        print(f"  Tested at: {datetime.now(timezone.utc).isoformat()}")
        print()


async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test system connections")
    parser.add_argument("--skip-brokers", action="store_true", help="Skip broker tests")
    parser.add_argument("--skip-redis", action="store_true", help="Skip Redis tests")
    parser.add_argument("--skip-database", action="store_true", help="Skip database tests")
    parser.add_argument("--only", type=str, help="Only test specified components (comma-separated)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  AI TRADING SYSTEM - CONNECTION TEST")
    print("=" * 60)
    print(f"  Started at: {datetime.now(timezone.utc).isoformat()}")
    
    tester = ConnectionTester()
    
    # Determine what to test
    if args.only:
        only_tests = [t.strip().lower() for t in args.only.split(",")]
        test_config = "config" in only_tests
        test_redis = "redis" in only_tests
        test_database = "database" in only_tests or "postgres" in only_tests or "timescale" in only_tests
        test_brokers = "brokers" in only_tests or "broker" in only_tests
    else:
        test_config = True
        test_redis = not args.skip_redis
        test_database = not args.skip_database
        test_brokers = not args.skip_brokers
    
    # Run tests
    if test_config:
        tester.test_config()
    
    if test_redis:
        await tester.test_redis()
    
    if test_database:
        await tester.test_database()
    
    if test_brokers:
        await tester.test_brokers()
    
    # Print summary
    tester.print_summary()
    
    # Return exit code
    all_connected = all(
        data.get("connected", data.get("loaded", False))
        for data in tester.results.values()
        if isinstance(data, dict) and ("connected" in data or "loaded" in data)
    )
    
    sys.exit(0 if all_connected else 1)


if __name__ == "__main__":
    asyncio.run(main())
