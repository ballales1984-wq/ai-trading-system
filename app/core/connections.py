"""
Connection Manager
==================
Unified connection management for all system components.
Provides centralized initialization and health monitoring.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from app.core.config import settings


logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Unified connection manager for all system components.
    
    Manages:
    - PostgreSQL/TimescaleDB (sync and async)
    - Redis cache
    - Broker connections (Binance, Bybit, Paper)
    
    Usage:
        manager = ConnectionManager()
        await manager.initialize()
        
        # Use connections
        cache = manager.cache
        db = manager.database
        
        # Health check
        health = await manager.health_check()
        
        # Cleanup
        await manager.shutdown()
    """
    
    def __init__(self):
        self._initialized = False
        
        # Connection instances
        self._cache = None
        self._database = None
        self._async_database = None
        self._brokers: Dict[str, Any] = {}
    
    @property
    def cache(self):
        """Get Redis cache manager."""
        return self._cache
    
    @property
    def database(self):
        """Get sync database manager."""
        return self._database
    
    @property
    def async_database(self):
        """Get async database manager."""
        return self._async_database
    
    @property
    def brokers(self) -> Dict[str, Any]:
        """Get broker connections."""
        return self._brokers
    
    async def initialize(
        self,
        connect_redis: bool = True,
        connect_database: bool = True,
        connect_brokers: bool = False,
    ) -> bool:
        """
        Initialize all connections.
        
        Args:
            connect_redis: Connect to Redis cache
            connect_database: Connect to PostgreSQL
            connect_brokers: Connect to trading brokers
        
        Returns:
            True if all requested connections succeeded
        """
        logger.info("Initializing connections...")
        
        results = {}
        
        # Initialize Redis
        if connect_redis:
            results["redis"] = await self._init_redis()
        
        # Initialize Database
        if connect_database:
            results["database_sync"] = await self._init_database_sync()
            results["database_async"] = await self._init_database_async()
        
        # Initialize Brokers
        if connect_brokers:
            results["brokers"] = await self._init_brokers()
        
        self._initialized = True
        
        # Log results
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        logger.info(f"Connections initialized: {success_count}/{total_count} successful")
        
        return all(results.values())
    
    async def _init_redis(self) -> bool:
        """Initialize Redis connection."""
        try:
            from app.core.cache import RedisCacheManager
            
            self._cache = RedisCacheManager()
            connected = await self._cache.connect()
            
            if connected:
                logger.info("Redis cache connected")
            else:
                logger.warning("Redis cache connection failed")
            
            return connected
            
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            return False
    
    async def _init_database_sync(self) -> bool:
        """Initialize sync database connection."""
        try:
            from app.core.database import DatabaseManager
            
            self._database = DatabaseManager()
            connected = self._database.connect()
            
            if connected:
                logger.info("Sync database connected")
            else:
                logger.warning("Sync database connection failed")
            
            return connected
            
        except Exception as e:
            logger.error(f"Sync database initialization error: {e}")
            return False
    
    async def _init_database_async(self) -> bool:
        """Initialize async database connection."""
        try:
            from app.core.database import AsyncDatabaseManager
            
            self._async_database = AsyncDatabaseManager()
            connected = await self._async_database.connect()
            
            if connected:
                logger.info("Async database connected")
            else:
                logger.warning("Async database connection failed")
            
            return connected
            
        except Exception as e:
            logger.error(f"Async database initialization error: {e}")
            return False
    
    async def _init_brokers(self) -> bool:
        """Initialize broker connections."""
        try:
            from app.execution.broker_connector import (
                BinanceConnector,
                BybitConnector,
                PaperTradingConnector,
                BrokerFactory,
            )
            
            results = {}
            
            # Paper trading (always available)
            paper = PaperTradingConnector(initial_balance=settings.paper_initial_balance)
            await paper.connect()
            self._brokers["paper"] = paper
            results["paper"] = True
            
            # Binance (if API keys configured)
            if settings.binance_api_key:
                binance = BinanceConnector(
                    api_key=settings.binance_api_key,
                    secret_key=settings.binance_secret_key,
                    testnet=settings.binance_testnet,
                )
                connected = await binance.connect()
                self._brokers["binance"] = binance
                results["binance"] = connected
            else:
                logger.info("Binance API keys not configured, skipping")
            
            # Bybit (if API keys configured)
            if settings.bybit_api_key:
                bybit = BybitConnector(
                    api_key=settings.bybit_api_key,
                    secret_key=settings.bybit_secret_key,
                    testnet=settings.bybit_testnet,
                )
                connected = await bybit.connect()
                self._brokers["bybit"] = bybit
                results["bybit"] = connected
            else:
                logger.info("Bybit API keys not configured, skipping")
            
            return all(results.values())
            
        except Exception as e:
            logger.error(f"Broker initialization error: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all connections."""
        logger.info("Shutting down connections...")
        
        # Close Redis
        if self._cache:
            try:
                await self._cache.disconnect()
            except Exception as e:
                logger.error(f"Redis disconnect error: {e}")
        
        # Close sync database
        if self._database:
            try:
                self._database.disconnect()
            except Exception as e:
                logger.error(f"Sync database disconnect error: {e}")
        
        # Close async database
        if self._async_database:
            try:
                await self._async_database.disconnect()
            except Exception as e:
                logger.error(f"Async database disconnect error: {e}")
        
        # Close brokers
        for name, broker in self._brokers.items():
            try:
                await broker.disconnect()
            except Exception as e:
                logger.error(f"Broker {name} disconnect error: {e}")
        
        self._initialized = False
        logger.info("All connections closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all connections.
        
        Returns:
            Dictionary with health status of each component
        """
        results = {
            "initialized": self._initialized,
            "components": {},
        }
        
        # Redis health
        if self._cache:
            results["components"]["redis"] = await self._cache.health_check()
        else:
            results["components"]["redis"] = {"connected": False, "error": "Not initialized"}
        
        # Sync database health
        if self._database:
            results["components"]["database_sync"] = self._database.health_check()
        else:
            results["components"]["database_sync"] = {"connected": False, "error": "Not initialized"}
        
        # Async database health
        if self._async_database:
            results["components"]["database_async"] = await self._async_database.health_check()
        else:
            results["components"]["database_async"] = {"connected": False, "error": "Not initialized"}
        
        # Broker health
        results["components"]["brokers"] = {}
        for name, broker in self._brokers.items():
            try:
                is_healthy = await broker.health_check()
                results["components"]["brokers"][name] = {
                    "connected": broker.connected,
                    "healthy": is_healthy,
                }
            except Exception as e:
                results["components"]["brokers"][name] = {
                    "connected": False,
                    "healthy": False,
                    "error": str(e),
                }
        
        # Overall status
        all_healthy = all(
            comp.get("connected", False) or comp.get("healthy", False)
            for comp in results["components"].values()
            if isinstance(comp, dict)
        )
        results["healthy"] = all_healthy
        
        return results
    
    def get_broker(self, name: str = None):
        """
        Get broker connection by name.
        
        Args:
            name: Broker name (binance, bybit, paper). 
                  If None, returns paper trading broker.
        
        Returns:
            Broker connector instance
        """
        if name is None:
            # Default to paper trading
            return self._brokers.get("paper")
        
        return self._brokers.get(name)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_connection_manager: Optional[ConnectionManager] = None


async def get_connection_manager() -> ConnectionManager:
    """Get or create the connection manager singleton."""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    
    return _connection_manager


async def initialize_connections(**kwargs) -> ConnectionManager:
    """Initialize all connections and return the manager."""
    manager = await get_connection_manager()
    
    if not manager._initialized:
        await manager.initialize(**kwargs)
    
    return manager


async def shutdown_connections() -> None:
    """Shutdown all connections."""
    global _connection_manager
    
    if _connection_manager is not None:
        await _connection_manager.shutdown()
        _connection_manager = None


# =============================================================================
# FASTAPI LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan context manager for connection management.
    
    Usage:
        from fastapi import FastAPI
        from app.core.connections import lifespan
        
        app = FastAPI(lifespan=lifespan)
    """
    # Startup
    manager = await initialize_connections(
        connect_redis=True,
        connect_database=True,
        connect_brokers=True,  # Connect brokers to get real data
    )
    
    yield
    
    # Shutdown
    await shutdown_connections()
