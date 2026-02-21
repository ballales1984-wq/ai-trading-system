"""
Database Connection Manager
==========================
Centralized PostgreSQL/TimescaleDB connection management.
Provides async and sync database connections with connection pooling.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, AsyncGenerator, Generator, Any, Dict

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.pool import QueuePool, NullPool

from app.core.config import settings


logger = logging.getLogger(__name__)


# =============================================================================
# SYNC DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """
    Synchronous database connection manager.
    
    Features:
    - Connection pooling
    - Automatic reconnection
    - Health monitoring
    - TimescaleDB support
    """
    
    def __init__(
        self,
        database_url: str = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        self.database_url = database_url or settings.database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        
        self._engine = None
        self._session_factory = None
        self._connected = False
        
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            # Convert postgresql:// to postgresql+psycopg2:// for sync
            db_url = self.database_url
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
            
            self._engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,  # Enable connection health checks
                echo=self.echo,
            )
            
            # Add event listeners
            self._setup_event_listeners()
            
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )
            
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self._connected = True
            logger.info(f"Connected to database: {self._get_safe_url()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connected = False
            return False
    
    def _get_safe_url(self) -> str:
        """Get database URL with password masked."""
        url = self.database_url
        if "@" in url:
            parts = url.split("@")
            if ":" in parts[0]:
                cred_parts = parts[0].rsplit(":", 1)
                return f"{cred_parts[0]}:****@{parts[1]}"
        return url
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners."""
        
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            logger.debug("Database connection created")
        
        @event.listens_for(self._engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Database connection checked out")
        
        @event.listens_for(self._engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            logger.debug("Database connection checked in")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
        
        self._engine = None
        self._session_factory = None
        self._connected = False
        logger.info("Disconnected from database")
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self._engine is not None
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if not self._session_factory:
            raise RuntimeError("Database not connected")
        return self._session_factory()
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection."""
        result = {
            "connected": False,
            "pool_size": None,
            "pool_checked_out": None,
            "pool_overflow": None,
            "database_version": None,
            "timescale_enabled": False,
            "error": None,
        }
        
        if not self._engine:
            result["error"] = "Database engine not initialized"
            return result
        
        try:
            with self._engine.connect() as conn:
                # Test query
                conn.execute(text("SELECT 1"))
                
                # Get PostgreSQL version
                version_result = conn.execute(text("SELECT version()"))
                version = version_result.scalar()
                
                # Check TimescaleDB
                timescale_result = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                )
                timescale_version = timescale_result.scalar()
                
                # Pool stats
                pool = self._engine.pool
                
                result.update({
                    "connected": True,
                    "pool_size": pool.size(),
                    "pool_checked_out": pool.checkedout(),
                    "pool_overflow": pool.overflow(),
                    "database_version": version,
                    "timescale_enabled": timescale_version is not None,
                    "timescale_version": timescale_version,
                })
                
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def init_timescaledb(self) -> bool:
        """Initialize TimescaleDB extension."""
        if not self._engine:
            return False
        
        try:
            with self._engine.connect() as conn:
                # Create TimescaleDB extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                conn.commit()
                
            logger.info("TimescaleDB extension initialized")
            return True
            
        except Exception as e:
            logger.warning(f"TimescaleDB initialization skipped: {e}")
            return False
    
    def create_tables(self, base) -> bool:
        """Create all tables from SQLAlchemy models."""
        if not self._engine:
            return False
        
        try:
            base.metadata.create_all(self._engine)
            logger.info("Database tables created")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False


# =============================================================================
# ASYNC DATABASE MANAGER
# =============================================================================

class AsyncDatabaseManager:
    """
    Asynchronous database connection manager.
    
    Features:
    - Async/await support
    - Connection pooling
    - TimescaleDB support
    """
    
    def __init__(
        self,
        database_url: str = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        echo: bool = False,
    ):
        self.database_url = database_url or settings.database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo
        
        self._engine = None
        self._session_factory = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Establish async database connection."""
        try:
            # Convert postgresql:// to postgresql+asyncpg:// for async
            db_url = self.database_url
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
            
            self._engine = create_async_engine(
                db_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                echo=self.echo,
            )
            
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
            )
            
            # Test connection
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            
            self._connected = True
            logger.info(f"Connected to async database: {self._get_safe_url()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to async database: {e}")
            self._connected = False
            return False
    
    def _get_safe_url(self) -> str:
        """Get database URL with password masked."""
        url = self.database_url
        if "@" in url:
            parts = url.split("@")
            if ":" in parts[0]:
                cred_parts = parts[0].rsplit(":", 1)
                return f"{cred_parts[0]}:****@{parts[1]}"
        return url
    
    async def disconnect(self) -> None:
        """Close async database connection."""
        if self._engine:
            await self._engine.dispose()
        
        self._engine = None
        self._session_factory = None
        self._connected = False
        logger.info("Disconnected from async database")
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self._engine is not None
    
    def get_session(self) -> AsyncSession:
        """Get a new async database session."""
        if not self._session_factory:
            raise RuntimeError("Async database not connected")
        return self._session_factory()
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async context manager for database sessions."""
        session = self.get_session()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async session error: {e}")
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on async database connection."""
        result = {
            "connected": False,
            "pool_size": None,
            "database_version": None,
            "timescale_enabled": False,
            "error": None,
        }
        
        if not self._engine:
            result["error"] = "Async database engine not initialized"
            return result
        
        try:
            async with self._engine.connect() as conn:
                # Test query
                await conn.execute(text("SELECT 1"))
                
                # Get PostgreSQL version
                version_result = await conn.execute(text("SELECT version()"))
                version = version_result.scalar()
                
                # Check TimescaleDB
                timescale_result = await conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                )
                timescale_version = timescale_result.scalar()
                
                result.update({
                    "connected": True,
                    "pool_size": self._engine.pool.size(),
                    "database_version": version,
                    "timescale_enabled": timescale_version is not None,
                    "timescale_version": timescale_version,
                })
                
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def init_timescaledb(self) -> bool:
        """Initialize TimescaleDB extension."""
        if not self._engine:
            return False
        
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                await conn.commit()
                
            logger.info("TimescaleDB extension initialized (async)")
            return True
            
        except Exception as e:
            logger.warning(f"TimescaleDB async initialization skipped: {e}")
            return False


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_db_manager: Optional[DatabaseManager] = None
_async_db_manager: Optional[AsyncDatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the sync database manager singleton."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.connect()
    
    return _db_manager


def get_async_db_manager() -> AsyncDatabaseManager:
    """Get or create the async database manager singleton."""
    global _async_db_manager
    
    if _async_db_manager is None:
        _async_db_manager = AsyncDatabaseManager()
    
    return _async_db_manager


async def init_async_db() -> AsyncDatabaseManager:
    """Initialize async database connection."""
    global _async_db_manager
    
    if _async_db_manager is None:
        _async_db_manager = AsyncDatabaseManager()
    
    if not _async_db_manager.is_connected:
        await _async_db_manager.connect()
    
    return _async_db_manager


def close_db_manager() -> None:
    """Close the sync database manager singleton."""
    global _db_manager
    
    if _db_manager is not None:
        _db_manager.disconnect()
        _db_manager = None


async def close_async_db_manager() -> None:
    """Close the async database manager singleton."""
    global _async_db_manager
    
    if _async_db_manager is not None:
        await _async_db_manager.disconnect()
        _async_db_manager = None


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_session_dependency() -> Generator[Session, None, None]:
    """FastAPI dependency for sync database sessions."""
    db = get_db_manager()
    session = db.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def get_async_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async database sessions."""
    db = await init_async_db()
    async with db.session() as session:
        yield session
