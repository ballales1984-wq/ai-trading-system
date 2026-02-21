"""
Core Module Init
==============
Central configuration, database, and caching services.
"""

from app.core.config import settings, get_settings
from app.core.database import (
    DatabaseManager,
    AsyncDatabaseManager,
    get_db_manager,
    get_async_db_manager,
    init_async_db,
    close_db_manager,
    close_async_db_manager,
    get_session_dependency,
    get_async_session_dependency,
)
from app.core.cache import (
    RedisCacheManager,
    get_cache_manager,
    close_cache_manager,
    cached,
)
from app.core.connections import (
    ConnectionManager,
    get_connection_manager,
    initialize_connections,
    shutdown_connections,
    lifespan,
)

__all__ = [
    # Configuration
    "settings",
    "get_settings",
    # Database
    "DatabaseManager",
    "AsyncDatabaseManager",
    "get_db_manager",
    "get_async_db_manager",
    "init_async_db",
    "close_db_manager",
    "close_async_db_manager",
    "get_session_dependency",
    "get_async_session_dependency",
    # Cache
    "RedisCacheManager",
    "get_cache_manager",
    "close_cache_manager",
    "cached",
    # Connections
    "ConnectionManager",
    "get_connection_manager",
    "initialize_connections",
    "shutdown_connections",
    "lifespan",
]

