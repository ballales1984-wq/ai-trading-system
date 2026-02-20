"""
PostgreSQL Database Configuration for Production
================================================
Configuration and utilities for PostgreSQL database connection.

Environment Variables:
    DATABASE_URL: Full PostgreSQL connection URL
    DB_HOST: Database host (default: localhost)
    DB_PORT: Database port (default: 5432)
    DB_NAME: Database name (default: ai_trading)
    DB_USER: Database user
    DB_PASSWORD: Database password

Usage:
    from src.database_config import get_production_database
    
    db = get_production_database()
"""

import os
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    Get database URL from environment variables.
    
    Priority:
    1. DATABASE_URL (full URL)
    2. Individual DB_* variables
    3. SQLite fallback for development
    
    Returns:
        Database connection URL
    """
    # Check for full URL first
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        return database_url
    
    # Check for individual variables
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_port = os.environ.get('DB_PORT', '5432')
    db_name = os.environ.get('DB_NAME', 'ai_trading')
    db_user = os.environ.get('DB_USER')
    db_password = os.environ.get('DB_PASSWORD')
    
    if db_user and db_password:
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Fallback to SQLite for development
    logger.warning("No PostgreSQL configuration found, using SQLite")
    return "sqlite:///data/ml_trading.db"


def is_postgresql() -> bool:
    """Check if using PostgreSQL database."""
    url = get_database_url()
    return url.startswith('postgresql')


def get_database_config() -> dict:
    """
    Get database configuration dictionary.
    
    Returns:
        Dictionary with database configuration
    """
    url = get_database_url()
    
    if url.startswith('postgresql'):
        parsed = urlparse(url)
        return {
            'driver': 'postgresql',
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 5432,
            'database': parsed.path.lstrip('/'),
            'user': parsed.username,
            'password': parsed.password,
            'pool_size': int(os.environ.get('DB_POOL_SIZE', '5')),
            'max_overflow': int(os.environ.get('DB_MAX_OVERFLOW', '10')),
        }
    else:
        return {
            'driver': 'sqlite',
            'path': url.replace('sqlite:///', ''),
        }


def get_production_database():
    """
    Get production database instance.
    
    Returns:
        SQLAlchemyDatabase instance configured for production
    """
    from src.database_sqlalchemy import SQLAlchemyDatabase
    
    db_url = get_database_url()
    
    if db_url.startswith('postgresql'):
        logger.info("Connecting to PostgreSQL database")
        return SQLAlchemyDatabase(db_url=db_url)
    else:
        logger.info("Using SQLite database (development mode)")
        return SQLAlchemyDatabase(db_path="data/ml_trading.db")


# Database health check
def check_database_connection() -> dict:
    """
    Check database connection health.
    
    Returns:
        Dictionary with connection status
    """
    try:
        from src.database_sqlalchemy import SQLAlchemyDatabase
        
        db = get_production_database()
        
        with db.get_session() as session:
            session.execute('SELECT 1')
        
        return {
            'status': 'healthy',
            'driver': get_database_config()['driver'],
            'url': get_database_url().split('@')[-1] if '@' in get_database_url() else get_database_url()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


# Connection pooling configuration for PostgreSQL
POSTGRESQL_POOL_CONFIG = {
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
    'echo': False,
}


def create_postgresql_engine(db_url: str):
    """
    Create PostgreSQL engine with optimized settings.
    
    Args:
        db_url: PostgreSQL connection URL
        
    Returns:
        SQLAlchemy Engine
    """
    from sqlalchemy import create_engine
    
    return create_engine(db_url, **POSTGRESQL_POOL_CONFIG)
