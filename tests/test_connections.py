"""Tests for connections module."""

import unittest
from unittest.mock import Mock, patch, AsyncMock


class TestConnectionManager(unittest.TestCase):
    """Tests for ConnectionManager class."""

    def test_connection_manager_creation(self):
        """Test ConnectionManager can be created."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        self.assertIsNotNone(manager)

    def test_connection_manager_initial_state(self):
        """Test ConnectionManager initial state."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        self.assertFalse(manager._initialized)
        self.assertIsNone(manager._cache)
        self.assertIsNone(manager._database)
        self.assertEqual(len(manager._brokers), 0)

    def test_connection_manager_properties(self):
        """Test ConnectionManager properties."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        
        # Test cache property
        self.assertIsNone(manager.cache)
        
        # Test database property
        self.assertIsNone(manager.database)
        
        # Test async_database property
        self.assertIsNone(manager.async_database)
        
        # Test brokers property
        self.assertEqual(manager.brokers, {})

    @patch('app.core.connections.settings')
    def test_connection_manager_initialize(self, mock_settings):
        """Test ConnectionManager initialize method."""
        from app.core.connections import ConnectionManager
        mock_settings.REDIS_URL = "redis://localhost:6379"
        mock_settings.DATABASE_URL = "postgresql://localhost:5432"
        
        manager = ConnectionManager()
        # Test async initialize exists
        self.assertTrue(hasattr(manager, 'initialize'))
        self.assertTrue(asyncio.iscoroutinefunction(manager.initialize))

    def test_connection_manager_shutdown(self):
        """Test ConnectionManager shutdown method."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        # Test async shutdown exists
        self.assertTrue(hasattr(manager, 'shutdown'))
        self.assertTrue(asyncio.iscoroutinefunction(manager.shutdown))

    def test_connection_manager_health_check(self):
        """Test ConnectionManager health_check method."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        # Test async health_check exists
        self.assertTrue(hasattr(manager, 'health_check'))
        self.assertTrue(asyncio.iscoroutinefunction(manager.health_check))

    def test_set_cache(self):
        """Test setting cache connection."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        mock_cache = Mock()
        manager._cache = mock_cache
        self.assertEqual(manager.cache, mock_cache)

    def test_set_database(self):
        """Test setting database connection."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        mock_db = Mock()
        manager._database = mock_db
        self.assertEqual(manager.database, mock_db)

    def test_set_broker(self):
        """Test setting broker connection."""
        from app.core.connections import ConnectionManager
        manager = ConnectionManager()
        mock_broker = Mock()
        manager._brokers['binance'] = mock_broker
        self.assertIn('binance', manager.brokers)
        self.assertEqual(manager.brokers['binance'], mock_broker)


# Import asyncio for the test
import asyncio


if __name__ == "__main__":
    unittest.main()
