"""Tests for data_adapter module."""

import unittest
from unittest.mock import Mock, patch


class TestDataAdapter(unittest.TestCase):
    """Tests for DataAdapter class."""

    def test_data_adapter_creation(self):
        """Test DataAdapter can be created."""
        from app.core.data_adapter import DataAdapter
        adapter = DataAdapter()
        self.assertIsNotNone(adapter)

    def test_data_adapter_initial_state(self):
        """Test DataAdapter initial state."""
        from app.core.data_adapter import DataAdapter
        adapter = DataAdapter()
        # State manager and simulator may or may not be initialized
        self.assertTrue(hasattr(adapter, 'state_manager'))
        self.assertTrue(hasattr(adapter, 'simulator'))

    def test_data_adapter_get_portfolio_summary(self):
        """Test get_portfolio_summary method exists."""
        from app.core.data_adapter import DataAdapter
        adapter = DataAdapter()
        self.assertTrue(hasattr(adapter, 'get_portfolio_summary'))
        self.assertTrue(callable(adapter.get_portfolio_summary))


if __name__ == "__main__":
    unittest.main()
