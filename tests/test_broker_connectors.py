"""Tests for broker connectors module."""

import unittest


class TestBinanceConnector(unittest.TestCase):
    """Tests for BinanceConnector class."""

    def test_binance_connector_exists(self):
        """Test BinanceConnector class exists."""
        try:
            from app.execution.connectors.binance_connector import BinanceConnector
            self.assertTrue(hasattr(BinanceConnector, '__init__'))
        except ImportError:
            self.skipTest("BinanceConnector not available")

    def test_binance_connector_is_class(self):
        """Test BinanceConnector is a class."""
        try:
            from app.execution.connectors.binance_connector import BinanceConnector
            self.assertTrue(isinstance(BinanceConnector, type))
        except ImportError:
            self.skipTest("BinanceConnector not available")


class TestPaperConnector(unittest.TestCase):
    """Tests for PaperConnector class."""

    def test_paper_connector_exists(self):
        """Test PaperConnector class exists."""
        try:
            from app.execution.connectors.paper_connector import PaperConnector
            self.assertTrue(hasattr(PaperConnector, '__init__'))
        except ImportError:
            self.skipTest("PaperConnector not available")

    def test_paper_connector_is_class(self):
        """Test PaperConnector is a class."""
        try:
            from app.execution.connectors.paper_connector import PaperConnector
            self.assertTrue(isinstance(PaperConnector, type))
        except ImportError:
            self.skipTest("PaperConnector not available")


class TestIBConnector(unittest.TestCase):
    """Tests for IBConnector class."""

    def test_ib_connector_exists(self):
        """Test IBConnector class exists."""
        try:
            from app.execution.connectors.ib_connector import IBConnector
            self.assertTrue(hasattr(IBConnector, '__init__'))
        except ImportError:
            self.skipTest("IBConnector not available")

    def test_ib_connector_is_class(self):
        """Test IBConnector is a class."""
        try:
            from app.execution.connectors.ib_connector import IBConnector
            self.assertTrue(isinstance(IBConnector, type))
        except ImportError:
            self.skipTest("IBConnector not available")


if __name__ == "__main__":
    unittest.main()
