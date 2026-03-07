"""Extended tests for core connections module"""
import pytest

class TestConnectionsExtended:
    def test_connections_module_exists(self):
        from app.core import connections
        assert connections is not None
    
    def test_connection_manager_exists(self):
        from app.core.connections import ConnectionManager
        assert ConnectionManager is not None
    
    def test_connection_manager_can_be_created(self):
        from app.core.connections import ConnectionManager
        cm = ConnectionManager()
        assert cm is not None
