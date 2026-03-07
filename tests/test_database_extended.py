"""Extended tests for database module"""
import pytest

class TestDatabaseExtended:
    def test_database_module_exists(self):
        from app.core import database
        assert database is not None
    
    def test_database_models_import(self):
        from app.database import models
        assert models is not None
    
    def test_repository_import(self):
        from app.database import repository
        assert repository is not None
    
    def test_async_repository_import(self):
        from app.database import async_repository
        assert async_repository is not None
