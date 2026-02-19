# test_database_tables.py
"""
Test script for new database tables in StateManager.
"""

import pytest
import tempfile
import os
from datetime import datetime


class TestDatabaseTables:
    """Test new database tables."""
    
    @pytest.fixture
    def state_manager(self):
        """Create a StateManager with temporary database."""
        from src.core.state_manager import StateManager
        
        # Create temp database
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        sm = StateManager(db_path=db_path)
        yield sm
        
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass
    
    def test_signals_table_exists(self, state_manager):
        """Test that signals table exists."""
        with state_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
            result = cursor.fetchone()
            assert result is not None
            print("[OK] signals table exists")
    
    def test_price_history_table_exists(self, state_manager):
        """Test that price_history table exists."""
        with state_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'")
            result = cursor.fetchone()
            assert result is not None
            print("[OK] price_history table exists")
    
    def test_model_performance_table_exists(self, state_manager):
        """Test that model_performance table exists."""
        with state_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_performance'")
            result = cursor.fetchone()
            assert result is not None
            print("[OK] model_performance table exists")
    
    def test_backtest_results_table_exists(self, state_manager):
        """Test that backtest_results table exists."""
        with state_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_results'")
            result = cursor.fetchone()
            assert result is not None
            print("[OK] backtest_results table exists")
    
    def test_save_and_get_signal(self, state_manager):
        """Test saving and retrieving signals."""
        from src.core.state_manager import SignalState
        
        signal = SignalState(
            symbol="BTCUSDT",
            signal_type="BUY",
            confidence=0.85,
            source="xgboost"
        )
        
        state_manager.save_signal(signal)
        
        signals = state_manager.get_signals(symbol="BTCUSDT", limit=10)
        assert len(signals) == 1
        assert signals[0].symbol == "BTCUSDT"
        assert signals[0].signal_type == "BUY"
        assert signals[0].confidence == 0.85
        print("[OK] Signal save and retrieve works")
    
    def test_save_and_get_price_history(self, state_manager):
        """Test saving and retrieving price history."""
        from src.core.state_manager import PriceHistoryState
        
        price = PriceHistoryState(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=95000.0,
            high=95500.0,
            low=94500.0,
            close=95200.0,
            volume=1000.0
        )
        
        state_manager.save_price_history(price)
        
        prices = state_manager.get_price_history("BTCUSDT", limit=10)
        assert len(prices) == 1
        assert prices[0].symbol == "BTCUSDT"
        assert prices[0].open == 95000.0
        print("[OK] Price history save and retrieve works")
    
    def test_save_and_get_model_performance(self, state_manager):
        """Test saving and retrieving model performance."""
        from src.core.state_manager import ModelPerformanceState
        
        perf = ModelPerformanceState(
            model_id="xgboost_v1",
            accuracy=0.75,
            precision=0.72,
            recall=0.78,
            f1=0.75,
            confusion_matrix="[[50,10],[15,25]]"
        )
        
        state_manager.save_model_performance(perf)
        
        perfs = state_manager.get_model_performance("xgboost_v1", limit=10)
        assert len(perfs) == 1
        assert perfs[0].model_id == "xgboost_v1"
        assert perfs[0].accuracy == 0.75
        print("[OK] Model performance save and retrieve works")
    
    def test_save_and_get_backtest_result(self, state_manager):
        """Test saving and retrieving backtest results."""
        from src.core.state_manager import BacktestResultState
        
        result = BacktestResultState(
            strategy="momentum_v1",
            initial_balance=100000.0,
            final_balance=125000.0,
            total_return=0.25,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.60,
            max_drawdown=0.10,
            sharpe_ratio=1.5
        )
        
        state_manager.save_backtest_result(result)
        
        results = state_manager.get_backtest_results(strategy="momentum_v1", limit=10)
        assert len(results) == 1
        assert results[0].strategy == "momentum_v1"
        assert results[0].total_return == 0.25
        print("[OK] Backtest result save and retrieve works")
    
    def test_indexes_exist(self, state_manager):
        """Test that indexes were created."""
        with state_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check for expected indexes
            expected = ['idx_signals_symbol', 'idx_price_history_symbol', 
                       'idx_model_performance_model', 'idx_backtest_strategy']
            
            for idx in expected:
                assert idx in indexes, f"Index {idx} not found"
            
            print(f"[OK] All indexes created: {len(expected)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
