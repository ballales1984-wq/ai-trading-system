# TODO: Add Missing Database Tables

## Task: Enhance SQLite Database with Additional Tables

### Current Tables (Already Exist):
- portfolio ✅
- positions ✅
- orders ✅
- models ✅
- trades ✅
- event_log ✅

### New Tables Added:
1. [x] **signals** - Store ML signals history
   - symbol, signal_type, confidence, timestamp, source
    
2. [x] **price_history** - Store OHLCV data for backtesting
   - symbol, timestamp, open, high, low, close, volume
    
3. [x] **model_performance** - Track ML model accuracy over time
   - model_id, accuracy, precision, recall, f1, timestamp
    
4. [x] **backtest_results** - Store backtest results
   - strategy, initial_balance, final_balance, total_return, trades, win_rate, timestamp

### Files Modified:
- src/core/state_manager.py - Added new table schemas and methods ✅
- test_database_tables.py - Added tests for new tables ✅

### Implementation Complete:
1. [x] Add new table schemas in `_init_database()` method
2. [x] Add new data classes for each table
3. [x] Add CRUD methods for each new table
4. [x] Add indexes for better query performance
5. [x] Add tests for all new tables

### Tests: 9/9 PASSED ✅
