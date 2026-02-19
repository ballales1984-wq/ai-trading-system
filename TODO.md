# TODO: Add Missing Database Tables

## Task: Enhance SQLite Database with Additional Tables

### Current Tables (Already Exist):
- portfolio ✅
- positions ✅
- orders ✅
- models ✅
- trades ✅
- event_log ✅

### New Tables to Add:
1. [ ] **signals** - Store ML signals history
   - symbol, signal_type, confidence, timestamp, source
   
2. [ ] **price_history** - Store OHLCV data for backtesting
   - symbol, timestamp, open, high, low, close, volume
   
3. [ ] **model_performance** - Track ML model accuracy over time
   - model_id, accuracy, precision, recall, f1, timestamp
   
4. [ ] **backtest_results** - Store backtest results
   - strategy, initial_balance, final_balance, total_return, trades, win_rate, timestamp

### Files to Modify:
- src/core/state_manager.py - Add new table schemas and methods

### Implementation Steps:
1. Add new table schemas in `_init_database()` method
2. Add new data classes for each table
3. Add CRUD methods for each new table
