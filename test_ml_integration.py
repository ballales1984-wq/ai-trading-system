
import sys
import os
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())

try:
    from ml_predictor import get_ml_predictor
    from data_collector import DataCollector
    import pandas as pd
    
    print("Testing ML Predictor Integration...")
    predictor = get_ml_predictor()
    collector = DataCollector(simulation=True)
    
    symbol = "BTCUSDT"
    df = collector.fetch_ohlcv(symbol, "1h", 100)
    
    if df is not None:
        print(f"Data fetched for {symbol}, length: {len(df)}")
        prediction = predictor.predict(df)
        print(f"Prediction result: {prediction}")
        if prediction:
            print("SUCCESS: ML Predictor is functional and connected.")
        else:
            print("FAILURE: Prediction returned None.")
    else:
        print("FAILURE: Could not fetch data.")

except Exception as e:
    print(f"ERROR during testing: {e}")
    import traceback
    traceback.print_exc()
