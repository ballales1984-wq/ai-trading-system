"""
Test file for Mont Blanck Strategy
"""
import numpy as np
from src.strategy.montblanck import MontBlanckStrategy, generate_montblanck_signal
from src.strategy.base_strategy import StrategyContext


def test_uptrend_buy_signal():
    """Test that uptrend generates BUY signal"""
    prices = np.array([100.0, 102.0, 105.0, 108.0, 110.0, 112.0, 115.0])
    strategy = MontBlanckStrategy(config={'params': {'window_size': 4, 'future_steps': 3, 'poly_degree': 3, 'buy_threshold': 0.01}})
    context = StrategyContext(symbol='BTC/USDT', prices=prices, volumes=np.array([]), timestamps=[])
    signal = strategy.generate_signal(context)
    assert signal is not None, "Signal should not be None"
    assert signal.signal_type.value == "BUY", f"Expected BUY, got {signal.signal_type.value}"
    print("Test 1 PASSED: Uptrend generates BUY signal")


def test_downtrend_sell_signal():
    """Test that downtrend with existing BUY generates SELL signal"""
    prices = np.array([100.0, 102.0, 105.0, 108.0, 110.0, 112.0, 111.0])
    strategy = MontBlanckStrategy(config={'params': {'window_size': 4, 'future_steps': 3, 'poly_degree': 3, 'buy_threshold': 0.01}})
    context = StrategyContext(symbol='BTC/USDT', prices=prices, volumes=np.array([]), timestamps=[], extra={'last_trade': {'type': 'BUY', 'price': 110}})
    signal = strategy.generate_signal(context)
    assert signal is not None, "Signal should not be None"
    assert signal.signal_type.value == "SELL", f"Expected SELL, got {signal.signal_type.value}"
    print("Test 2 PASSED: Downtrend with existing BUY generates SELL signal")


def test_standalone_function():
    """Test standalone function"""
    result = generate_montblanck_signal([100, 102, 105, 108, 110, 112, 115], None, 4, 3, 3, 0.01)
    assert result == "BUY", f"Expected BUY, got {result}"
    print("Test 3 PASSED: Standalone function works correctly")


def test_peak_prediction():
    """Test peak prediction"""
    prices = np.array([100.0, 102.0, 105.0, 108.0, 110.0, 112.0, 115.0])
    strategy = MontBlanckStrategy(config={'params': {'window_size': 4, 'future_steps': 3, 'poly_degree': 3}})
    predicted_peak = strategy.predict_peak(prices)
    assert predicted_peak is not None, "Predicted peak should not be None"
    assert predicted_peak > 115.0, f"Predicted peak should be above current price, got {predicted_peak}"
    print("Test 4 PASSED: Peak prediction works correctly")


if __name__ == "__main__":
    print("=== Mont Blanck Strategy Test Suite ===\n")
    test_uptrend_buy_signal()
    test_downtrend_sell_signal()
    test_standalone_function()
    test_peak_prediction()
    print("\n=== All tests passed! ===")

