# test_execution.py
"""
Test Execution - Institutional-Grade Execution Improvements
============================================================
Tests for TCA, Slippage Modeling, Order Book Simulator, and Best Execution
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from datetime import datetime, timedelta

from src.core.execution.tca import (
    TransactionCostAnalyzer,
    SlippageModel,
    TradeRecord,
    OrderSnapshot,
    MarketImpactModel,
    create_tca_analyzer,
)

from src.core.execution.orderbook_simulator import (
    OrderBookSimulator,
    create_order_book_from_depth,
)

from src.core.execution.best_execution import (
    BestExecutionEngine,
    ExecutionStrategy,
    ExecutionConfig,
    MarketDataSnapshot,
    create_execution_engine,
)


def test_tca_implementation_shortfall():
    """Test TCA implementation shortfall calculation."""
    print("\n=== Testing TCA Implementation Shortfall ===")
    
    # Create TCA analyzer
    tca = create_tca_analyzer(impact_model="sqrt", commission=0.0002)
    
    # Create trade records
    trades = [
        TradeRecord(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="buy",
            quantity=0.5,
            price=50000.0,
            commission=5.0,  # $5 commission
        ),
        TradeRecord(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="buy",
            quantity=0.3,
            price=50050.0,
            commission=3.0,
        ),
    ]
    
    # Create order snapshot
    snapshot = OrderSnapshot(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        side="buy",
        order_type="limit",
        quantity=1.0,
        filled_quantity=0.8,
        arrival_price=49950.0,
        best_bid=49940.0,
        best_ask=49960.0,
        mid_price=49950.0,
        vwap=49980.0,
        volume_traded=0.0,
        market_volume=1000.0,
    )
    
    # Calculate IS
    quality = tca.calculate_implementation_shortfall(trades, snapshot)
    
    print(f"Execution Price: ${quality.execution_price:.2f}")
    print(f"Arrival Price: ${quality.arrival_price:.2f}")
    print(f"Implementation Shortfall: {quality.total_implementation_shortfall:.2f} bps")
    print(f"Market Impact Cost: ${quality.market_impact_cost:.2f}")
    print(f"Spread Cost: ${quality.spread_cost:.2f}")
    
    # Generate TCA report
    report = tca.generate_tca_report(trades, snapshot)
    print(f"\nOverall Quality: {report['overall_quality']}")
    print(f"VWAP Slippage: {report['vwap_comparison'].get('vwap_slippage_bps', 0):.2f} bps")
    
    assert quality.execution_price > 0, "Execution price should be positive"
    assert quality.arrival_price > 0, "Arrival price should be positive"
    print("\n[PASS] TCA Implementation Shortfall test passed!")
    

def test_slippage_model():
    """Test slippage modeling."""
    print("\n=== Testing Slippage Model ===")
    
    # Create slippage model
    slippage_model = SlippageModel(
        base_slippage_bps=1.0,
        volatility_scalar=2.0,
        liquidity_scalar=0.5,
    )
    
    # Estimate slippage for a large order
    result = slippage_model.estimate_slippage(
        order_size=100000,  # $100k order
        market_volume=1000000,  # $1M daily volume
        volatility=0.03,  # 3% volatility
        order_type="market",
        urgency=0.5,
    )
    
    print(f"Estimated Slippage: {result['slippage_bps']:.2f} bps")
    print(f"Slippage Cost: ${result['slippage_cost']:.2f}")
    print(f"Volume Component: {result['volume_component']:.2f}")
    print(f"Volatility Component: {result['volatility_component']:.2f}")
    print(f"Participation Rate: {result['participation_rate']:.2%}")
    
    # Test market impact decay
    decay_result = slippage_model.estimate_market_impact_decay(
        time_horizon=300,  # 5 minutes
        initial_impact=10.0,  # 10 bps
        decay_half_life=300,  # 5 minute half-life
    )
    
    print(f"\nMarket Impact after 5 min: {decay_result:.2f} bps")
    
    # Test iceberg impact
    iceberg_result = slippage_model.calculate_iceberg_impact(
        visible_size=1000,
        hidden_size=9000,
        market_volume=100000,
    )
    
    print(f"Iceberg Total Impact: {iceberg_result['total_impact']:.2f}")
    print(f"Iceberg Hidden Impact: {iceberg_result['hidden_impact']:.2f}")
    print(f"Impact Saved by Iceberg: {iceberg_result['impact_saved']:.2f}")
    
    assert result['slippage_bps'] > 0, "Slippage should be positive"
    print("\n[PASS] Slippage Model test passed!")


def test_orderbook_simulator():
    """Test order book simulator."""
    print("\n=== Testing Order Book Simulator ===")
    
    # Create simulator
    simulator = OrderBookSimulator(num_levels=10)
    
    # Create order book from depth data
    bids = [
        (50000.0, 10.0),
        (49999.0, 20.0),
        (49998.0, 30.0),
        (49997.0, 40.0),
        (49996.0, 50.0),
    ]
    
    asks = [
        (50001.0, 15.0),
        (50002.0, 25.0),
        (50003.0, 35.0),
        (50004.0, 45.0),
        (50005.0, 55.0),
    ]
    
    book = create_order_book_from_depth("BTCUSDT", bids, asks)
    
    print(f"Best Bid: ${book.best_bid:.2f}")
    print(f"Best Ask: ${book.best_ask:.2f}")
    print(f"Mid Price: ${book.mid_price:.2f}")
    print(f"Spread: ${book.spread:.2f} ({book.spread_bps:.2f} bps)")
    print(f"Imbalance: {book.imbalance:.2%}")
    print(f"Total Bid Volume: {book.total_bid_volume:.2f}")
    print(f"Total Ask Volume: {book.total_ask_volume:.2f}")
    
    # Estimate market impact for buy order
    impact = simulator.estimate_market_impact(
        order_book=book,
        order_side="buy",
        order_quantity=20.0,  # Buy 20 BTC
        order_value=20.0 * 50000,
    )
    
    print(f"\n--- Buy Order Impact (20 BTC) ---")
    print(f"Immediate Impact: {impact.immediate_impact:.2f} bps")
    print(f"Permanent Impact: {impact.permanent_impact:.2f} bps")
    print(f"Temporary Impact: {impact.temporary_impact:.2f} bps")
    print(f"Queue Position: {impact.queue_position}")
    print(f"Fill Probability: {impact.fill_probability:.2%}")
    print(f"Adverse Selection Risk: {impact.adverse_selection_risk:.2%}")
    
    # Get queue position
    queue_info = simulator.get_queue_position(
        order_book=book,
        order_side="buy",
        order_price=50000.0,  # At best bid
        order_quantity=5.0,
    )
    
    print(f"\n--- Queue Position Info ---")
    print(f"Queue Position: {queue_info['queue_position']}")
    print(f"Queue Ahead Quantity: {queue_info['queue_ahead_quantity']:.2f}")
    print(f"Estimated Fill Time: {queue_info['estimated_fill_time']:.1f}s")
    
    assert book.mid_price > 0, "Mid price should be positive"
    assert impact.immediate_impact > 0, "Impact should be positive for buy"
    print("\n[PASS] Order Book Simulator test passed!")


def test_best_execution():
    """Test best execution algorithms."""
    print("\n=== Testing Best Execution Engine ===")
    
    # Create execution engine
    engine = create_execution_engine(strategy="vwap")
    
    # Create market data snapshot
    market_data = MarketDataSnapshot(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        last_price=50000.0,
        bid=49990.0,
        ask=50010.0,
        mid=50000.0,
        volume=100.0,
        vwap=49995.0,
        volatility=0.02,
        bid_ask_spread=20.0,
        market_volume_1m=500.0,
        market_volume_5m=2500.0,
    )
    
    # Test strategy selection
    strategy, analysis = engine.select_best_strategy(
        symbol="BTCUSDT",
        quantity=10.0,
        market_data=market_data,
    )
    
    print(f"Selected Strategy: {analysis['strategy']}")
    print(f"Reason: {analysis['reason']}")
    print(f"Conditions: {analysis['conditions']}")
    
    # Test VWAP execution
    plan = engine.create_execution_order(
        strategy=ExecutionStrategy.VWAP,
        symbol="BTCUSDT",
        side="buy",
        quantity=10.0,
        market_data=market_data,
    )
    
    print(f"\n--- VWAP Execution Plan ---")
    print(f"Strategy: {plan.strategy.value}")
    print(f"Total Quantity: {plan.total_quantity:.4f}")
    print(f"Number of Slices: {len(plan.slices)}")
    print(f"Start Time: {plan.start_time}")
    print(f"End Time: {plan.end_time}")
    
    for i, slice_ in enumerate(plan.slices[:3]):  # Show first 3
        print(f"  Slice {i+1}: {slice_.quantity:.4f} @ {slice_.timestamp.strftime('%H:%M:%S')}")
    
    # Test TWAP execution
    twap_plan = engine.create_execution_order(
        strategy=ExecutionStrategy.TWAP,
        symbol="ETHUSDT",
        side="sell",
        quantity=100.0,
        market_data=market_data,
    )
    
    print(f"\n--- TWAP Execution Plan ---")
    print(f"Strategy: {twap_plan.strategy.value}")
    print(f"Number of Slices: {len(twap_plan.slices)}")
    
    # Test adaptive execution
    adaptive_plan = engine.create_execution_order(
        strategy=ExecutionStrategy.ADAPTIVE,
        symbol="BTCUSDT",
        side="buy",
        quantity=5.0,
        market_data=market_data,
    )
    
    print(f"\n--- Adaptive Execution Plan ---")
    print(f"Strategy: {adaptive_plan.strategy.value}")
    print(f"Duration: {(adaptive_plan.end_time - adaptive_plan.start_time).total_seconds():.0f}s")
    
    # Test cost calculation
    costs = engine.calculate_expected_cost(plan, market_data)
    print(f"\n--- Expected Costs ---")
    print(f"Spread Cost: ${costs['spread_cost']:.2f}")
    print(f"Market Impact: ${costs['market_impact']:.2f}")
    print(f"Total Cost: ${costs['total_cost']:.2f}")
    print(f"Total Cost (bps): {costs['total_cost_bps']:.2f} bps")
    
    assert len(plan.slices) > 0, "Plan should have slices"
    assert plan.total_quantity > 0, "Total quantity should be positive"
    print("\n[PASS] Best Execution test passed!")


def test_tca_report():
    """Test comprehensive TCA report generation."""
    print("\n=== Testing TCA Report ===")
    
    tca = create_tca_analyzer(impact_model="sqrt")
    
    # Simulate a full day of trading
    trades = [
        TradeRecord(
            timestamp=datetime.now() - timedelta(minutes=i*10),
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            price=50000 + np.random.randn() * 50,
            commission=1.0,
        )
        for i in range(10)
    ]
    
    snapshot = OrderSnapshot(
        timestamp=datetime.now() - timedelta(minutes=100),
        symbol="BTCUSDT",
        side="buy",
        order_type="limit",
        quantity=1.0,
        filled_quantity=1.0,
        arrival_price=49950.0,
        best_bid=49940.0,
        best_ask=49960.0,
        mid_price=49950.0,
        vwap=49980.0,
        volume_traded=1000.0,
        market_volume=10000.0,
    )
    
    report = tca.generate_tca_report(trades, snapshot)
    
    print(f"Symbol: {report['symbol']}")
    print(f"Side: {report['side']}")
    print(f"Quantity: {report['quantity']}")
    print(f"Filled: {report['filled_quantity']}")
    print(f"Implementation Shortfall: {report['implementation_shortfall']['total_bps']:.2f} bps")
    print(f"VWAP Comparison: {report['vwap_comparison'].get('execution_vwap', 0):.2f}")
    print(f"Arrival Price Comparison: {report['arrival_comparison'].get('price_improvement_bps', 0):.2f} bps")
    print(f"Overall Quality: {report['overall_quality']}")
    
    assert 'implementation_shortfall' in report, "Report should have IS"
    assert 'vwap_comparison' in report, "Report should have VWAP"
    print("\n[PASS] TCA Report test passed!")


def test_slippage_model_calibration():
    """Test slippage model calibration."""
    print("\n=== Testing Slippage Model Calibration ===")
    
    tca = create_tca_analyzer()
    
    # Generate calibration data
    historical_data = []
    for _ in range(20):
        participation = np.random.uniform(0.01, 0.1)
        volatility = np.random.uniform(0.01, 0.05)
        
        # Simulated impact based on model
        impact = 0.0001 * np.sqrt(participation) * (volatility / 0.02)
        volume = 1000000
        
        historical_data.append((participation, volatility, impact, volume))
    
    # Calibrate
    tca.calibrate_impact_model(historical_data)
    
    print(f"Calibrated coefficients: {tca.impact_coefficients}")
    print("\n[PASS] Slippage Model Calibration test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Institutional-Grade Execution Improvements Test Suite")
    print("=" * 60)
    
    try:
        test_tca_implementation_shortfall()
        test_slippage_model()
        test_orderbook_simulator()
        test_best_execution()
        test_tca_report()
        test_slippage_model_calibration()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
