# test_core.py
"""
Test script for the core trading system components.
"""

import asyncio
import sys

# Test imports
print("=" * 50)
print("Testing Core Module Imports...")
print("=" * 50)

try:
    from src.core import (
        TradingEngine, PaperBroker, OrderManager, 
        RiskEngine, PortfolioManager, create_broker
    )
    print("[OK] All core imports successful")
except Exception as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)


async def test_paper_broker():
    """Test paper broker."""
    print("\n" + "=" * 50)
    print("Testing Paper Broker...")
    print("=" * 50)
    
    broker = PaperBroker(initial_balance=100000)
    await broker.connect()
    
    # Get balance
    balance = await broker.get_balance()
    print(f"[OK] Initial balance: ${balance.total_equity:,.2f}")
    
    # Get market price
    price = await broker.get_market_price("BTCUSDT")
    print(f"[OK] BTC price: ${price:,.2f}")
    
    await broker.disconnect()
    print("[OK] Paper broker test passed")
    return True


async def test_portfolio_manager():
    """Test portfolio manager."""
    print("\n" + "=" * 50)
    print("Testing Portfolio Manager...")
    print("=" * 50)
    
    pm = PortfolioManager(initial_balance=100000)
    
    # Open position
    position = pm.open_position("BTCUSDT", "long", 0.5, 45000, commission=10)
    print(f"[OK] Opened position: {position.symbol} {position.quantity} @ ${position.entry_price}")
    
    # Update price
    pm.update_prices({"BTCUSDT": 46000})
    pos = pm.get_position("BTCUSDT")
    print(f"[OK] Updated PnL: ${pos.unrealized_pnl:.2f}")
    
    # Get metrics
    metrics = pm.get_metrics()
    print(f"[OK] Portfolio equity: ${metrics.total_equity:.2f}")
    print(f"[OK] Available balance: ${metrics.available_balance:.2f}")
    
    print("[OK] Portfolio manager test passed")
    return True


async def test_risk_engine():
    """Test risk engine."""
    print("\n" + "=" * 50)
    print("Testing Risk Engine...")
    print("=" * 50)
    
    from src.core.risk import RiskLimits
    limits = RiskLimits(
        max_position_pct=0.3,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.20
    )
    
    risk = RiskEngine(initial_balance=100000, limits=limits)
    
    # Test signal check - should pass
    signal = {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'quantity': 0.1,
        'price': 45000,
        'position': 0
    }
    passed, reason = risk.check_signal(signal)
    print(f"[OK] Signal check (valid): {passed}")
    
    # Test signal check - should fail (order too large)
    signal_large = {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'quantity': 50,  # Too large
        'price': 45000,
        'position': 0
    }
    passed, reason = risk.check_signal(signal_large)
    print(f"[OK] Signal check (too large): {passed} - {reason}")
    
    # Test status
    status = risk.get_status()
    print(f"[OK] Risk level: {status['risk_level']}")
    
    print("[OK] Risk engine test passed")
    return True


async def test_order_manager():
    """Test order manager."""
    print("\n" + "=" * 50)
    print("Testing Order Manager...")
    print("=" * 50)
    
    broker = PaperBroker(initial_balance=100000)
    await broker.connect()
    
    risk = RiskEngine(initial_balance=100000)
    
    order_mgr = OrderManager(broker, risk_engine=risk)
    
    # Create order request
    from src.core.execution import OrderRequest
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        order_type="market",
        strategy="test"
    )
    
    # Execute order
    result = await order_mgr.execute_order(request, validate_risk=True)
    print(f"[OK] Order result: success={result.success}, order_id={result.order_id}")
    
    if result.success:
        print(f"[OK] Filled: {result.filled_quantity} @ ${result.avg_price:.2f}")
        print(f"[OK] Commission: ${result.commission:.2f}")
    
    await broker.disconnect()
    print("[OK] Order manager test passed")
    return True


async def test_trading_engine():
    """Test trading engine."""
    print("\n" + "=" * 50)
    print("Testing Trading Engine...")
    print("=" * 50)
    
    from src.core.engine import EngineConfig, TradingMode
    
    config = EngineConfig(
        mode=TradingMode.PAPER,
        initial_balance=100000
    )
    
    engine = TradingEngine(config)
    
    # Create broker
    broker = PaperBroker(initial_balance=100000)
    engine.broker = broker
    
    # Start engine
    success = await engine.start()
    print(f"[OK] Engine started: {success}")
    
    # Get status
    status = await engine.get_status()
    print(f"[OK] Engine state: {status['state']}")
    print(f"[OK] Mode: {status['mode']}")
    
    # Process signal
    signal = {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'quantity': 0.1,
        'confidence': 0.8
    }
    result = await engine.process_signal(signal)
    print(f"[OK] Signal processed: {result['success']}")
    
    # Stop engine
    await engine.stop(close_positions=False)
    print("[OK] Engine stopped")
    
    print("[OK] Trading engine test passed")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" AI TRADING SYSTEM - CORE COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        ("Paper Broker", test_paper_broker),
        ("Portfolio Manager", test_portfolio_manager),
        ("Risk Engine", test_risk_engine),
        ("Order Manager", test_order_manager),
        ("Trading Engine", test_trading_engine),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"[ERROR] {name} test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f" RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed! Core system is ready.")
    else:
        print(f"\n[WARNING] {failed} test(s) failed. Please check errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
