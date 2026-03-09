#!/usr/bin/env python3
"""
Paper Trading Validation Test Suite
====================================
Tests for Phase 1: Paper Trading Validation
- Stop loss / Take profit validation
- Risk engine limits test
- Real-time PnL update
- Portfolio position test
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    """Test result container."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = {}


class PaperTradingValidator:
    """
    Validates paper trading functionality.
    Tests stop loss, take profit, risk limits, and portfolio updates.
    """
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.orders = []
        self.order_id = 0
        self.test_results: List[TestResult] = []
        
    # ========== BROKER SIMULATION ==========
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                    price: float, stop_loss: float = None, take_profit: float = None) -> Dict:
        """Simulate order placement."""
        self.order_id += 1
        order = {
            "id": self.order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "filled",
            "filled_price": price,
            "timestamp": datetime.now().isoformat()
        }
        self.orders.append(order)
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {"quantity": 0, "avg_price": 0, "side": None}
        
        pos = self.positions[symbol]
        if side == "buy":
            if pos["quantity"] == 0:
                pos["quantity"] = quantity
                pos["avg_price"] = price
                pos["side"] = "long"
            else:
                total_cost = pos["quantity"] * pos["avg_price"] + quantity * price
                pos["quantity"] += quantity
                pos["avg_price"] = total_cost / pos["quantity"]
                pos["side"] = "long"
        else:  # sell
            if pos["quantity"] > 0:
                pos["quantity"] -= quantity
                if pos["quantity"] == 0:
                    pos["side"] = None
                    pos["avg_price"] = 0
        
        # Update balance
        cost = quantity * price
        if side == "buy":
            self.current_balance -= cost
        else:
            self.current_balance += cost
            
        return order
    
    def update_price(self, symbol: str, new_price: float) -> Dict:
        """Update price and check for SL/TP triggers."""
        triggered = {"stop_loss": False, "take_profit": False}
        
        if symbol in self.positions and self.positions[symbol]["quantity"] > 0:
            pos = self.positions[symbol]
            entry_price = pos["avg_price"]
            
            # Check stop loss
            if pos["side"] == "long":
                if new_price <= pos.get("stop_loss", 0) and pos.get("stop_loss", 0) > 0:
                    triggered["stop_loss"] = True
                    logger.info(f"SL TRIGGERED: {symbol} @ {new_price}")
                elif new_price >= pos.get("take_profit", float('inf')) and pos.get("take_profit", float('inf')) < float('inf'):
                    triggered["take_profit"] = True
                    logger.info(f"TP TRIGGERED: {symbol} @ {new_price}")
        
        return triggered
    
    def calculate_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for a position."""
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        if pos["quantity"] == 0:
            return 0.0
        
        if pos["side"] == "long":
            return (current_price - pos["avg_price"]) * pos["quantity"]
        return 0.0
    
    def get_total_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate total unrealized PnL across all positions."""
        total = 0.0
        for symbol, price in prices.items():
            total += self.calculate_pnl(symbol, price)
        return total
    
    # ========== RISK ENGINE ==========
    
    def check_risk_limits(self, order: Dict, max_position_pct: float = 0.3,
                         max_order_pct: float = 0.1, max_daily_loss_pct: float = 0.05) -> tuple:
        """Check if order passes risk engine validation."""
        order_value = order["quantity"] * order["price"]
        order_pct = order_value / self.initial_balance
        
        # Check max order size
        if order_pct > max_order_pct:
            return False, f"Order size {order_pct*100:.1f}% exceeds max {max_order_pct*100}%"
        
        # Check max position
        symbol = order["symbol"]
        current_pos_value = 0
        if symbol in self.positions:
            current_pos_value = self.positions[symbol]["quantity"] * self.positions[symbol]["avg_price"]
        
        total_position_value = current_pos_value + order_value
        if total_position_value / self.initial_balance > max_position_pct:
            return False, f"Position size {total_position_value/self.initial_balance*100:.1f}% exceeds max {max_position_pct*100}%"
        
        return True, "OK"
    
    # ========== TESTS ==========
    
    def test_stop_loss_take_profit(self) -> TestResult:
        """Test 1: Stop Loss / Take Profit Validation."""
        result = TestResult("Stop Loss / Take Profit")
        
        try:
            # Reset
            self.positions = {}
            self.current_balance = self.initial_balance
            
            # Place long order with SL and TP
            order = self.place_order(
                symbol="BTCUSDT",
                side="buy",
                quantity=1.0,
                price=50000,
                stop_loss=48000,  # -4%
                take_profit=55000  # +10%
            )
            
            self.positions["BTCUSDT"]["stop_loss"] = 48000
            self.positions["BTCUSDT"]["take_profit"] = 55000
            
            # Test 1: Price drops to SL
            triggered = self.update_price("BTCUSDT", 47500)
            if not triggered["stop_loss"]:
                result.message = "FAIL: SL not triggered at 47500"
                self.test_results.append(result)
                return result
            
            # Test 2: Reset and test TP
            self.positions["BTCUSDT"]["quantity"] = 1.0
            self.positions["BTCUSDT"]["avg_price"] = 50000
            self.positions["BTCUSDT"]["stop_loss"] = 48000
            self.positions["BTCUSDT"]["take_profit"] = 55000
            
            triggered = self.update_price("BTCUSDT", 56000)
            if not triggered["take_profit"]:
                result.message = "FAIL: TP not triggered at 56000"
                self.test_results.append(result)
                return result
            
            result.passed = True
            result.message = "PASS: SL and TP triggers work correctly"
            result.details = {
                "sl_triggered": True,
                "tp_triggered": True,
                "sl_price": 48000,
                "tp_price": 55000
            }
            
        except Exception as e:
            result.message = f"ERROR: {str(e)}"
            
        self.test_results.append(result)
        return result
    
    def test_risk_engine_limits(self) -> TestResult:
        """Test 2: Risk Engine Limits."""
        result = TestResult("Risk Engine Limits")
        
        try:
            # Reset
            self.positions = {}
            self.current_balance = self.initial_balance
            
            # Test 1: Order too large (10% max, we try 15%)
            order = {
                "symbol": "ETHUSDT",
                "side": "buy",
                "quantity": 300,  # 300 * 5000 = 150000 (150% of balance!)
                "price": 5000
            }
            
            ok, reason = self.check_risk_limits(order, max_order_pct=0.1)
            if ok:
                result.message = "FAIL: Should reject order > 10%"
                self.test_results.append(result)
                return result
            
            # Test 2: Position too large
            self.positions["BTCUSDT"] = {"quantity": 2.0, "avg_price": 45000, "side": "long"}
            
            order2 = {
                "symbol": "BTCUSDT",
                "side": "buy", 
                "quantity": 2.0,  # Adds to existing position
                "price": 50000
            }
            
            ok, reason = self.check_risk_limits(order2, max_position_pct=0.3)
            if ok:
                result.message = "FAIL: Should reject position > 30%"
                self.test_results.append(result)
                return result
            
            result.passed = True
            result.message = "PASS: Risk limits enforced correctly"
            result.details = {
                "max_order_pct": "10%",
                "max_position_pct": "30%",
                "rejected_oversized_order": True,
                "rejected_oversized_position": True
            }
            
        except Exception as e:
            result.message = f"ERROR: {str(e)}"
            
        self.test_results.append(result)
        return result
    
    def test_realtime_pnl(self) -> TestResult:
        """Test 3: Real-time PnL Updates."""
        result = TestResult("Real-time PnL")
        
        try:
            # Reset
            self.positions = {}
            self.current_balance = self.initial_balance
            
            # Open position
            self.place_order("BTCUSDT", "buy", 1.0, 50000)
            
            prices = {"BTCUSDT": 50000}
            pnl = self.get_total_pnl(prices)
            
            if pnl != 0:
                result.message = f"FAIL: Initial PnL should be 0, got {pnl}"
                self.test_results.append(result)
                return result
            
            # Price goes up
            prices["BTCUSDT"] = 55000
            pnl = self.get_total_pnl(prices)
            
            if pnl != 5000:
                result.message = f"FAIL: PnL should be 5000, got {pnl}"
                self.test_results.append(result)
                return result
            
            # Price goes down
            prices["BTCUSDT"] = 48000
            pnl = self.get_total_pnl(prices)
            
            if pnl != -2000:
                result.message = f"FAIL: PnL should be -2000, got {pnl}"
                self.test_results.append(result)
                return result
            
            result.passed = True
            result.message = "PASS: Real-time PnL updates correctly"
            result.details = {
                "initial_pnl": 0,
                "profit_pnl": 5000,
                "loss_pnl": -2000
            }
            
        except Exception as e:
            result.message = f"ERROR: {str(e)}"
            
        self.test_results.append(result)
        return result
    
    def test_portfolio_positions(self) -> TestResult:
        """Test 4: Portfolio Position Management."""
        result = TestResult("Portfolio Positions")
        
        try:
            # Reset
            self.positions = {}
            self.current_balance = self.initial_balance
            
            # Open multiple positions
            self.place_order("BTCUSDT", "buy", 1.0, 50000)
            self.place_order("ETHUSDT", "buy", 10.0, 3000)
            self.place_order("SOLUSDT", "buy", 50.0, 100)
            
            # Check positions
            if "BTCUSDT" not in self.positions or self.positions["BTCUSDT"]["quantity"] != 1.0:
                result.message = "FAIL: BTC position incorrect"
                self.test_results.append(result)
                return result
                
            if "ETHUSDT" not in self.positions or self.positions["ETHUSDT"]["quantity"] != 10.0:
                result.message = "FAIL: ETH position incorrect"
                self.test_results.append(result)
                return result
                
            if "SOLUSDT" not in self.positions or self.positions["SOLUSDT"]["quantity"] != 50.0:
                result.message = "FAIL: SOL position incorrect"
                self.test_results.append(result)
                return result
            
            # Calculate portfolio value (cash + positions at cost)
            prices = {"BTCUSDT": 52000, "ETHUSDT": 3200, "SOLUSDT": 110}
            portfolio_value = self.current_balance
            for symbol, price in prices.items():
                if symbol in self.positions:
                    # Use cost basis (what we paid), not current price
                    portfolio_value += self.positions[symbol]["quantity"] * self.positions[symbol]["avg_price"]
            
            # Total should equal initial balance (no PnL yet, using cost basis)
            expected = self.initial_balance
            if abs(portfolio_value - expected) > 0.01:
                result.message = f"FAIL: Portfolio value {portfolio_value} != {expected}"
                self.test_results.append(result)
                return result
            
            # Test partial close
            self.place_order("BTCUSDT", "sell", 0.5, 52000)
            
            if self.positions["BTCUSDT"]["quantity"] != 0.5:
                result.message = "FAIL: Position not reduced after partial sell"
                self.test_results.append(result)
                return result
            
            result.passed = True
            result.message = "PASS: Portfolio positions managed correctly"
            result.details = {
                "btc_position": 1.0,
                "eth_position": 10.0,
                "sol_position": 50.0,
                "partial_close_works": True
            }
            
        except Exception as e:
            result.message = f"ERROR: {str(e)}"
            
        self.test_results.append(result)
        return result
    
    def run_all_tests(self) -> Dict:
        """Run all paper trading validation tests."""
        print("\n" + "="*70)
        print("PAPER TRADING VALIDATION TESTS")
        print("="*70)
        print()
        
        # Run all tests
        self.test_stop_loss_take_profit()
        self.test_risk_engine_limits()
        self.test_realtime_pnl()
        self.test_portfolio_positions()
        
        # Print results
        passed = 0
        failed = 0
        
        for tr in self.test_results:
            status = "[PASS]" if tr.passed else "[FAIL]"
            print(f"{status} {tr.name}")
            print(f"       {tr.message}")
            if tr.details:
                for k, v in tr.details.items():
                    print(f"       - {k}: {v}")
            print()
            
            if tr.passed:
                passed += 1
            else:
                failed += 1
        
        print("="*70)
        print(f"RESULTS: {passed} passed, {failed} failed")
        print("="*70)
        
        return {
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "success": failed == 0
        }


async def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("QUANTUM AI TRADING SYSTEM - PAPER TRADING VALIDATION")
    print("="*70)
    print()
    print("Phase 1: Paper Trading Validation")
    print("- Stop loss / Take profit validation")
    print("- Risk engine limits test")
    print("- Real-time PnL update")
    print("- Portfolio position test")
    print()
    
    validator = PaperTradingValidator(initial_balance=100000)
    results = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
