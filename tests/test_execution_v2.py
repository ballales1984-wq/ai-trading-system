"""
Tests for Institutional Execution Engine v2.0
"""

import asyncio
import unittest
from unittest.mock import MagicMock
from app.execution.broker_connector import PaperTradingConnector, BrokerOrder, OrderSide
from app.execution.execution_engine import ExecutionEngine
from app.execution.algo_execution import AlgoConfig, TWAPExecutor

class TestExecutionV2(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Setup mock brokers
        self.primary_broker = PaperTradingConnector(initial_balance=1000000.0)
        self.secondary_broker = PaperTradingConnector(initial_balance=1000000.0)
        
        # Force different prices to test SOR
        self.primary_broker.get_symbol_price = MagicMock(return_value=asyncio.Future())
        self.primary_broker.get_symbol_price.return_value.set_result(43500.0)
        
        self.secondary_broker.get_symbol_price = MagicMock(return_value=asyncio.Future())
        self.secondary_broker.get_symbol_price.return_value.set_result(43400.0) # Better price for BUY
        
        self.engine = ExecutionEngine(
            broker=self.primary_broker,
            additional_brokers=[self.secondary_broker]
        )
        await self.primary_broker.connect()
        await self.secondary_broker.connect()

    async def test_sor_routing(self):
        """Verify SOR selects the broker with the best price."""
        result = await self.engine.execute_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            use_sor=True
        )
        
        self.assertTrue(result.success)
        self.assertIn("Routed via SOR", result.message)
        # Should have routed to secondary_broker (43400 < 43500)
        self.assertEqual(result.avg_price, 43400.0)

    async def test_twap_execution(self):
        """Verify TWAP splits orders and executes over time."""
        # Use short duration for test
        result = await self.engine.execute_twap(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            duration_minutes=0, # Instant execution for test
            num_chunks=2
        )
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.filled_quantity, 0.1)
        self.assertIn("100.0% executed", result.message)

if __name__ == "__main__":
    unittest.main()
