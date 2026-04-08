"""
Smart Order Routing (SOR)
=========================
Orchestrates orders across multiple brokers to achieve Best Execution.
"""

import logging
from typing import List, Dict, Optional, Tuple
from app.execution.broker_connector import BrokerConnector, BrokerOrder, OrderSide

logger = logging.getLogger(__name__)

class SmartOrderRouter:
    """
    Selects the best broker(s) for a given order based on real-time prices 
    and available liquidity.
    """
    
    def __init__(self, brokers: List[BrokerConnector]):
        self.brokers = brokers
        if not brokers:
            raise ValueError("SmartOrderRouter requires at least one broker.")

    async def get_best_price(self, symbol: str, side: OrderSide) -> Tuple[BrokerConnector, float]:
        """
        Queries all brokers for the best price.
        Returns (Broker, Price).
        """
        best_broker = None
        best_price = float('inf') if side == OrderSide.BUY else 0.0
        
        for broker in self.brokers:
            try:
                price = await broker.get_symbol_price(symbol)
                
                if side == OrderSide.BUY:
                    if price < best_price:
                        best_price = price
                        best_broker = broker
                else:  # SELL
                    if price > best_price:
                        best_price = price
                        best_broker = broker
            except Exception as e:
                logger.error(f"Failed to get price from {broker.__class__.__name__}: {e}")
                
        if not best_broker:
            # Fallback to first broker if price discovery failed
            return self.brokers[0], await self.brokers[0].get_symbol_price(symbol)
            
        return best_broker, best_price

    async def route_order(self, order: BrokerOrder) -> BrokerOrder:
        """
        Routes the order to the single best broker found at execution time.
        """
        side = OrderSide(order.side.upper())
        best_broker, price = await self.get_best_price(order.symbol, side)
        
        logger.info(f"SOR: Routing {order.symbol} {order.side} to {best_broker.__class__.__name__} at {price}")
        
        # update order with the selected broker connector logic if needed
        # in this architecture, we let the broker connector handle the internal ID
        return await best_broker.place_order(order)

    async def split_route_order(self, order: BrokerOrder) -> List[BrokerOrder]:
        """
        Optional: Split order across all brokers proportionally?
        Usually better for very large orders to minimize market impact.
        For now, we stick to the 'Best Venue' routing.
        """
        # Placeholder for future multi-venue splitting logic
        executed_order = await self.route_order(order)
        return [executed_order]
