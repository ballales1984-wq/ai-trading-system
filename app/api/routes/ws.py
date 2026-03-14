"""
WebSocket Routes
================
WebSocket endpoints for real-time data streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List

# pyre-ignore[21]: Missing module attribute
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# pyre-ignore[21]: Missing module attribute
from app.api.mock_data import get_market_prices, get_portfolio_summary
# pyre-ignore[21]: Missing module attribute
from app.core.demo_mode import get_demo_mode

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    """Manages WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                # We'll clean up disconnected ones in the next disconnect call
                # or if the send fails consistently.

manager = ConnectionManager()

@router.websocket("/prices")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time prices and portfolio updates.
    Exposed at /ws/prices to match frontend expectation.
    """
    await manager.connect(websocket)
    
    try:
        # Initial data send
        if get_demo_mode():
            await websocket.send_json({
                "type": "price_update",
                "data": get_market_prices()
            })
            await websocket.send_json({
                "type": "portfolio_update",
                "data": get_portfolio_summary()
            })

        # Keep connection alive and send periodic updates
        while True:
            # Wait for any message from client (can be used for heartbeats or subscriptions)
            # using data = await websocket.receive_text() but we'll just use a timeout/sleep loop
            
            if get_demo_mode():
                # Send price updates every 5 seconds
                await manager.broadcast({
                    "type": "price_update",
                    "data": get_market_prices()
                })
                # Send portfolio updates every 10 seconds
                await manager.broadcast({
                    "type": "portfolio_update",
                    "data": get_portfolio_summary()
                })
            
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
