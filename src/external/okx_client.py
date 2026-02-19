"""
OKX API Client
==============
Client for OKX (OKEx) exchange API.
"""

import asyncio
import hashlib
import hmac
import base64
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import logging
import json

logger = logging.getLogger(__name__)


class OKXEnvironment(Enum):
    PRODUCTION = "https://www.okx.com"
    TESTNET = "https://www.okx.com"


@dataclass
class OKXTicker:
    inst_id: str
    last_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    timestamp: datetime = field(default_factory=datetime.now)


class OKXClient:
    """
    OKX API Client
    """
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = False,
        timeout: int = 30
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        self.timeout = timeout
        self.base_url = "https://www.okx.com"
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"OKX client initialized (testnet={testnet})")
    
    async def __aenter__(self):
        await self.connect()
        return self
    
async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        if self._session is None:
            # Force use of threaded resolver instead of aiodns (fixes Windows DNS issues)
            try:
                import aiodns
                # If aiodns is installed, force using the threaded resolver
                resolver = aiohttp.ThreadedResolver()
            except ImportError:
                # aiodns not installed, use default
                resolver = aiohttp.DefaultResolver()
            
            connector = aiohttp.TCPConnector(
                limit=100,
                resolver=resolver
            )
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=connector
            )
    
    async def disconnect(self):
        if self._session:
            await self._session.close()
            self._session = None
    
    async def get_ticker(self, inst_id: str) -> OKXTicker:
        params = {"instId": inst_id}
        data = await self._request("GET", "/api/v5/market/ticker", params)
        
        if data:
            t = data[0]
            return OKXTicker(
                inst_id=t.get("instId", inst_id),
                last_price=float(t.get("last", 0)),
                bid_price=float(t.get("bidPx", 0)),
                ask_price=float(t.get("askPx", 0)),
                volume_24h=float(t.get("vol24h", 0))
            )
        raise ValueError(f"No ticker for {inst_id}")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(params) if params else ""
        
        async with self._session.request(method, url, data=body) as response:
            data = await response.json()
            if data.get("code") == "0":
                return data.get("data", [])
            raise Exception(data.get("msg", "Error"))
    
    async def get_price(self, inst_id: str) -> float:
        ticker = await self.get_ticker(inst_id)
        return ticker.last_price


def create_okx_client(
    api_key: str = "",
    api_secret: str = "",
    passphrase: str = "",
    testnet: bool = False
) -> OKXClient:
    return OKXClient(api_key, api_secret, passphrase, testnet)


async def test():
    client = OKXClient()
    async with client:
        ticker = await client.get_ticker("BTC-USDT")
        print(f"BTC: ${ticker.last_price}")

if __name__ == "__main__":
    asyncio.run(test())

