"""
Live Portfolio Manager with WebSocket
=====================================
Integrates WebSocket streaming with PortfolioManager for real-time price updates.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime