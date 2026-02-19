"""
Hedge Fund Trading System - Professional Architecture
======================================================

Multi-asset, multi-strategy trading system with institutional-grade
risk management, order execution, and portfolio optimization.

Author: AI Trading System
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "Hedge Fund Trading Team"

from app.core.config import settings
from app.core.logging import setup_logging

__all__ = ["settings", "setup_logging"]
