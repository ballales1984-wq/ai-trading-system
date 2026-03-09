#!/usr/bin/env python3
"""
run_live.py — Live Trading Launcher
====================================
Entry point for running the AI Trading System in live mode.

Supports:
  - Paper trading (default, no real money)
  - Live trading (Binance, Bybit, OKX via --broker)
  - Multi-asset portfolio mode (--multi)
  - Single asset mode (--symbol BTCUSDT)

Usage:
  python run_live.py                          # paper trading, BTC
  python run_live.py --broker binance --live  # live on Binance
  python run_live.py --multi --symbols BTCUSDT ETHUSDT SOLUSDT
  python run_live.py --dashboard              # also start dashboard API
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/live_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger("run_live")

# ---------------------------------------------------------------------------
# Ensure logs directory exists
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_event = asyncio.Event()


def _handle_signal(sig, frame):
    logger.warning(f"Received signal {sig}, shutting down gracefully...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
async def run_single_asset(args):
    """Run live trading on a single asset."""
    from src.live_trading import LiveTrader

    trader = LiveTrader(
        symbol=args.symbol,
        exchange=args.broker,
        testnet=not args.live,
        interval=args.interval,
    )

    logger.info(
        f"Starting single-asset live trading: {args.symbol} on {args.broker} "
        f"({'LIVE' if args.live else 'PAPER/TESTNET'})"
    )

    try:
        await trader.start()
        # Keep running until shutdown signal
        await _shutdown_event.wait()
    finally:
        await trader.stop()
        logger.info("Single-asset trader stopped.")


async def run_multi_asset(args):
    """Run live trading on multiple assets."""
    from live_multi_asset import MultiAssetTrader

    symbols = args.symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    trader = MultiAssetTrader(
        symbols=symbols,
        exchange=args.broker,
        testnet=not args.live,
        interval=args.interval,
    )

    logger.info(
        f"Starting multi-asset live trading: {symbols} on {args.broker} "
        f"({'LIVE' if args.live else 'PAPER/TESTNET'})"
    )

    try:
        await trader.start()
        await _shutdown_event.wait()
    finally:
        await trader.stop()
        logger.info("Multi-asset trader stopped.")


async def run_auto_trader(args):
    """Run the full auto-trader with decision engine."""
    from auto_trader import AutoTrader

    trader = AutoTrader(
        symbols=args.symbols or ["BTCUSDT"],
        exchange=args.broker,
        testnet=not args.live,
    )

    logger.info(
        f"Starting auto-trader on {args.broker} "
        f"({'LIVE' if args.live else 'PAPER/TESTNET'})"
    )

    try:
        await trader.run()
        await _shutdown_event.wait()
    finally:
        logger.info("Auto-trader stopped.")


async def run_dashboard_api():
    """Start the FastAPI dashboard in background."""
    import uvicorn
    from app.main import app

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main(args):
    """Main async entry point."""
    tasks = []

    # Optionally start dashboard API
    if args.dashboard:
        tasks.append(asyncio.create_task(run_dashboard_api()))
        logger.info("Dashboard API starting on port %s", os.getenv("API_PORT", "8000"))

    # Select trading mode
    if args.auto:
        tasks.append(asyncio.create_task(run_auto_trader(args)))
    elif args.multi:
        tasks.append(asyncio.create_task(run_multi_asset(args)))
    else:
        tasks.append(asyncio.create_task(run_single_asset(args)))

    # Wait for all tasks (shutdown event will stop them)
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Tasks cancelled.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Trading System — Live Trading Launcher"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Multiple symbols for multi-asset mode",
    )
    parser.add_argument(
        "--broker",
        type=str,
        default="paper",
        choices=["paper", "binance", "bybit", "okx"],
        help="Broker/exchange to use (default: paper)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Candle interval (default: 1h)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable LIVE trading (real money). Without this flag, uses testnet/paper.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        default=False,
        help="Enable multi-asset portfolio mode",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Enable full auto-trader with decision engine",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Also start the FastAPI dashboard API",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Safety check for live trading
    if args.live:
        logger.warning("=" * 60)
        logger.warning("  ⚠️  LIVE TRADING MODE — REAL MONEY AT RISK  ⚠️")
        logger.warning("=" * 60)
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm.strip() != "YES":
            logger.info("Live trading cancelled.")
            sys.exit(0)

    asyncio.run(main(args))
