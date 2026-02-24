#!/usr/bin/env python3
"""
Desktop launcher for AI Trading System.

Starts the local FastAPI server and opens the dashboard in the default browser.
This file is intended to be packaged into a Windows executable with PyInstaller.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser
from contextlib import closing
from typing import Optional

import uvicorn


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _wait_for_server(host: str, port: int, timeout_seconds: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if _is_port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading System Desktop Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host for local API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for local API server")
    parser.add_argument(
        "--path",
        default="/dashboard",
        help="Route to open in the browser (default: /dashboard)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start server without opening browser automatically",
    )
    return parser


def run() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    if not _wait_for_server(args.host, args.port, timeout_seconds=30):
        print(f"[ERROR] Local server not reachable on http://{args.host}:{args.port}")
        return 1

    app_url = f"http://{args.host}:{args.port}{args.path}"
    print(f"[INFO] Server running at http://{args.host}:{args.port}")
    print(f"[INFO] Dashboard: {app_url}")

    if not args.no_browser:
        webbrowser.open(app_url)

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        server.should_exit = True
        server_thread.join(timeout=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
