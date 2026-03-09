"""Demo mode configuration and utilities."""

import os

# Demo mode configuration - set to True for demo mode by default
# Can be overridden by environment variable DEMO_MODE
_demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"


def get_demo_mode() -> bool:
    """Get current DEMO_MODE setting."""
    return _demo_mode


def set_demo_mode(value: bool) -> None:
    """Set DEMO_MODE at runtime."""
    global _demo_mode
    _demo_mode = value