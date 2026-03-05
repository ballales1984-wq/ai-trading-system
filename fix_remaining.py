#!/usr/bin/env python3
"""Fix remaining test issues"""

# Fix 1: Add OrderSide to broker_connector.py
print("Fixing broker_connector.py...")
with open('app/execution/broker_connector.py', 'r') as f:
    content = f.read()

old = '''class OrderStatus(str, Enum):
    """Broker order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# Alias for backward compatibility'''

new = '''class OrderStatus(str, Enum):
    """Broker order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(str, Enum):
    """Order side (BUY/SELL)."""
    BUY = "BUY"
    SELL = "SELL"


# Alias for backward compatibility'''

if old in content:
    content = content.replace(old, new)
    with open('app/execution/broker_connector.py', 'w') as f:
        f.write(content)
    print("  OrderSide added")
else:
    print("  OrderSide already exists or pattern not found")

# Fix 2: Add SignalHistory to timescale_models.py
print("Fixing timescale_models.py...")
with open('app/database/timescale_models.py', 'r') as f:
    content = f.read()

# Add SignalHistory alias
if 'SignalHistory' not in content:
    old = '''# Alias for backward compatibility
OHLCVCandle = OHLCVBar
TickData = TradeTick'''
    new = '''# Alias for backward compatibility
OHLCVCandle = OHLCVBar
TickData = TradeTick
SignalHistory = SignalTimeseries'''
    if old in content:
        content = content.replace(old, new)
        with open('app/database/timescale_models.py', 'w') as f:
            f.write(content)
        print("  SignalHistory added")
    else:
        print("  Could not find pattern for SignalHistory")
else:
    print("  SignalHistory already exists")

# Fix 3: Add mock data functions
print("Fixing mock_data.py...")
with open('app/api/mock_data.py', 'r') as f:
    content = f.read()

# Add MOCK_PRICES and functions if not present
if 'MOCK_PRICES' not in content:
    # Find the __all__ and add exports
    old_all = '''__all__ = [
    "get_news",
]'''
    new_all = '''__all__ = [
    "get_news",
    # Backward compatibility
    "MOCK_PRICES",
    "get_mock_ticker",
    "get_mock_orderbook",
    "get_mock_ohlcv",
]'''
    content = content.replace(old_all, new_all)
    
    # Add the functions and aliases at the end
    mock_code = '''

# Backward compatibility aliases
MOCK_PRICES = BASE_PRICES


def get_mock_ticker(symbol: str):
    """Get mock ticker data."""
    return get_price_data(symbol)


def get_mock_orderbook(symbol: str, depth: int = 10):
    """Get mock orderbook data."""
    import random
    ticker = get_price_data(symbol)
    if "error" in ticker:
        return ticker
    
    price = ticker.get("price", 100.0)
    
    bids = []
    asks = []
    for i in range(depth):
        bid_price = price * (1 - 0.001 * (i + 1))
        ask_price = price * (1 + 0.001 * (i + 1))
        bids.append([round(bid_price, 2), round(random.uniform(100, 10000), 2)])
        asks.append([round(ask_price, 2), round(random.uniform(100, 10000), 2)])
    
    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
    }


def get_mock_ohlcv(symbol: str, interval: str = "1h", limit: int = 100):
    """Get mock OHLCV data."""
    return get_candle_data(symbol, interval, limit)
'''
    content = content.rstrip() + mock_code
    
    with open('app/api/mock_data.py', 'w') as f:
        f.write(content)
    print("  Mock data functions added")
else:
    print("  MOCK_PRICES already exists")

# Fix 4: Add roles property to RBACManager
print("Fixing rbac.py...")
with open('app/core/rbac.py', 'r') as f:
    content = f.read()

if 'def roles' not in content and 'property\n    def roles' not in content:
    # Add roles property to RBACManager class
    old = '''    def get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role."""
        return self._roles.get(role, [])'''
    new = '''    def get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role."""
        return self._roles.get(role, [])

    @property
    def roles(self) -> List[str]:
        """Get list of all roles."""
        return list(self._roles.keys())

    def get_roles(self) -> List[str]:
        """Get list of all roles (alias for roles property)."""
        return self.roles'''
    if old in content:
        content = content.replace(old, new)
        with open('app/core/rbac.py', 'w') as f:
            f.write(content)
        print("  roles property added")
    else:
        print("  Could not find pattern for roles")
else:
    print("  roles already exists")

print("\nDone! Please run the tests again.")
