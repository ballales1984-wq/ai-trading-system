#!/usr/bin/env python3
"""Script to add OrderSide class to broker_connector.py"""

# Read the file
with open('app/execution/broker_connector.py', 'r') as f:
    content = f.read()

# Define old and new text
old_text = '''class OrderStatus(str, Enum):
    """Broker order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# Alias for backward compatibility'''

new_text = '''class OrderStatus(str, Enum):
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

# Replace
if old_text in content:
    content = content.replace(old_text, new_text)
    with open('app/execution/broker_connector.py', 'w') as f:
        f.write(content)
    print("SUCCESS: OrderSide added")
else:
    print("ERROR: Could not find the text to replace")
