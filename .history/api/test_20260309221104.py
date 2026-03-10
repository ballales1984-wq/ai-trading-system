"""Minimal Vercel Python test - no dependencies."""

def handler(event, context):
    """Simple Vercel handler returning JSON."""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": '{"status": "ok"}'
    }
